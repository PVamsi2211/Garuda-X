from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import Callable

from src.audit.audit_logger import AuditLogger
from src.decision.errors import (
    AlreadyAuthorizedError,
    AlreadyFinalizedError,
    ApprovalRequiredError,
    ExecutionNotAuthorizedError,
    InvalidDecisionError,
    InvalidTransitionError,
    ValidationError,
)
from src.decision.recommendation import Recommendation, build_recommendation


_FINAL_STATUSES = frozenset({"APPROVED", "MODIFIED", "REJECTED"})
_HUMAN_PLACEHOLDERS = frozenset({"ai", "system", "anonymous", "assistant", "automated", "bot"})


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _human_actor(value: object) -> str:
    actor = _required_text(value, "approver_id")
    if actor.casefold() in _HUMAN_PLACEHOLDERS:
        raise ValidationError("approver_id must identify a human reviewer")
    return actor


class DecisionService:
    def __init__(
        self,
        *,
        audit_logger: AuditLogger | None = None,
        clock: Callable[[], datetime] | None = None,
        decision_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._decisions: dict[str, Recommendation] = {}
        self._audit = audit_logger or AuditLogger(clock=clock)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._decision_id_factory = decision_id_factory or (lambda: str(uuid.uuid4()))
        self._lock = RLock()

    def _now(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValidationError("Decision clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc).isoformat()

    def create_recommendation(self, **inputs: object) -> dict[str, object]:
        with self._lock:
            if "decision_id" not in inputs or inputs["decision_id"] is None:
                inputs["decision_id"] = self._decision_id_factory()
            if "generated_at" not in inputs or inputs["generated_at"] is None:
                inputs["generated_at"] = self._now()
            decision = build_recommendation(**inputs)
            if decision.decision_id in self._decisions:
                raise ValidationError(f"Duplicate decision_id: {decision.decision_id}")
            self._audit.record_recommendation(decision.to_dict(), timestamp=decision.generated_at)
            self._decisions[decision.decision_id] = decision
            return decision.to_dict()

    def _get_current(self, decision_id: object) -> Recommendation:
        identifier = _required_text(decision_id, "decision_id")
        current = self._decisions.get(identifier)
        if current is None:
            raise InvalidDecisionError(f"Unknown decision_id: {identifier}")
        if current.status not in {"PENDING", "APPROVED", "MODIFIED", "REJECTED"}:
            raise InvalidTransitionError(f"Unknown decision status: {current.status}")
        return current

    def _require_pending(self, current: Recommendation) -> None:
        if current.status != "PENDING":
            if current.status in _FINAL_STATUSES:
                raise AlreadyFinalizedError(
                    f"Decision {current.decision_id} is already finalized as {current.status}"
                )
            raise InvalidTransitionError(f"Cannot transition from {current.status}")
        if not current.requires_human_approval:
            raise ApprovalRequiredError("Decision record is missing the mandatory approval gate")

    def approve_decision(self, decision_id: str, approver_id: str) -> dict[str, object]:
        actor = _human_actor(approver_id)
        with self._lock:
            current = self._get_current(decision_id)
            self._require_pending(current)
            timestamp = self._now()
            proposed = replace(
                current,
                status="APPROVED",
                final_action=current.original_recommended_action,
                approver_id=actor,
                status_updated_at=timestamp,
            )
            event = self._audit.record_transition(
                decision=current.to_dict(),
                event_type="DECISION_APPROVED",
                previous_status="PENDING",
                new_status="APPROVED",
                approver_id=actor,
                final_action=proposed.final_action,
                timestamp=timestamp,
            )
            updated = replace(proposed, approval_event_id=str(event["event_id"]))
            self._decisions[current.decision_id] = updated
            return updated.to_dict()

    def modify_decision(
        self,
        decision_id: str,
        approver_id: str,
        final_action: str,
        reason: str,
    ) -> dict[str, object]:
        actor = _human_actor(approver_id)
        action = _required_text(final_action, "final_action")
        modification_reason = _required_text(reason, "reason")
        with self._lock:
            current = self._get_current(decision_id)
            self._require_pending(current)
            timestamp = self._now()
            proposed = replace(
                current,
                status="MODIFIED",
                final_action=action,
                approver_id=actor,
                modification_reason=modification_reason,
                status_updated_at=timestamp,
            )
            event = self._audit.record_transition(
                decision=current.to_dict(),
                event_type="DECISION_MODIFIED",
                previous_status="PENDING",
                new_status="MODIFIED",
                approver_id=actor,
                final_action=action,
                reason=modification_reason,
                timestamp=timestamp,
            )
            updated = replace(proposed, approval_event_id=str(event["event_id"]))
            self._decisions[current.decision_id] = updated
            return updated.to_dict()

    def reject_decision(self, decision_id: str, approver_id: str, reason: str) -> dict[str, object]:
        actor = _human_actor(approver_id)
        rejection_reason = _required_text(reason, "reason")
        with self._lock:
            current = self._get_current(decision_id)
            self._require_pending(current)
            timestamp = self._now()
            updated = replace(
                current,
                status="REJECTED",
                approver_id=actor,
                rejection_reason=rejection_reason,
                status_updated_at=timestamp,
            )
            self._audit.record_transition(
                decision=current.to_dict(),
                event_type="DECISION_REJECTED",
                previous_status="PENDING",
                new_status="REJECTED",
                approver_id=actor,
                final_action=None,
                reason=rejection_reason,
                timestamp=timestamp,
            )
            self._decisions[current.decision_id] = updated
            return updated.to_dict()

    def authorize_execution(self, decision_id: str) -> dict[str, object]:
        with self._lock:
            current = self._get_current(decision_id)
            if current.status not in {"APPROVED", "MODIFIED"}:
                raise ExecutionNotAuthorizedError(
                    "Only approved or modified decisions may receive execution authorization"
                )
            if current.execution_authorized:
                raise AlreadyAuthorizedError("Execution authorization already exists")
            if not current.requires_human_approval:
                raise ApprovalRequiredError("Decision record is missing the mandatory approval gate")
            if not current.approver_id or not current.approval_event_id:
                raise ExecutionNotAuthorizedError("A valid human approval event is required")
            if not isinstance(current.final_action, str) or not current.final_action.strip():
                raise ExecutionNotAuthorizedError("A valid final action is required")
            approval_events = self._audit.get_history(current.decision_id)
            valid_event_type = "DECISION_APPROVED" if current.status == "APPROVED" else "DECISION_MODIFIED"
            matching = [event for event in approval_events if event.get("event_id") == current.approval_event_id]
            if (
                len(matching) != 1
                or matching[0].get("event_type") != valid_event_type
                or matching[0].get("approver_id") != current.approver_id
                or matching[0].get("new_status") != current.status
                or matching[0].get("previous_status") != "PENDING"
                or matching[0].get("original_recommendation") != current.original_recommended_action
                or matching[0].get("final_action") != current.final_action
            ):
                raise ExecutionNotAuthorizedError("Approval audit record is missing or inconsistent")
            timestamp = self._now()
            event = self._audit.record_transition(
                decision=current.to_dict(),
                event_type="EXECUTION_AUTHORIZED",
                previous_status=current.status,
                new_status="EXECUTION_AUTHORIZED",
                approver_id=current.approver_id,
                final_action=current.final_action,
                reason=None,
                metadata={"authorization_only": True},
                timestamp=timestamp,
            )
            if event.get("decision_id") != current.decision_id:
                raise ExecutionNotAuthorizedError("Execution audit record is inconsistent")
            updated = replace(
                current,
                execution_authorized=True,
                execution_authorized_at=timestamp,
            )
            self._decisions[current.decision_id] = updated
            return updated.to_dict()

    def is_execution_authorized(self, decision_id: str) -> bool:
        with self._lock:
            current = self._get_current(decision_id)
            if (
                current.status not in {"APPROVED", "MODIFIED"}
                or not current.execution_authorized
                or not current.requires_human_approval
                or not current.approver_id
                or not current.approval_event_id
                or not isinstance(current.final_action, str)
                or not current.final_action.strip()
            ):
                return False
            history = self._audit.get_history(current.decision_id)
            approval_type = "DECISION_APPROVED" if current.status == "APPROVED" else "DECISION_MODIFIED"
            approval_events = [
                event for event in history
                if event.get("event_id") == current.approval_event_id
            ]
            authorization_events = [
                event for event in history
                if event.get("event_type") == "EXECUTION_AUTHORIZED"
            ]
            if len(approval_events) != 1 or len(authorization_events) != 1:
                return False
            approval = approval_events[0]
            authorization = authorization_events[0]
            metadata = authorization.get("metadata")
            return (
                approval.get("event_type") == approval_type
                and approval.get("previous_status") == "PENDING"
                and approval.get("new_status") == current.status
                and approval.get("approver_id") == current.approver_id
                and approval.get("original_recommendation") == current.original_recommended_action
                and approval.get("final_action") == current.final_action
                and authorization.get("previous_status") == current.status
                and authorization.get("new_status") == "EXECUTION_AUTHORIZED"
                and authorization.get("approver_id") == current.approver_id
                and authorization.get("original_recommendation") == current.original_recommended_action
                and authorization.get("final_action") == current.final_action
                and isinstance(metadata, dict)
                and metadata.get("authorization_only") is True
            )

    def get_decision(self, decision_id: str) -> dict[str, object]:
        with self._lock:
            return self._get_current(decision_id).to_dict()

    def list_decisions(self) -> list[dict[str, object]]:
        with self._lock:
            return [decision.to_dict() for decision in self._decisions.values()]

    def get_pending_decisions(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                decision.to_dict()
                for decision in self._decisions.values()
                if decision.status == "PENDING"
            ]

    def get_accounts(self) -> list[dict[str, object]]:
        with self._lock:
            accounts: dict[tuple[object | None, object | None], dict[str, object]] = {}
            for decision in self._decisions.values():
                key = (decision.account_id, decision.user_id)
                account = accounts.setdefault(key, {
                    "account_id": decision.account_id,
                    "user_id": decision.user_id,
                    "latest_decision_id": decision.decision_id,
                    "decision_count": 0,
                })
                account["latest_decision_id"] = decision.decision_id
                account["decision_count"] = int(account["decision_count"]) + 1
            return [dict(account) for account in accounts.values()]

    def get_decision_history(self, decision_id: str) -> list[dict[str, object]]:
        with self._lock:
            self._get_current(decision_id)
            return self._audit.get_history(decision_id)

    def get_audit_history(self, decision_id: str | None = None) -> list[dict[str, object]]:
        with self._lock:
            if decision_id is None:
                return self._audit.list_events()
            self._get_current(decision_id)
            return self._audit.get_history(decision_id)
