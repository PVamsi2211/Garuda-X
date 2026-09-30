from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Mapping

from src.decision.errors import AuditIntegrityError, ValidationError


AUDIT_EVENT_TYPES = frozenset({
    "RECOMMENDATION_CREATED",
    "DECISION_APPROVED",
    "DECISION_MODIFIED",
    "DECISION_REJECTED",
    "EXECUTION_AUTHORIZED",
})


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    decision_id: str
    account_id: object | None
    user_id: object | None
    timestamp: str
    event_type: str
    previous_status: str | None
    new_status: str
    approver_id: str | None
    original_recommendation: str
    final_action: str | None
    reason: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["metadata"] = copy.deepcopy(self.metadata)
        return result


def _utc_timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError("Audit timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


class AuditLogger:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._status_by_decision: dict[str, str] = {}
        self._last_timestamp: datetime | None = None
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()

    def _timestamp(self, value: str | datetime | None) -> str:
        stamp = _utc_timestamp(value if value is not None else self._clock())
        parsed = datetime.fromisoformat(stamp)
        if self._last_timestamp is not None and parsed < self._last_timestamp:
            raise AuditIntegrityError("Audit timestamps must remain chronological")
        return stamp

    def record_recommendation(self, decision: Mapping[str, object], *, timestamp: str | datetime | None = None) -> dict[str, object]:
        decision_id = decision.get("decision_id")
        original_action = decision.get("original_recommended_action")
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValidationError("A real decision_id is required for audit logging")
        if not isinstance(original_action, str) or not original_action.strip():
            raise ValidationError("A recommendation action is required for audit logging")
        with self._lock:
            if decision_id in self._status_by_decision:
                raise AuditIntegrityError(f"Decision already registered: {decision_id}")
            stamp = self._timestamp(timestamp)
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                decision_id=decision_id,
                account_id=decision.get("account_id"),
                user_id=decision.get("user_id"),
                timestamp=stamp,
                event_type="RECOMMENDATION_CREATED",
                previous_status=None,
                new_status="PENDING",
                approver_id=None,
                original_recommendation=original_action,
                final_action=None,
                reason=None,
                metadata={
                    "playbook_id": decision.get("playbook_id"),
                    "playbook_version": decision.get("playbook_version"),
                    "risk_band": decision.get("risk_band"),
                    "revenue_band": decision.get("revenue_band"),
                },
            )
            self._events.append(event)
            self._status_by_decision[decision_id] = "PENDING"
            self._last_timestamp = datetime.fromisoformat(stamp)
            return event.to_dict()

    def record_transition(
        self,
        *,
        decision: Mapping[str, object],
        event_type: str,
        previous_status: str,
        new_status: str,
        approver_id: str | None,
        final_action: str | None,
        reason: str | None = None,
        metadata: Mapping[str, object] | None = None,
        timestamp: str | datetime | None = None,
    ) -> dict[str, object]:
        decision_id = decision.get("decision_id")
        original_action = decision.get("original_recommended_action")
        if event_type not in AUDIT_EVENT_TYPES - {"RECOMMENDATION_CREATED"}:
            raise ValidationError(f"Unknown audit event type: {event_type}")
        if not isinstance(decision_id, str) or not isinstance(original_action, str):
            raise ValidationError("Audit transition requires a valid decision")
        expected = {
            "DECISION_APPROVED": ("PENDING", "APPROVED"),
            "DECISION_MODIFIED": ("PENDING", "MODIFIED"),
            "DECISION_REJECTED": ("PENDING", "REJECTED"),
            "EXECUTION_AUTHORIZED": (("APPROVED", "MODIFIED"), "EXECUTION_AUTHORIZED"),
        }[event_type]
        if previous_status not in ((expected[0],) if isinstance(expected[0], str) else expected[0]) or new_status != expected[1]:
            raise AuditIntegrityError(f"Incorrect status transition for {event_type}")
        with self._lock:
            current = self._status_by_decision.get(decision_id)
            if current is None:
                raise AuditIntegrityError(f"Unknown decision_id: {decision_id}")
            if current != previous_status:
                raise AuditIntegrityError(
                    f"Audit status mismatch for {decision_id}: expected {current}, received {previous_status}"
                )
            if not isinstance(approver_id, str) or not approver_id.strip():
                raise AuditIntegrityError("A human approver is required for this event")
            if approver_id.strip().casefold() in {"ai", "system", "anonymous", "assistant", "automated", "bot"}:
                raise AuditIntegrityError("Approver must be a human reviewer")
            if event_type in {"DECISION_MODIFIED", "DECISION_REJECTED"} and (not reason or not reason.strip()):
                raise AuditIntegrityError("A reason is required for modification and rejection")
            if event_type in {"DECISION_APPROVED", "DECISION_MODIFIED", "EXECUTION_AUTHORIZED"} and not final_action:
                raise AuditIntegrityError("A final action is required for this event")
            stamp = self._timestamp(timestamp)
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                decision_id=decision_id,
                account_id=decision.get("account_id"),
                user_id=decision.get("user_id"),
                timestamp=stamp,
                event_type=event_type,
                previous_status=previous_status,
                new_status=new_status,
                approver_id=approver_id,
                original_recommendation=original_action,
                final_action=final_action,
                reason=reason,
                metadata=copy.deepcopy(dict(metadata or {})),
            )
            self._events.append(event)
            self._status_by_decision[decision_id] = new_status
            self._last_timestamp = datetime.fromisoformat(stamp)
            return event.to_dict()

    def list_events(self) -> list[dict[str, object]]:
        with self._lock:
            return [event.to_dict() for event in self._events]

    def get_history(self, decision_id: str) -> list[dict[str, object]]:
        with self._lock:
            return [event.to_dict() for event in self._events if event.decision_id == decision_id]

    def has_event(self, decision_id: str, event_type: str) -> bool:
        with self._lock:
            return any(
                event.decision_id == decision_id and event.event_type == event_type
                for event in self._events
            )
