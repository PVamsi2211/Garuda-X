from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Real
from typing import Any, Mapping

from src.data.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.decision.errors import ValidationError
from src.decision.playbooks import (
    classify_revenue_band,
    classify_risk_band,
    select_playbook_for_scores,
)
from src.prediction.revenue_risk import calculate_revenue_at_risk


@dataclass(frozen=True)
class PrimaryDriver:
    feature: str
    feature_value: object
    shap_value: float
    direction: str


@dataclass(frozen=True)
class CrmEvidence:
    source: str
    source_id: object
    interaction_date: str
    text: str
    similarity_score: float
    account_id: object | None = None
    user_id: object | None = None
    interaction_type: str | None = None


@dataclass(frozen=True)
class Recommendation:
    decision_id: str
    account_id: object | None
    user_id: object | None
    generated_at: str
    risk_score: float
    risk_band: str
    mrr: float
    revenue_at_risk: float
    revenue_band: str
    primary_drivers: tuple[PrimaryDriver, ...]
    crm_evidence: tuple[CrmEvidence, ...]
    evidence_status: str
    playbook_id: str
    recommended_action: str
    rationale: str
    priority: str
    owner: str
    safeguards: tuple[str, ...]
    status: str = "PENDING"
    requires_human_approval: bool = True
    playbook_version: str = "1.0"
    original_recommended_action: str = ""
    final_action: str | None = None
    approver_id: str | None = None
    modification_reason: str | None = None
    rejection_reason: str | None = None
    status_updated_at: str | None = None
    approval_event_id: str | None = None
    execution_authorized: bool = False
    execution_authorized_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["primary_drivers"] = [asdict(item) for item in self.primary_drivers]
        result["crm_evidence"] = [asdict(item) for item in self.crm_evidence]
        result["safeguards"] = list(self.safeguards)
        return result


def _timestamp(value: object | None, *, name: str, allow_date: bool = False) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    elif allow_date and isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValidationError(f"{name} must be an ISO-8601 timestamp") from error
    else:
        isoformat = getattr(value, "isoformat", None)
        if not callable(isoformat):
            raise ValidationError(f"{name} must be an ISO-8601 timestamp")
        try:
            return _timestamp(isoformat(), name=name, allow_date=allow_date)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _identifier(value: object | None, name: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValidationError(f"{name} must be a non-empty string or integer")
    if not str(value).strip():
        raise ValidationError(f"{name} must not be empty")
    return value


def _finite_number(value: object, name: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValidationError(f"{name} must be at most {maximum}")
    return result


def _driver_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _driver_value(item())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _primary_drivers(shap_result: Mapping[str, Any] | None) -> tuple[PrimaryDriver, ...]:
    if shap_result is None:
        return ()
    if not isinstance(shap_result, Mapping):
        raise ValidationError("shap_result must be a mapping")
    approved = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    raw_drivers = shap_result.get("top_risk_drivers", [])
    if not isinstance(raw_drivers, (list, tuple)):
        raise ValidationError("top_risk_drivers must be a list")
    selected: dict[str, PrimaryDriver] = {}
    for item in raw_drivers:
        if not isinstance(item, Mapping):
            raise ValidationError("each SHAP driver must be a mapping")
        feature = item.get("feature")
        if not isinstance(feature, str) or feature not in approved:
            continue
        raw_value = item.get("shap_value")
        shap_value = _finite_number(raw_value, "shap_value")
        if shap_value <= 0 or feature in selected:
            continue
        selected[str(feature)] = PrimaryDriver(
            feature=str(feature),
            feature_value=_driver_value(item.get("feature_value")),
            shap_value=shap_value,
            direction="increases_risk",
        )
    return tuple(sorted(selected.values(), key=lambda driver: (-driver.shap_value, driver.feature)))


def _crm_evidence(
    result: Mapping[str, Any] | None,
    *,
    account_id: object | None,
    user_id: object | None,
    cutoff: datetime,
) -> tuple[CrmEvidence, ...]:
    if result is None:
        return ()
    if not isinstance(result, Mapping):
        raise ValidationError("crm_result must be a mapping")
    for key, expected in (("account_id", account_id), ("user_id", user_id)):
        observed = result.get(key)
        if observed is not None and expected is not None and observed != expected:
            raise ValidationError(f"CRM result {key} does not match the recommendation")
    rows = result.get("crm_evidence", [])
    if not isinstance(rows, (list, tuple)):
        raise ValidationError("crm_evidence must be a list")
    retained: list[CrmEvidence] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("source") != "crm_interaction":
            continue
        row_account = row.get("account_id")
        row_user = row.get("user_id")
        if row_account is not None and (account_id is None or row_account != account_id):
            continue
        if row_user is not None and (user_id is None or row_user != user_id):
            continue
        if row_account is None and row_user is None:
            continue
        source_id = None
        for candidate in (row.get("interaction_id"), row.get("conversation_id"), row.get("source_row")):
            candidate = _driver_value(candidate) if candidate is not None else None
            valid_candidate = (
                isinstance(candidate, str) and bool(candidate.strip())
            ) or (
                isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0
            )
            if valid_candidate:
                source_id = candidate
                break
        text = row.get("text")
        raw_date = row.get("interaction_date", row.get("date"))
        if source_id is None or not isinstance(text, str) or not text.strip() or raw_date is None:
            continue
        try:
            interaction_date = _timestamp(raw_date, name="CRM interaction date", allow_date=True)
        except ValidationError:
            continue
        if interaction_date > cutoff:
            continue
        similarity = _finite_number(row.get("similarity_score"), "similarity_score")
        if not -1.0 <= similarity <= 1.0:
            raise ValidationError("similarity_score must be within [-1, 1]")
        raw_type = row.get("interaction_type")
        retained.append(CrmEvidence(
            source="crm_interaction",
            source_id=source_id,
            interaction_date=interaction_date.isoformat(),
            text=text,
            similarity_score=similarity,
            account_id=row_account,
            user_id=row_user,
            interaction_type=raw_type if isinstance(raw_type, str) else None,
        ))
    return tuple(retained)


def build_recommendation(
    *,
    risk_score: Real,
    mrr: Real,
    account_id: object | None = None,
    user_id: object | None = None,
    shap_result: Mapping[str, Any] | None = None,
    crm_result: Mapping[str, Any] | None = None,
    as_of: object | None = None,
    decision_id: str | None = None,
    generated_at: object | None = None,
) -> Recommendation:
    account_id = _identifier(account_id, "account_id")
    user_id = _identifier(user_id, "user_id")
    if account_id is None and user_id is None:
        raise ValidationError("account_id or user_id is required")
    if decision_id is None:
        decision_id = str(uuid.uuid4())
    if not isinstance(decision_id, str) or not decision_id.strip():
        raise ValidationError("decision_id must be a non-empty string")

    score = _finite_number(risk_score, "risk_score", minimum=0.0, maximum=1.0)
    current_mrr = _finite_number(mrr, "mrr", minimum=0.0)
    generated = _timestamp(generated_at, name="generated_at")
    cutoff = min(generated, _timestamp(as_of, name="as_of")) if as_of is not None else generated
    revenue_at_risk = float(calculate_revenue_at_risk(current_mrr, score))
    if not math.isfinite(revenue_at_risk) or revenue_at_risk < 0:
        raise ValidationError("revenue_at_risk must be finite and non-negative")

    playbook = select_playbook_for_scores(score, current_mrr)
    drivers = _primary_drivers(shap_result)
    evidence = _crm_evidence(
        crm_result, account_id=account_id, user_id=user_id, cutoff=cutoff
    )
    risk_band = classify_risk_band(score)
    revenue_band = classify_revenue_band(current_mrr)
    rationale = f"Model-predicted churn risk is {risk_band.lower()}."
    rationale += f" Revenue at risk is {revenue_at_risk:.2f}."
    if drivers:
        names = ", ".join(driver.feature.replace("_", " ") for driver in drivers)
        rationale += f" Primary contributing signals include {names}."
    else:
        rationale += " No eligible SHAP drivers were supplied."
    rationale += " CRM evidence was unavailable." if not evidence else f" {len(evidence)} matching CRM evidence record(s) were retrieved."

    return Recommendation(
        decision_id=decision_id,
        account_id=account_id,
        user_id=user_id,
        generated_at=generated.isoformat(),
        risk_score=score,
        risk_band=risk_band,
        mrr=current_mrr,
        revenue_at_risk=revenue_at_risk,
        revenue_band=revenue_band,
        primary_drivers=drivers,
        crm_evidence=evidence,
        evidence_status="retrieved" if evidence else "no_matching_crm_records",
        playbook_id=playbook.playbook_id,
        recommended_action=playbook.recommended_action,
        rationale=rationale,
        priority=playbook.priority,
        owner=playbook.owner,
        safeguards=playbook.safeguards,
        original_recommended_action=playbook.recommended_action,
    )
