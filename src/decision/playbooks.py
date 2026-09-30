from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from numbers import Real
from types import MappingProxyType

from config.thresholds import HIGH_REVENUE_THRESHOLD
from src.prediction.risk_scoring import classify_risk
from src.decision.errors import ValidationError


@dataclass(frozen=True)
class Playbook:
    playbook_id: str
    name: str
    description: str
    priority: str
    recommended_action: str
    owner: str
    required_evidence: tuple[str, ...]
    safeguards: tuple[str, ...]
    version: str
    execution_policy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SAFEGUARDS = (
    "Requires human review before any customer-facing action.",
    "Do not apply pricing or contractual changes automatically.",
    "Verify CRM evidence before any customer outreach.",
)

_PLAYBOOKS = {
    "HIGH_RISK_HIGH_REVENUE": Playbook(
        "HIGH_RISK_HIGH_REVENUE", "Priority retention review",
        "Escalate a high-risk, high-revenue account for internal review.",
        "CRITICAL", "Escalate account for priority retention review",
        "Customer Success", ("current_month_mrr", "predicted_churn_risk"),
        _SAFEGUARDS, "1.0", "human_approval_required",
    ),
    "HIGH_RISK_LOW_REVENUE": Playbook(
        "HIGH_RISK_LOW_REVENUE", "Targeted health review",
        "Review a high-risk account through the customer-success process.",
        "HIGH", "Initiate targeted customer-success follow-up",
        "Customer Success", ("current_month_mrr", "predicted_churn_risk"),
        _SAFEGUARDS, "1.0", "human_approval_required",
    ),
    "MEDIUM_RISK_HIGH_REVENUE": Playbook(
        "MEDIUM_RISK_HIGH_REVENUE", "Proactive account review",
        "Schedule an internal proactive review for a medium-risk, high-revenue account.",
        "HIGH", "Schedule proactive account review",
        "Account Executive", ("current_month_mrr", "predicted_churn_risk"),
        _SAFEGUARDS, "1.0", "human_approval_required",
    ),
    "MEDIUM_RISK_LOW_REVENUE": Playbook(
        "MEDIUM_RISK_LOW_REVENUE", "Account health monitoring",
        "Review account health and monitor engagement for a medium-risk account.",
        "MEDIUM", "Review account health and monitor engagement",
        "Customer Success", ("current_month_mrr", "predicted_churn_risk"),
        _SAFEGUARDS, "1.0", "human_approval_required",
    ),
    "LOW_RISK_MONITOR": Playbook(
        "LOW_RISK_MONITOR", "Routine monitoring",
        "Continue routine monitoring of current account health.",
        "LOW", "Continue monitoring account health",
        "Customer Success", ("current_month_mrr", "predicted_churn_risk"),
        _SAFEGUARDS, "1.0", "human_approval_required",
    ),
}

PLAYBOOKS = MappingProxyType(_PLAYBOOKS)


def classify_risk_band(risk_score: Real) -> str:
    if isinstance(risk_score, bool) or not isinstance(risk_score, (Real, Decimal)):
        raise ValidationError("risk_score must be numeric")
    value = float(risk_score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValidationError("risk_score must be finite and within [0, 1]")
    return classify_risk(value).upper()


def classify_revenue_band(mrr: Real) -> str:
    if isinstance(mrr, bool) or not isinstance(mrr, (Real, Decimal)):
        raise ValidationError("mrr must be numeric")
    value = float(mrr)
    threshold = float(HIGH_REVENUE_THRESHOLD)
    if not math.isfinite(value) or value < 0:
        raise ValidationError("mrr must be finite and non-negative")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValidationError("HIGH_REVENUE_THRESHOLD must be finite and non-negative")
    return "HIGH_REVENUE" if value >= threshold else "LOW_REVENUE"


def select_playbook(risk_band: str, revenue_band: str) -> Playbook:
    normalized_risk = str(risk_band).upper()
    normalized_revenue = str(revenue_band).upper()
    if normalized_revenue not in {"HIGH_REVENUE", "LOW_REVENUE"}:
        raise ValidationError(f"Unknown revenue band: {revenue_band}")
    if normalized_risk == "LOW":
        playbook_id = "LOW_RISK_MONITOR"
    elif normalized_risk in {"MEDIUM", "HIGH"}:
        playbook_id = f"{normalized_risk}_RISK_{normalized_revenue}"
    else:
        raise ValidationError(f"Unknown risk band: {risk_band}")
    return PLAYBOOKS[playbook_id]


def select_playbook_for_scores(risk_score: Real, mrr: Real) -> Playbook:
    return select_playbook(classify_risk_band(risk_score), classify_revenue_band(mrr))


def get_playbook(playbook_id: str) -> Playbook:
    if not isinstance(playbook_id, str) or playbook_id not in PLAYBOOKS:
        raise ValidationError(f"Unknown playbook: {playbook_id}")
    return PLAYBOOKS[playbook_id]
