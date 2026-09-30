from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
import pandas as pd

from config.thresholds import HIGH_REVENUE_THRESHOLD
from src.decision.approval import DecisionService
from src.decision.errors import (
    AlreadyAuthorizedError,
    AlreadyFinalizedError,
    ExecutionNotAuthorizedError,
    InvalidDecisionError,
    ValidationError,
)
from src.decision.playbooks import (
    PLAYBOOKS,
    classify_revenue_band,
    classify_risk_band,
    select_playbook_for_scores,
)
from src.decision.recommendation import build_recommendation


class FrozenClock:
    def __init__(self):
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        value = self.value
        self.value += timedelta(seconds=1)
        return value


def make_service():
    return DecisionService(clock=FrozenClock())


def create(service=None, **overrides):
    service = service or make_service()
    values = {
        "account_id": "acct-1",
        "user_id": "user-1",
        "risk_score": 0.8,
        "mrr": HIGH_REVENUE_THRESHOLD,
    }
    values.update(overrides)
    return service, service.create_recommendation(**values)


def test_five_baseline_playbooks_are_exact_and_complete():
    expected = {
        "HIGH_RISK_HIGH_REVENUE",
        "HIGH_RISK_LOW_REVENUE",
        "MEDIUM_RISK_HIGH_REVENUE",
        "MEDIUM_RISK_LOW_REVENUE",
        "LOW_RISK_MONITOR",
    }
    assert set(PLAYBOOKS) == expected
    for playbook_id, playbook in PLAYBOOKS.items():
        assert playbook.playbook_id == playbook_id
        assert all((playbook.name, playbook.description, playbook.priority,
                    playbook.recommended_action, playbook.owner, playbook.version,
                    playbook.execution_policy))
        assert playbook.required_evidence
        assert playbook.safeguards


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "LOW"), (0.299999, "LOW"), (0.30, "MEDIUM"),
     (0.699999, "MEDIUM"), (0.70, "HIGH"), (1.0, "HIGH")],
)
def test_risk_band_boundaries(score, expected):
    assert classify_risk_band(score) == expected


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_risk_values_fail(score):
    with pytest.raises(ValidationError):
        classify_risk_band(score)


@pytest.mark.parametrize("mrr", [0.0, HIGH_REVENUE_THRESHOLD - 0.01])
def test_low_revenue_boundary(mrr):
    assert classify_revenue_band(mrr) == "LOW_REVENUE"


def test_high_revenue_boundary_is_inclusive():
    assert classify_revenue_band(HIGH_REVENUE_THRESHOLD) == "HIGH_REVENUE"


@pytest.mark.parametrize("mrr", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_mrr_values_fail(mrr):
    with pytest.raises(ValidationError):
        classify_revenue_band(mrr)


@pytest.mark.parametrize(
    ("score", "mrr", "playbook_id"),
    [
        (0.8, HIGH_REVENUE_THRESHOLD, "HIGH_RISK_HIGH_REVENUE"),
        (0.8, 0.0, "HIGH_RISK_LOW_REVENUE"),
        (0.4, HIGH_REVENUE_THRESHOLD, "MEDIUM_RISK_HIGH_REVENUE"),
        (0.4, 0.0, "MEDIUM_RISK_LOW_REVENUE"),
        (0.0, HIGH_REVENUE_THRESHOLD, "LOW_RISK_MONITOR"),
    ],
)
def test_playbook_combinations(score, mrr, playbook_id):
    assert select_playbook_for_scores(score, mrr).playbook_id == playbook_id


def test_recommendation_uses_canonical_revenue_risk_and_initial_gate():
    recommendation = build_recommendation(
        account_id="acct-1", risk_score=0.4, mrr=125.0, decision_id="decision-1"
    )
    assert recommendation.revenue_at_risk == 50.0
    assert recommendation.status == "PENDING"
    assert recommendation.requires_human_approval is True
    assert recommendation.risk_band == "MEDIUM"


def test_zero_mrr_yields_zero_revenue_risk():
    recommendation = build_recommendation(user_id="user-1", risk_score=1.0, mrr=0)
    assert recommendation.revenue_at_risk == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"account_id": "acct", "risk_score": -0.1, "mrr": 1},
        {"account_id": "acct", "risk_score": 1.1, "mrr": 1},
        {"account_id": "acct", "risk_score": float("nan"), "mrr": 1},
        {"account_id": "acct", "risk_score": float("inf"), "mrr": 1},
        {"account_id": "acct", "risk_score": 0.2, "mrr": -1},
        {"account_id": "acct", "risk_score": 0.2, "mrr": float("nan")},
        {"account_id": "acct", "risk_score": 0.2, "mrr": float("inf")},
    ],
)
def test_recommendation_rejects_invalid_scores_and_mrr(kwargs):
    with pytest.raises(ValidationError):
        build_recommendation(**kwargs)


def test_shap_evidence_uses_only_supplied_positive_approved_drivers():
    shap = {"top_risk_drivers": [
        {"feature": "nps_score", "feature_value": 2, "shap_value": 0.5},
        {"feature": "sessions", "feature_value": 4, "shap_value": -0.4},
        {"feature": "churned_next_month", "feature_value": 1, "shap_value": 9.0},
    ]}
    original = deepcopy(shap)
    recommendation = build_recommendation(
        user_id="user-1", risk_score=0.8, mrr=100, shap_result=shap
    )
    assert [(item.feature, item.shap_value) for item in recommendation.primary_drivers] == [
        ("nps_score", 0.5)
    ]
    assert "nps score" in recommendation.rationale
    assert "cause" not in recommendation.rationale.casefold()
    assert shap == original


def test_missing_shap_is_empty_and_does_not_fabricate_drivers():
    recommendation = build_recommendation(user_id="user-1", risk_score=0.3, mrr=0)
    assert recommendation.primary_drivers == ()
    assert "No eligible SHAP drivers" in recommendation.rationale


def test_matching_crm_evidence_preserves_source_text_and_similarity():
    crm = {"user_id": "user-1", "crm_evidence": [{
        "source": "crm_interaction", "user_id": "user-1", "interaction_id": "crm-17",
        "date": "2025-12-31T00:00:00+00:00", "text": "Original interaction text.",
        "similarity_score": 0.81,
    }]}
    recommendation = build_recommendation(
        user_id="user-1", risk_score=0.6, mrr=0,
        crm_result=crm, generated_at="2026-01-01T00:00:00Z",
    )
    evidence = recommendation.crm_evidence[0]
    assert recommendation.evidence_status == "retrieved"
    assert evidence.source_id == "crm-17"
    assert evidence.text == "Original interaction text."
    assert evidence.similarity_score == 0.81


def test_missing_unscoped_mismatched_and_future_crm_is_not_included():
    assert build_recommendation(
        user_id="user-1", risk_score=0.2, mrr=0
    ).evidence_status == "no_matching_crm_records"
    crm = {"user_id": "user-1", "crm_evidence": [
        {"source": "crm_interaction", "user_id": "other", "interaction_id": "x",
         "date": "2025-12-31T00:00:00Z", "text": "Wrong account", "similarity_score": 0.9},
        {"source": "crm_interaction", "user_id": "user-1", "interaction_id": "future",
         "date": "2026-01-02T00:00:00Z", "text": "Future record", "similarity_score": 0.9},
        {"source": "crm_interaction", "interaction_id": "unscoped",
         "date": "2025-12-31T00:00:00Z", "text": "Unscoped record", "similarity_score": 0.9},
    ]}
    recommendation = build_recommendation(
        user_id="user-1", risk_score=0.2, mrr=0, crm_result=crm,
        generated_at="2026-01-01T00:00:00Z",
    )
    assert recommendation.crm_evidence == ()
    assert recommendation.evidence_status == "no_matching_crm_records"
    assert "CRM evidence was unavailable" in recommendation.rationale


def test_mismatched_crm_result_scope_fails_validation():
    with pytest.raises(ValidationError):
        build_recommendation(
            user_id="user-1", risk_score=0.2, mrr=0,
            crm_result={"user_id": "other", "crm_evidence": []},
        )


def test_missing_crm_date_or_source_id_is_not_retained():
    crm = {"user_id": "user-1", "crm_evidence": [{
        "source": "crm_interaction", "user_id": "user-1", "text": "No trace id or date",
        "similarity_score": 0.7,
    }]}
    result = build_recommendation(user_id="user-1", risk_score=0.2, mrr=0, crm_result=crm)
    assert result.crm_evidence == ()


def test_missing_identifier_is_rejected():
    with pytest.raises(ValidationError):
        build_recommendation(risk_score=0.2, mrr=10)


@pytest.mark.parametrize(
    "forbidden_field",
    ["churned", "churned_next_month", "future_mrr", "future_revenue", "post_churn_feature"],
)
def test_outcome_and_future_fields_are_rejected(forbidden_field):
    with pytest.raises(TypeError):
        build_recommendation(**{
            "user_id": "user-1", "risk_score": 0.2, "mrr": 10,
            forbidden_field: 0,
        })


def test_deterministic_recommendation_logic_except_id_and_time():
    first = build_recommendation(user_id="user-1", risk_score=0.8, mrr=700)
    second = build_recommendation(user_id="user-1", risk_score=0.8, mrr=700)
    assert first.playbook_id == second.playbook_id
    assert first.recommended_action == second.recommended_action
    assert first.priority == second.priority
    assert first.rationale == second.rationale
    assert first.decision_id != second.decision_id


def test_decision_ids_are_not_account_ids_and_missing_decisions_fail():
    service, pending = create()
    assert pending["decision_id"] != pending["account_id"]
    with pytest.raises(InvalidDecisionError):
        service.get_decision("missing")


def test_pending_to_approved_records_human_and_audit_event():
    service, pending = create()
    approved = service.approve_decision(pending["decision_id"], "reviewer-1")
    assert approved["status"] == "APPROVED"
    assert approved["approver_id"] == "reviewer-1"
    assert approved["final_action"] == pending["original_recommended_action"]
    history = service.get_decision_history(pending["decision_id"])
    assert [event["event_type"] for event in history] == [
        "RECOMMENDATION_CREATED", "DECISION_APPROVED"
    ]
    assert history[1]["previous_status"] == "PENDING"
    assert history[1]["new_status"] == "APPROVED"


def test_pending_to_modified_preserves_original_action_and_reason():
    service, pending = create()
    modified = service.modify_decision(
        pending["decision_id"], "reviewer-1", "Review the account internally", "Policy review"
    )
    assert modified["status"] == "MODIFIED"
    assert modified["original_recommended_action"] == pending["recommended_action"]
    assert modified["final_action"] == "Review the account internally"
    assert modified["modification_reason"] == "Policy review"
    event = service.get_decision_history(pending["decision_id"])[1]
    assert event["previous_status"] == "PENDING"
    assert event["new_status"] == "MODIFIED"
    assert event["reason"] == "Policy review"


def test_pending_to_rejected_preserves_original_and_reason():
    service, pending = create()
    rejected = service.reject_decision(pending["decision_id"], "reviewer-1", "Insufficient evidence")
    assert rejected["status"] == "REJECTED"
    assert rejected["original_recommended_action"] == pending["recommended_action"]
    assert rejected["final_action"] is None
    assert rejected["rejection_reason"] == "Insufficient evidence"
    assert service.get_decision_history(pending["decision_id"])[1]["new_status"] == "REJECTED"


@pytest.mark.parametrize("actor", [None, "", "  ", "AI", "system", "anonymous"])
def test_missing_or_nonhuman_approver_fails_without_transition(actor):
    service, pending = create()
    with pytest.raises(ValidationError):
        service.approve_decision(pending["decision_id"], actor)
    assert service.get_decision(pending["decision_id"])["status"] == "PENDING"
    assert len(service.get_decision_history(pending["decision_id"])) == 1


def test_missing_decision_id_fails_without_transition():
    service, pending = create()
    with pytest.raises(ValidationError):
        service.approve_decision("", "reviewer")
    assert service.get_decision(pending["decision_id"])["status"] == "PENDING"


@pytest.mark.parametrize(
    "operation",
    [
        lambda s, d: s.modify_decision(d, "reviewer", " ", "reason"),
        lambda s, d: s.modify_decision(d, "reviewer", "Action", " "),
        lambda s, d: s.reject_decision(d, "reviewer", " "),
    ],
)
def test_empty_actions_or_reasons_fail_without_transition(operation):
    service, pending = create()
    with pytest.raises(ValidationError):
        operation(service, pending["decision_id"])
    assert service.get_decision(pending["decision_id"])["status"] == "PENDING"
    assert len(service.get_decision_history(pending["decision_id"])) == 1


@pytest.mark.parametrize("decision", ["approved", "modified", "rejected"])
@pytest.mark.parametrize("next_operation", ["approve", "modify", "reject"])
def test_finalized_decisions_reject_all_further_transitions(decision, next_operation):
    service, pending = create()
    if decision == "approved":
        service.approve_decision(pending["decision_id"], "reviewer")
    elif decision == "modified":
        service.modify_decision(pending["decision_id"], "reviewer", "Action", "Reason")
    else:
        service.reject_decision(pending["decision_id"], "reviewer", "Reason")
    before = service.get_decision(pending["decision_id"])
    with pytest.raises(AlreadyFinalizedError):
        if next_operation == "approve":
            service.approve_decision(pending["decision_id"], "reviewer-2")
        elif next_operation == "modify":
            service.modify_decision(pending["decision_id"], "reviewer-2", "Other", "Reason")
        else:
            service.reject_decision(pending["decision_id"], "reviewer-2", "Reason")
    assert service.get_decision(pending["decision_id"]) == before


def test_pending_and_rejected_cannot_receive_execution_authorization():
    service, pending = create()
    with pytest.raises(ExecutionNotAuthorizedError):
        service.authorize_execution(pending["decision_id"])
    service.reject_decision(pending["decision_id"], "reviewer", "Reason")
    with pytest.raises(ExecutionNotAuthorizedError):
        service.authorize_execution(pending["decision_id"])
    assert not any(
        event["event_type"] == "EXECUTION_AUTHORIZED"
        for event in service.get_decision_history(pending["decision_id"])
    )


@pytest.mark.parametrize("modified", [False, True])
def test_approved_or_modified_decision_can_be_authorized_once(modified):
    service, pending = create()
    if modified:
        service.modify_decision(pending["decision_id"], "reviewer", "Review internally", "Reason")
    else:
        service.approve_decision(pending["decision_id"], "reviewer")
    authorized = service.authorize_execution(pending["decision_id"])
    assert authorized["execution_authorized"] is True
    assert service.is_execution_authorized(pending["decision_id"])
    with pytest.raises(AlreadyAuthorizedError):
        service.authorize_execution(pending["decision_id"])
    assert sum(
        event["event_type"] == "EXECUTION_AUTHORIZED"
        for event in service.get_decision_history(pending["decision_id"])
    ) == 1


def test_audit_lifecycle_keeps_same_decision_id_and_chronological_statuses():
    service, pending = create()
    service.modify_decision(pending["decision_id"], "reviewer", "Review internally", "Reason")
    service.authorize_execution(pending["decision_id"])
    history = service.get_decision_history(pending["decision_id"])
    assert {event["decision_id"] for event in history} == {pending["decision_id"]}
    assert [event["event_type"] for event in history] == [
        "RECOMMENDATION_CREATED", "DECISION_MODIFIED", "EXECUTION_AUTHORIZED"
    ]
    assert [(event["previous_status"], event["new_status"]) for event in history] == [
        (None, "PENDING"), ("PENDING", "MODIFIED"),
        ("MODIFIED", "EXECUTION_AUTHORIZED"),
    ]
    assert [event["timestamp"] for event in history] == sorted(
        event["timestamp"] for event in history
    )


def test_returned_decisions_and_audit_history_are_copies():
    service, pending = create()
    pending["status"] = "APPROVED"
    service_copy = service.get_decision(pending["decision_id"])
    service_copy["safeguards"].clear()
    history = service.get_decision_history(pending["decision_id"])
    history[0]["metadata"]["risk_band"] = "CORRUPTED"
    assert service.get_decision(pending["decision_id"])["status"] == "PENDING"
    assert service.get_decision(pending["decision_id"])["safeguards"]
    assert service.get_decision_history(pending["decision_id"])[0]["metadata"]["risk_band"] == "HIGH"


def test_crm_input_mapping_and_list_are_not_mutated():
    rows = [{
        "source": "crm_interaction", "user_id": "user-1", "interaction_id": "crm-1",
        "date": "2025-12-31T00:00:00Z", "text": "Source text", "similarity_score": 0.4,
    }]
    crm = {"user_id": "user-1", "crm_evidence": rows}
    original = deepcopy(crm)
    build_recommendation(
        user_id="user-1", risk_score=0.5, mrr=0, crm_result=crm,
        generated_at="2026-01-01T00:00:00Z",
    )
    assert crm == original
    assert rows == original["crm_evidence"]


def test_dataframe_input_is_rejected_without_mutation():
    frame = pd.DataFrame({"feature": ["nps_score"], "shap_value": [0.2]})
    original = frame.copy(deep=True)
    with pytest.raises(ValidationError):
        build_recommendation(user_id="user-1", risk_score=0.5, mrr=0, shap_result=frame)
    pd.testing.assert_frame_equal(frame, original)


def test_service_getters_support_later_ui_without_exposing_internal_state():
    service, pending = create()
    assert service.get_pending_decisions()[0]["decision_id"] == pending["decision_id"]
    assert service.get_accounts() == [{
        "account_id": "acct-1", "user_id": "user-1",
        "latest_decision_id": pending["decision_id"], "decision_count": 1,
    }]
    assert service.list_decisions()[0]["decision_id"] == pending["decision_id"]
    assert service.get_audit_history()[0]["decision_id"] == pending["decision_id"]


def test_duplicate_decision_id_does_not_add_event_or_replace_decision():
    service, pending = create()
    with pytest.raises(ValidationError):
        service.create_recommendation(
            decision_id=pending["decision_id"], account_id="acct-2", risk_score=0.2, mrr=2
        )
    assert service.get_decision(pending["decision_id"])["account_id"] == "acct-1"
    assert len(service.get_audit_history()) == 1
