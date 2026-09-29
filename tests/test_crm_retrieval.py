import json

import numpy as np
import pandas as pd
import pytest

from src.nlp.crm_retrieval import (
    CRMEmbeddingIndex,
    DEFAULT_CRM_PATH,
    build_retrieval_query,
    load_crm_records,
    retrieve_account_evidence,
)
from src.nlp.transformer import (
    DEFAULT_MODEL_PATH,
    encode_query,
    load_embedding_model,
)


@pytest.fixture(scope="module")
def embedding_model():
    return load_embedding_model()


@pytest.fixture(scope="module")
def scoped_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "account-a",
                "interaction_id": "int-a-1",
                "date": "2025-09-10",
                "interaction_type": "support_ticket",
                "text": "Customer reports repeated login failures and cannot access the platform.",
            },
            {
                "user_id": "account-a",
                "interaction_id": "int-a-2",
                "date": "2025-09-12",
                "interaction_type": "email",
                "text": "Customer is happy with the new dashboard and uses analytics every day.",
            },
            {
                "user_id": "account-a",
                "interaction_id": "int-a-3",
                "date": "2025-09-14",
                "interaction_type": "billing",
                "text": "Customer reports a duplicate charge on the latest invoice.",
            },
            {
                "user_id": "account-b",
                "interaction_id": "int-b-1",
                "date": "2025-09-11",
                "interaction_type": "billing",
                "text": "Customer reports a billing issue with the latest invoice.",
            },
        ]
    )


@pytest.fixture(scope="module")
def scoped_index(embedding_model, scoped_records) -> CRMEmbeddingIndex:
    return CRMEmbeddingIndex(scoped_records, model=embedding_model)


def test_existing_local_minilm_path_exists() -> None:
    assert DEFAULT_MODEL_PATH.is_dir()


def test_minilm_loads_from_local_path_and_is_cached(embedding_model) -> None:
    assert embedding_model is load_embedding_model()


def test_actual_crm_source_loads_using_its_schema() -> None:
    records = load_crm_records()
    assert DEFAULT_CRM_PATH.is_file()
    assert len(records) == 500
    assert {"interaction_id", "interaction_type", "date", "text", "source_row"} <= set(
        records.columns
    )
    assert records["interaction_id"].iloc[0].startswith("CCC-")
    assert isinstance(records["text"].iloc[0], str)
    assert records["date"].isna().all()
    assert not {"user_id", "account_id"}.intersection(records.columns)
    assert not {
        "churn_risk_level",
        "churn_signals",
        "resolution_outcome",
        "summary",
    }.intersection(records.columns)


def test_actual_crm_text_is_preserved_from_source_turns() -> None:
    records = load_crm_records()
    with DEFAULT_CRM_PATH.open("r", encoding="utf-8") as source:
        original = json.loads(next(source))
    expected = "\n".join(turn["text"] for turn in original["conversation"])
    assert records.loc[0, "text"] == expected
    assert records.loc[0, "interaction_id"] == original["conversation_id"]
    assert records.loc[0, "interaction_type"] == original["channel"]


def test_actual_unlinked_conversations_are_not_assigned_to_stage3_accounts(
    embedding_model,
) -> None:
    records = load_crm_records()
    index = CRMEmbeddingIndex(records, model=embedding_model)
    result = index.retrieve(
        user_id="U0000001", query="support issue", as_of="2025-12-31"
    )
    assert index.unscoped_record_count == 500
    assert len(index.records) == 0
    assert index.embeddings.shape == (0, 384)
    assert result["crm_evidence"] == []
    assert result["evidence_status"] == "no_matching_crm_records"
    assert result["as_of"] == "2025-12-31T00:00:00+00:00"


def test_embeddings_are_finite_and_have_consistent_dimensions(scoped_index) -> None:
    assert scoped_index.embeddings.shape == (4, 384)
    assert np.isfinite(scoped_index.embeddings).all()
    assert np.allclose(np.linalg.norm(scoped_index.embeddings, axis=1), 1.0, atol=1e-4)
    assert all(len(vector) == scoped_index.embedding_dimension for vector in scoped_index.records["embedding"])


def test_account_scoping_excludes_other_customers(scoped_index) -> None:
    result = scoped_index.retrieve(
        user_id="account-a",
        query="Customer reports a billing issue with an invoice.",
        top_k=5,
    )
    assert result["crm_evidence"]
    assert all(item["user_id"] == "account-a" for item in result["crm_evidence"])
    assert not any(item["interaction_id"] == "int-b-1" for item in result["crm_evidence"])


def test_other_customer_records_never_leak_into_account_results(scoped_index) -> None:
    account_a = scoped_index.retrieve(user_id="account-a", query="billing issue")
    account_b = scoped_index.retrieve(user_id="account-b", query="billing issue")
    assert all(item["user_id"] == "account-a" for item in account_a["crm_evidence"])
    assert all(item["user_id"] == "account-b" for item in account_b["crm_evidence"])
    assert {item["interaction_id"] for item in account_a["crm_evidence"]}.isdisjoint(
        {item["interaction_id"] for item in account_b["crm_evidence"]}
    )


def test_similarity_is_cosine_dot_product_for_normalized_embeddings(scoped_index) -> None:
    query = "Customer reports repeated login failures and cannot access the platform."
    result = scoped_index.retrieve(user_id="account-a", query=query, top_k=1)
    query_embedding = encode_query(query, model=scoped_index.model)
    expected = float(scoped_index.embeddings[0] @ query_embedding)
    assert result["crm_evidence"][0]["interaction_id"] == "int-a-1"
    assert result["crm_evidence"][0]["similarity_score"] == pytest.approx(expected)
    assert result["crm_evidence"][0]["similarity_score"] == pytest.approx(1.0, abs=1e-5)


def test_results_are_sorted_by_similarity_descending(scoped_index) -> None:
    result = scoped_index.retrieve(user_id="account-a", query="billing invoice issue", top_k=5)
    scores = [item["similarity_score"] for item in result["crm_evidence"]]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_result_count(scoped_index) -> None:
    result = scoped_index.retrieve(user_id="account-a", query="customer support", top_k=1)
    assert len(result["crm_evidence"]) == 1


def test_fewer_than_top_k_records_returns_only_available_records(scoped_index) -> None:
    result = scoped_index.retrieve(user_id="account-b", query="billing issue", top_k=5)
    assert len(result["crm_evidence"]) == 1
    assert result["crm_evidence"][0]["interaction_id"] == "int-b-1"


def test_minimum_similarity_threshold_filters_results(scoped_index) -> None:
    query = "Customer reports repeated login failures and cannot access the platform."
    result = scoped_index.retrieve(
        user_id="account-a",
        query=query,
        top_k=5,
        min_similarity=0.9999,
    )
    assert [item["interaction_id"] for item in result["crm_evidence"]] == ["int-a-1"]
    assert result["crm_evidence"][0]["similarity_score"] >= 0.9999


def test_similarity_threshold_returns_empty_evidence_when_nothing_matches(
    scoped_index,
) -> None:
    result = scoped_index.retrieve(
        user_id="account-a",
        query="billing invoice",
        min_similarity=1.0,
    )
    assert result["crm_evidence"] == []
    assert result["evidence_status"] == "below_similarity_threshold"


def test_as_of_cutoff_excludes_later_interactions(scoped_index) -> None:
    result = scoped_index.retrieve(
        user_id="account-a",
        query="billing invoice",
        top_k=5,
        as_of="2025-09-10",
    )
    assert [item["interaction_id"] for item in result["crm_evidence"]] == ["int-a-1"]
    assert result["as_of"] == "2025-09-10T00:00:00+00:00"


def test_as_of_cutoff_returns_no_records_when_all_are_future(scoped_index) -> None:
    result = scoped_index.retrieve(
        user_id="account-a",
        query="billing invoice",
        as_of="2025-09-01",
    )
    assert result["crm_evidence"] == []
    assert result["evidence_status"] == "no_records_as_of"


def test_original_crm_text_is_preserved_and_only_embedding_copy_is_normalized(
    embedding_model,
) -> None:
    original_text = "  Customer   reported a billing issue.\nPlease follow up.  "
    records = pd.DataFrame(
        [{"account_id": "acct-1", "text": original_text, "interaction_id": "crm-1"}]
    )
    before = records.copy(deep=True)
    index = CRMEmbeddingIndex(records, model=embedding_model)
    result = index.retrieve(account_id="acct-1", query="billing issue", top_k=1)
    assert result["crm_evidence"][0]["text"] == original_text
    assert index.records.loc[0, "embedding_text"] == (
        "Customer reported a billing issue. Please follow up."
    )
    pd.testing.assert_frame_equal(records, before)


def test_evidence_output_preserves_interaction_metadata_and_traceability(
    scoped_index,
) -> None:
    result = scoped_index.retrieve(
        user_id="account-a",
        query="Customer reports repeated login failures and cannot access the platform.",
        top_k=1,
    )
    item = result["crm_evidence"][0]
    assert {
        "user_id",
        "date",
        "interaction_type",
        "text",
        "similarity_score",
        "rank",
        "interaction_id",
        "source_row",
        "source",
    } <= set(item)
    assert item["date"] == "2025-09-10"
    assert item["interaction_type"] == "support_ticket"
    assert item["interaction_id"] == "int-a-1"
    assert item["source"] == "crm_interaction"


def test_no_matching_account_returns_no_fabricated_records(scoped_index) -> None:
    result = scoped_index.retrieve(user_id="unknown-account", query="support issue")
    assert result["crm_evidence"] == []
    assert result["evidence_status"] == "no_matching_crm_records"


def test_empty_text_and_missing_text_are_skipped(embedding_model) -> None:
    records = pd.DataFrame(
        [
            {"user_id": "account-a", "text": "   "},
            {"user_id": "account-a", "text": None},
            {"user_id": "account-a", "text": "An actual interaction."},
        ]
    )
    index = CRMEmbeddingIndex(records, model=embedding_model)
    assert index.invalid_text_count == 2
    assert len(index.records) == 1


def test_multiple_accounts_are_retrieved_independently(scoped_index) -> None:
    results = scoped_index.retrieve_many(
        {
            "account-a": "customer reports login trouble",
            "account-b": "billing issue invoice",
        },
        top_k=3,
    )
    assert [result["user_id"] for result in results] == ["account-a", "account-b"]
    assert all(
        item["user_id"] == result["user_id"]
        for result in results
        for item in result["crm_evidence"]
    )


def test_records_with_both_ids_can_be_scoped_by_either_exact_identifier(
    embedding_model,
) -> None:
    records = pd.DataFrame(
        [
            {
                "user_id": "user-1",
                "account_id": "account-1",
                "text": "Account one has a billing issue.",
                "interaction_id": "interaction-1",
            },
            {
                "user_id": "user-2",
                "account_id": "account-2",
                "text": "Account two has a billing issue.",
                "interaction_id": "interaction-2",
            },
        ]
    )
    index = CRMEmbeddingIndex(records, model=embedding_model)
    by_user = index.retrieve(user_id="user-1", query="billing issue")
    by_account = index.retrieve(account_id="account-2", query="billing issue")
    assert [item["interaction_id"] for item in by_user["crm_evidence"]] == [
        "interaction-1"
    ]
    assert [item["interaction_id"] for item in by_account["crm_evidence"]] == [
        "interaction-2"
    ]


def test_query_generation_uses_stage5_drivers_deterministically() -> None:
    explanation = {
        "user_id": "account-a",
        "churn_probability": 0.8,
        "top_risk_drivers": [
            {
                "feature": "support_tickets",
                "feature_value": 3,
                "shap_value": 0.4,
                "direction": "increases_risk",
            },
            {
                "feature": "payment_failures",
                "feature_value": 1,
                "shap_value": 0.2,
                "direction": "increases_risk",
            },
        ],
    }
    assert build_retrieval_query(explanation) == build_retrieval_query(explanation)
    query = build_retrieval_query(explanation)
    assert "support tickets customer support" in query
    assert "payment failures billing payments" in query
    assert "value 3" in query
    assert "value 1" in query


def test_query_ignores_target_and_future_fields() -> None:
    explanation = {
        "top_risk_drivers": [
            {
                "feature": "nps_score",
                "feature_value": 2,
                "shap_value": 0.3,
                "direction": "increases_risk",
            }
        ],
        "churned": 1,
        "churned_next_month": 1,
        "churn_date": "2025-12-01",
        "future_mrr": 99999,
        "future_sessions": 1000,
    }
    changed = {
        **explanation,
        "churned": 0,
        "churned_next_month": 0,
        "churn_date": None,
        "future_mrr": 1,
        "future_sessions": 0,
    }
    query = build_retrieval_query(explanation)
    assert query == build_retrieval_query(changed)
    assert all(
        forbidden not in query
        for forbidden in (
            "churned",
            "churn_date",
            "future_mrr",
            "future_sessions",
            "2025-12-01",
            "99999",
        )
    )


def test_retrieval_does_not_mutate_input_records(scoped_index, scoped_records) -> None:
    before = scoped_records.copy(deep=True)
    scoped_index.retrieve(user_id="account-a", query="support issue")
    pd.testing.assert_frame_equal(scoped_records, before)


def test_repeated_retrieval_is_deterministic(scoped_index) -> None:
    first = scoped_index.retrieve(user_id="account-a", query="billing invoice", top_k=3)
    second = scoped_index.retrieve(user_id="account-a", query="billing invoice", top_k=3)
    assert first == second


def test_stage5_structured_and_stage6_crm_evidence_stay_separate(
    embedding_model,
) -> None:
    explanation = {
        "evidence": [{"feature": "support_tickets", "source": "model_feature"}],
        "top_risk_drivers": [
            {
                "feature": "support_tickets",
                "feature_value": 3,
                "shap_value": 0.4,
                "direction": "increases_risk",
            }
        ],
    }
    unlinked_index = CRMEmbeddingIndex(load_crm_records(), model=embedding_model)
    combined = retrieve_account_evidence(
        unlinked_index,
        user_id="U0000001",
        structured_explanation=explanation,
    )
    assert combined["structured_evidence"] == explanation["evidence"]
    assert combined["crm_evidence"] == []
    assert combined["evidence_status"] == "no_matching_crm_records"
    assert "support tickets customer support" in combined["retrieval_query"]


def test_empty_query_and_invalid_configuration_are_handled(scoped_index) -> None:
    empty = scoped_index.retrieve(user_id="account-a", query="   ")
    assert empty["crm_evidence"] == []
    assert empty["evidence_status"] == "empty_query"
    with pytest.raises(ValueError, match="top_k"):
        scoped_index.retrieve(user_id="account-a", query="support", top_k=0)
    with pytest.raises(ValueError, match="within"):
        scoped_index.retrieve(user_id="account-a", query="support", min_similarity=1.1)


def test_unscoped_index_requires_explicit_scope_for_multiple_queries(
    embedding_model,
) -> None:
    index = CRMEmbeddingIndex(load_crm_records(), model=embedding_model)
    with pytest.raises(ValueError, match="scope_column"):
        index.retrieve_many({"U0000001": "support issue"})
