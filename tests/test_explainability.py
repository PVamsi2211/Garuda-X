import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.explainability.evidence_engine import format_feature_evidence
from src.explainability.shap_engine import (
    APPROVED_FEATURES,
    SHAP_OUTPUT_SPACE,
    ShapEngine,
    build_feature_explanations,
    load_shap_engine,
    select_top_drivers,
)
from src.prediction.predictor import load_churn_predictor

FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@pytest.fixture(scope="module")
def engine() -> ShapEngine:
    return load_shap_engine()


def example_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user-alpha", "user-beta"],
            "mrr": [120.0, 45.5],
            "sessions": [5, 19],
            "feature_usage_score": [0.42, 0.81],
            "support_tickets": [3, 0],
            "payment_failures": [1, 0],
            "nps_score": [4, 9],
            "product_incident": [1, 0],
            "active_seats": [2, 7],
            "tenure_month": [6, 18],
            "plan_type": ["starter", "pro"],
        }
    )


def test_engine_loads_existing_saved_model(engine: ShapEngine) -> None:
    assert engine.predictor.model.get_booster() is not None
    assert engine.predictor.metadata["feature_list"] == APPROVED_FEATURES


def test_explanation_output_exists_and_has_expected_feature_count(
    engine: ShapEngine,
) -> None:
    result = engine.explain_records(example_records().iloc[[0]])[0]
    assert len(result["feature_explanations"]) == len(FEATURES) == 10


def test_feature_explanation_records_have_required_fields(engine: ShapEngine) -> None:
    result = engine.explain_records(example_records().iloc[[0]])[0]
    required = {
        "feature",
        "feature_value",
        "shap_value",
        "direction",
        "absolute_importance",
    }
    assert all(required <= set(item) for item in result["feature_explanations"])


def test_positive_negative_zero_directions_and_absolute_importance() -> None:
    explanations = build_feature_explanations(
        {"sessions": 5, "feature_usage_score": 0.82, "nps_score": 8},
        {"sessions": 0.31, "feature_usage_score": -0.18, "nps_score": 0.0},
    )
    by_feature = {item["feature"]: item for item in explanations}
    assert by_feature["sessions"]["direction"] == "increases_risk"
    assert by_feature["feature_usage_score"]["direction"] == "decreases_risk"
    assert by_feature["nps_score"]["direction"] == "neutral"
    assert by_feature["sessions"]["absolute_importance"] == pytest.approx(0.31)
    assert by_feature["feature_usage_score"]["absolute_importance"] == pytest.approx(0.18)
    assert by_feature["nps_score"]["absolute_importance"] == 0.0


def test_explanations_sort_by_absolute_shap_with_stable_ties() -> None:
    explanations = build_feature_explanations(
        {"zeta": 1, "beta": 2, "alpha": 3},
        {"zeta": -0.4, "beta": 0.4, "alpha": 0.1},
    )
    assert [item["feature"] for item in explanations] == ["beta", "zeta", "alpha"]


def test_positive_drivers_and_protective_factors_sort_by_absolute_value() -> None:
    explanations = build_feature_explanations(
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 0.12, "b": 0.51, "c": -0.7, "d": -0.2},
    )
    assert [
        item["feature"]
        for item in select_top_drivers(explanations, direction="increases_risk")
    ] == ["b", "a"]
    assert [
        item["feature"]
        for item in select_top_drivers(explanations, direction="decreases_risk")
    ] == ["c", "d"]


def test_top_n_limits_each_driver_list(engine: ShapEngine) -> None:
    result = engine.explain_records(example_records().iloc[[0]], top_n=1)[0]
    assert len(result["top_risk_drivers"]) <= 1
    assert len(result["protective_factors"]) <= 1
    assert len(result["feature_explanations"]) == 10


def test_user_id_is_preserved_exactly(engine: ShapEngine) -> None:
    result = engine.explain_records(example_records().iloc[[0]])[0]
    assert result["user_id"] == "user-alpha"


def test_explanation_does_not_mutate_input(engine: ShapEngine) -> None:
    records = example_records()
    before = records.copy(deep=True)
    engine.explain_records(records)
    pd.testing.assert_frame_equal(records, before)


def test_multiple_accounts_keep_independent_results(engine: ShapEngine) -> None:
    records = example_records()
    results = engine.explain_records(records)
    assert [item["user_id"] for item in results] == ["user-alpha", "user-beta"]
    assert results[0]["feature_explanations"] != results[1]["feature_explanations"]
    assert results[0]["churn_probability"] != results[1]["churn_probability"]


def test_saved_predictor_probability_is_reused(engine: ShapEngine) -> None:
    records = example_records()
    expected = engine.predictor.predict_proba(records[FEATURES]).tolist()
    actual = [item["churn_probability"] for item in engine.explain_records(records)]
    assert actual == pytest.approx(expected)
    assert all(item["probability_calibration"] == "uncalibrated" for item in engine.explain_records(records))


def test_model_predictions_and_shap_values_are_finite(engine: ShapEngine) -> None:
    results = engine.explain_records(example_records())
    assert all(np.isfinite(item["churn_probability"]) for item in results)
    assert all(
        np.isfinite(float(feature["shap_value"]))
        for item in results
        for feature in item["feature_explanations"]
    )


def test_raw_margin_reconstruction_matches_saved_model(engine: ShapEngine) -> None:
    results = engine.explain_records(example_records())
    assert all(item["shap_output_space"] == SHAP_OUTPUT_SPACE for item in results)
    assert all(
        abs(float(item["reconstruction_error"])) <= engine.reconstruction_tolerance
        for item in results
    )
    assert all(
        item["reconstructed_raw_margin"]
        == pytest.approx(item["raw_margin"], abs=engine.reconstruction_tolerance)
        for item in results
    )


def test_prediction_probability_matches_sigmoid_of_raw_margin(
    engine: ShapEngine,
) -> None:
    result = engine.explain_records(example_records().iloc[[0]])[0]
    probability = 1 / (1 + np.exp(-float(result["raw_margin"])))
    assert result["churn_probability"] == pytest.approx(probability, abs=1e-5)


def test_structured_evidence_is_generated_with_source(engine: ShapEngine) -> None:
    result = engine.explain_records(example_records().iloc[[0]])[0]
    assert len(result["evidence"]) == len(FEATURES)
    assert all(item["source"] == "structured_model_feature" for item in result["evidence"])


@pytest.mark.parametrize(
    ("shap_value", "direction", "phrase"),
    [
        (0.31, "increases_risk", "toward higher churn risk"),
        (-0.18, "decreases_risk", "toward lower churn risk"),
    ],
)
def test_evidence_text_reflects_actual_shap_direction(
    shap_value: float,
    direction: str,
    phrase: str,
) -> None:
    evidence = format_feature_evidence(
        {
            "feature": "sessions",
            "feature_value": 5,
            "shap_value": shap_value,
            "direction": direction,
        }
    )
    assert "Sessions of 5" in evidence["explanation"]
    assert phrase in evidence["explanation"]


def test_no_crm_evidence_is_fabricated(engine: ShapEngine) -> None:
    result = engine.explain_records(example_records().iloc[[0]])[0]
    assert result["evidence_scope"] == "structured_features_only"
    assert "crm_evidence" not in result
    assert all("crm" not in item["source"].lower() for item in result["evidence"])


def test_missing_required_feature_is_reported(engine: ShapEngine) -> None:
    with pytest.raises(ValueError, match="Missing required model features"):
        engine.explain_records(example_records().drop(columns="sessions"))


def test_invalid_numeric_feature_uses_stage3_validation(engine: ShapEngine) -> None:
    records = example_records().iloc[[0]].copy()
    records["mrr"] = records["mrr"].astype(object)
    records.loc[records.index[0], "mrr"] = "not-a-number"
    with pytest.raises(ValueError, match="Invalid numeric values in mrr"):
        engine.explain_records(records)


def test_target_and_future_fields_are_excluded_from_explanation(
    engine: ShapEngine,
) -> None:
    records = example_records().iloc[[0]].copy()
    records["churned"] = 1
    records["churned_next_month"] = 1
    records["churn_date"] = pd.Timestamp("2025-11-01")
    records["future_mrr"] = 10000.0
    records["future_sessions"] = 0
    records["future_support_tickets"] = 99
    records["future_nps_score"] = 0
    changed = records.copy(deep=True)
    changed[
        [
            "churned",
            "churned_next_month",
            "churn_date",
            "future_mrr",
            "future_sessions",
            "future_support_tickets",
            "future_nps_score",
        ]
    ] = [0, 0, pd.NaT, 1.0, 100, 10, 10]
    first = engine.explain_records(records)[0]
    second = engine.explain_records(changed)[0]
    assert first["churn_probability"] == second["churn_probability"]
    assert first["feature_explanations"] == second["feature_explanations"]
    assert {item["feature"] for item in first["feature_explanations"]} == set(FEATURES)


def test_unidentified_input_does_not_receive_a_fabricated_id(engine: ShapEngine) -> None:
    result = engine.explain_records(example_records().drop(columns="user_id").iloc[[0]])[0]
    assert "user_id" not in result


def test_saved_predictor_artifacts_and_feature_order_are_compatible(
    engine: ShapEngine,
) -> None:
    predictor = load_churn_predictor()
    assert predictor.metadata["feature_list"] == FEATURES
    assert engine.transformed_feature_order == predictor.metadata["preprocessing"][
        "transformed_feature_order"
    ]
    assert len(engine.feature_order) == 10
    assert len(engine.transformed_feature_order) == 13


def test_driver_order_is_deterministic(engine: ShapEngine) -> None:
    records = example_records().iloc[[0]]
    first = engine.explain_records(records)[0]
    second = engine.explain_records(records)[0]
    assert first["feature_explanations"] == second["feature_explanations"]
    assert first["top_risk_drivers"] == second["top_risk_drivers"]
    assert first["protective_factors"] == second["protective_factors"]
