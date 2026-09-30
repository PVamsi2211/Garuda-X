from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd
from pandas.api import types as ptypes

SYNTHETIC_MONTHLY_REQUIRED = (
    "user_id",
    "month",
    "mrr",
    "sessions",
    "feature_usage_score",
    "support_tickets",
    "payment_failures",
    "churned_next_month",
)

CRM_REQUIRED = (
    "conversation_id",
    "churn_risk_level",
    "churn_signals",
    "conversation",
    "summary",
)

ACCOUNT_TABLE_SCHEMAS = {
    "account_analysis": {
        "required": ("account_id",),
        "unique": (("account_id",),),
    },
    "subscription_revenue_analysis": {
        "required": ("subscription_id", "account_id"),
        "unique": (("subscription_id",),),
    },
    "support_analysis": {
        "required": ("account_id",),
        "unique": (("account_id",),),
    },
    "usage_analysis": {
        "required": ("subscription_id", "account_id"),
        "unique": (("subscription_id",),),
    },
    "python_insights_summary": {
        "required": ("analysis_area", "key_finding"),
        "unique": (),
    },
}


@dataclass
class ValidationReport:
    dataset_name: str
    missing_required_columns: list[str] = field(default_factory=list)
    duplicate_identifiers: dict[str, int] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    non_nullable_null_counts: dict[str, int] = field(default_factory=dict)
    datatype_problems: dict[str, str] = field(default_factory=dict)
    invalid_target_values: dict[str, list[object]] = field(default_factory=dict)
    invalid_numeric_ranges: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(
            (
                self.missing_required_columns,
                self.duplicate_identifiers,
                self.non_nullable_null_counts,
                self.datatype_problems,
                self.invalid_target_values,
                self.invalid_numeric_ranges,
            )
        )

    @property
    def errors(self) -> list[str]:
        messages: list[str] = []
        if self.missing_required_columns:
            messages.append(
                "missing required columns: "
                + ", ".join(self.missing_required_columns)
            )
        if self.duplicate_identifiers:
            messages.extend(
                f"duplicate identifier {key}: {count} duplicate rows"
                for key, count in self.duplicate_identifiers.items()
            )
        if self.non_nullable_null_counts:
            messages.extend(
                f"null values in required field {column}: {count}"
                for column, count in self.non_nullable_null_counts.items()
            )
        if self.datatype_problems:
            messages.extend(
                f"datatype problem in {column}: {problem}"
                for column, problem in self.datatype_problems.items()
            )
        if self.invalid_target_values:
            messages.extend(
                f"invalid values in target {column}: {values}"
                for column, values in self.invalid_target_values.items()
            )
        if self.invalid_numeric_ranges:
            messages.extend(
                f"values outside valid numeric range in {column}: {count}"
                for column, count in self.invalid_numeric_ranges.items()
            )
        return messages

    def raise_if_invalid(self) -> None:
        """Raise a readable error when this report contains schema problems."""
        if not self.is_valid:
            raise ValueError(f"Invalid {self.dataset_name}: " + "; ".join(self.errors))


def validate_required_columns(
    frame: pd.DataFrame, required_columns: list[str] | tuple[str, ...]
) -> None:
    """Raise when any requested column is absent."""
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _matches_type(series: pd.Series, expected: str) -> bool:
    if expected == "numeric":
        return ptypes.is_numeric_dtype(series.dtype)
    if expected == "text":
        return (
            ptypes.is_string_dtype(series.dtype)
            or ptypes.is_object_dtype(series.dtype)
            or ptypes.is_categorical_dtype(series.dtype)
        )
    if expected == "date":
        if ptypes.is_datetime64_any_dtype(series.dtype):
            return True
        values = series.dropna()
        return values.empty or pd.to_datetime(values, errors="coerce").notna().all()
    if expected == "sequence":
        return series.dropna().map(lambda value: isinstance(value, (list, tuple))).all()
    raise ValueError(f"Unknown expected datatype: {expected}")


def _validate_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    required: tuple[str, ...],
    unique_keys: tuple[tuple[str, ...], ...] = (),
    expected_types: Mapping[str, str] | None = None,
    target_values: Mapping[str, set[object]] | None = None,
    numeric_ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
    nullable_columns: set[str] | frozenset[str] = frozenset(),
) -> ValidationReport:
    report = ValidationReport(dataset_name=dataset_name)
    report.missing_required_columns = sorted(set(required) - set(frame.columns))
    report.null_counts = {column: int(count) for column, count in frame.isna().sum().items()}
    critical_columns = (set(required) - set(nullable_columns)) | set(target_values or {})
    report.non_nullable_null_counts = {
        column: count
        for column, count in report.null_counts.items()
        if column in critical_columns and count > 0
    }

    for key in unique_keys:
        if all(column in frame.columns for column in key):
            count = int(frame.duplicated(subset=list(key)).sum())
            if count:
                report.duplicate_identifiers[" + ".join(key)] = count

    for column, expected in (expected_types or {}).items():
        if column in frame.columns and not _matches_type(frame[column], expected):
            report.datatype_problems[column] = (
                f"expected {expected}, received {frame[column].dtype}"
            )

    for column, allowed in (target_values or {}).items():
        if column not in frame.columns:
            continue
        invalid: list[object] = []
        for value in frame[column].dropna().unique():
            try:
                accepted = value in allowed
            except TypeError:
                accepted = False
            if not accepted:
                invalid.append(value.item() if hasattr(value, "item") else value)
        if invalid:
            report.invalid_target_values[column] = invalid[:20]

    for column, (minimum, maximum) in (numeric_ranges or {}).items():
        if column not in frame.columns or not ptypes.is_numeric_dtype(frame[column].dtype):
            continue
        values = frame[column].dropna()
        invalid_mask = pd.Series(False, index=values.index)
        if minimum is not None:
            invalid_mask |= values < minimum
        if maximum is not None:
            invalid_mask |= values > maximum
        invalid_mask |= ~values.map(lambda value: bool(pd.notna(value) and abs(value) != float("inf")))
        count = int(invalid_mask.sum())
        if count:
            report.invalid_numeric_ranges[column] = count
    return report


def validate_synthetic_monthly(frame: pd.DataFrame) -> ValidationReport:
    """Validate the user-month grain and target schema for churn training."""
    expected = {
        "user_id": "text",
        "month": "date",
        "mrr": "numeric",
        "monthly_price": "numeric",
        "sessions": "numeric",
        "feature_usage_score": "numeric",
        "support_tickets": "numeric",
        "payment_failures": "numeric",
        "churned": "numeric",
        "churned_next_month": "numeric",
        "plan_type": "text",
        "nps_score": "numeric",
        "product_incident": "numeric",
        "active_seats": "numeric",
        "tenure_month": "numeric",
    }
    ranges = {
        "mrr": (0, None),
        "sessions": (0, None),
        "feature_usage_score": (0, 100),
        "support_tickets": (0, None),
        "payment_failures": (0, None),
        "churned": (0, 1),
        "nps_score": (0, 10),
        "product_incident": (0, 1),
        "active_seats": (0, None),
        "tenure_month": (0, None),
    }
    return _validate_frame(
        frame,
        dataset_name="synthetic monthly dataset",
        required=SYNTHETIC_MONTHLY_REQUIRED,
        unique_keys=(("user_id", "month"),),
        expected_types=expected,
        target_values={"churned_next_month": {0, 1}},
        numeric_ranges=ranges,
    )


def validate_prediction_monthly(frame: pd.DataFrame) -> ValidationReport:
    """Validate unlabeled user-month rows for next-month churn inference."""
    feature_list: tuple[str, ...] = (
        "mrr",
        "sessions",
        "feature_usage_score",
        "support_tickets",
        "payment_failures",
        "nps_score",
        "product_incident",
        "active_seats",
        "tenure_month",
        "plan_type",
    )
    required = ("user_id", "month", *feature_list)
    expected = {
        "user_id": "text",
        "month": "date",
        **{column: "numeric" for column in feature_list if column != "plan_type"},
        "plan_type": "text",
    }
    ranges = {
        "mrr": (0, None),
        "sessions": (0, None),
        "feature_usage_score": (0, 100),
        "support_tickets": (0, None),
        "payment_failures": (0, None),
        "nps_score": (0, 10),
        "product_incident": (0, 1),
        "active_seats": (0, None),
        "tenure_month": (0, None),
    }
    return _validate_frame(
        frame,
        dataset_name="monthly prediction dataset",
        required=required,
        unique_keys=(("user_id", "month"),),
        expected_types=expected,
        numeric_ranges=ranges,
        nullable_columns=set(feature_list) - {"mrr"},
    )


def validate_synthetic_users(frame: pd.DataFrame) -> ValidationReport:
    """Validate the one-row-per-user companion table."""
    return _validate_frame(
        frame,
        dataset_name="synthetic users dataset",
        required=("user_id",),
        unique_keys=(("user_id",),),
        expected_types={
            "user_id": "text",
            "signup_date": "date",
            "country": "text",
            "company_size": "numeric",
            "plan_type": "text",
            "monthly_price": "numeric",
            "sessions_per_month": "numeric",
            "feature_usage_score": "numeric",
            "support_tickets": "numeric",
            "payment_failures": "numeric",
            "last_active_date": "date",
            "churned": "numeric",
            "churn_date": "date",
            "tenure_months": "numeric",
            "total_revenue": "numeric",
            "ltv_12m": "numeric",
            "industry": "text",
            "acquisition_channel": "text",
            "billing_period": "text",
            "seats_purchased": "numeric",
            "discount_pct": "numeric",
            "payment_method": "text",
        },
        target_values={"churned": {0, 1}},
        numeric_ranges={
            "company_size": (0, None),
            "monthly_price": (0, None),
            "feature_usage_score": (0, 100),
            "support_tickets": (0, None),
            "payment_failures": (0, None),
            "tenure_months": (0, None),
            "total_revenue": (0, None),
            "ltv_12m": (0, None),
            "seats_purchased": (0, None),
            "discount_pct": (0, 100),
        },
    )


def validate_account_table(frame: pd.DataFrame, table_name: str) -> ValidationReport:
    """Validate one table from the account intelligence dataset."""
    if table_name not in ACCOUNT_TABLE_SCHEMAS:
        raise ValueError(f"Unknown account dataset table: {table_name}")
    schema = ACCOUNT_TABLE_SCHEMAS[table_name]
    required = schema["required"]
    expected_types = {column: "text" for column in ("account_id", "subscription_id")}
    expected_types.update(
        {
            column: "date"
            for column in ("signup_date", "start_date", "end_date")
        }
    )
    expected_types.update(
        {
            column: "numeric"
            for column in (
                "seats",
                "is_trial",
                "churn_flag",
                "total_mrr",
                "total_arr",
                "avg_mrr",
                "subscription_count",
                "churned_subscription_records",
                "total_tickets",
                "avg_resolution_time_hours",
                "avg_first_response_time_minutes",
                "avg_satisfaction_score",
                "total_escalations",
                "mrr_amount",
                "arr_amount",
                "upgrade_flag",
                "downgrade_flag",
                "auto_renew_flag",
                "total_usage_count",
                "avg_usage_duration_secs",
                "total_errors",
                "beta_feature_events",
                "unique_features_used",
            )
        }
    )
    expected_types.update(
        {
            column: "text"
            for column in (
                "analysis_area",
                "key_finding",
                "business_interpretation",
                "recommended_action",
                "account_name",
                "industry",
                "country",
                "referral_source",
                "plan_tier",
                "billing_frequency",
            )
        }
    )
    targets = {"churn_flag": {0, 1}, "is_trial": {0, 1}}
    targets.update(
        {column: {0, 1} for column in ("upgrade_flag", "downgrade_flag", "auto_renew_flag")}
    )
    ranges = {
        column: (0, None)
        for column in (
            "seats",
            "total_mrr",
            "total_arr",
            "avg_mrr",
            "mrr_amount",
            "arr_amount",
            "total_tickets",
            "total_escalations",
            "avg_resolution_time_hours",
            "avg_first_response_time_minutes",
            "avg_satisfaction_score",
        )
    }
    return _validate_frame(
        frame,
        dataset_name=f"account dataset table '{table_name}'",
        required=required,
        unique_keys=schema["unique"],
        expected_types=expected_types,
        target_values=targets,
        numeric_ranges=ranges,
    )


def validate_account_dataset(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, ValidationReport]:
    """Validate any supplied tables from the account intelligence dataset."""
    return {name: validate_account_table(frame, name) for name, frame in tables.items()}


def validate_crm_dataset(frame: pd.DataFrame) -> ValidationReport:
    """Validate CRM conversation rows without joining them to account records."""
    return _validate_frame(
        frame,
        dataset_name="CRM conversation dataset",
        required=CRM_REQUIRED,
        unique_keys=(("conversation_id",),),
        expected_types={
            "conversation_id": "text",
            "churn_risk_level": "text",
            "churn_signals": "sequence",
            "conversation": "sequence",
            "summary": "text",
            "channel": "text",
            "product_category": "text",
            "customer_tenure": "text",
            "sentiment_arc": "text",
            "resolution_outcome": "text",
            "injection_style": "text",
            "customer_persona": "text",
            "agent_persona": "text",
            "company_size": "text",
            "plan_name": "text",
            "plan_type": "text",
            "participants": "sequence",
            "tenure_months": "numeric",
            "seats": "numeric",
            "active_seats": "numeric",
            "per_seat_price_usd": "numeric",
            "mrr_usd": "numeric",
            "discount_offered_pct": "numeric",
            "turn_count": "numeric",
            "word_count": "numeric",
            "quality_score": "numeric",
            "stale_phrase_count": "numeric",
        },
        target_values={"churn_risk_level": {"low", "medium", "high", "churned"}},
        numeric_ranges={
            "tenure_months": (0, None),
            "seats": (0, None),
            "active_seats": (0, None),
            "per_seat_price_usd": (0, None),
            "mrr_usd": (0, None),
            "discount_offered_pct": (15, 30),
            "turn_count": (0, None),
            "word_count": (0, None),
            "quality_score": (0, 1),
            "stale_phrase_count": (0, None),
        },
    )
