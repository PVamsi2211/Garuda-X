from collections.abc import Iterable, Mapping

import pandas as pd


SYNTHETIC_MONTHLY_NUMERIC = (
    "tenure_month",
    "monthly_price",
    "mrr",
    "sessions",
    "feature_usage_score",
    "support_tickets",
    "payment_failures",
    "churned",
    "churned_next_month",
    "nps_score",
    "product_incident",
    "active_seats",
)


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column labels without changing row data."""
    result = frame.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    return result


def convert_numeric_columns(
    frame: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    """Convert selected columns to numeric, raising for non-empty invalid values."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Cannot convert missing numeric column: {column}")
        original = result[column]
        normalized = original
        if pd.api.types.is_object_dtype(original.dtype) or pd.api.types.is_string_dtype(
            original.dtype
        ):
            stripped = original.astype("string").str.strip()
            normalized = stripped.mask(stripped.eq(""), pd.NA)
        try:
            converted = pd.to_numeric(normalized, errors="coerce")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column {column} cannot be converted to numeric") from error
        invalid = normalized.notna() & converted.isna()
        if invalid.any():
            examples = original[invalid].astype(str).drop_duplicates().head(5).tolist()
            raise ValueError(
                f"Invalid numeric values in {column}: "
                f"{int(invalid.sum())} value(s), examples={examples}"
            )
        result[column] = converted
    return result


def parse_date_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Parse selected date columns and reject non-null values that cannot be parsed."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Cannot parse missing date column: {column}")
        original = result[column]
        normalized = original
        if pd.api.types.is_object_dtype(original.dtype) or pd.api.types.is_string_dtype(
            original.dtype
        ):
            stripped = original.astype("string").str.strip()
            normalized = stripped.mask(stripped.eq(""), pd.NA)
        converted = pd.to_datetime(normalized, errors="coerce")
        invalid = normalized.notna() & converted.isna()
        if invalid.any():
            examples = original[invalid].astype(str).drop_duplicates().head(5).tolist()
            raise ValueError(
                f"Invalid date values in {column}: "
                f"{int(invalid.sum())} value(s), examples={examples}"
            )
        result[column] = converted
    return result


def normalize_categorical_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    lowercase: bool = False,
) -> pd.DataFrame:
    """Trim categorical text and optionally normalize its case."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Cannot normalize missing categorical column: {column}")
        values = result[column].astype("string").str.strip()
        if lowercase:
            values = values.str.lower()
        result[column] = values.mask(values.eq(""), pd.NA)
    return result


def drop_duplicate_rows(
    frame: pd.DataFrame,
    *,
    subset: Iterable[str] | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Drop exact duplicate rows by default; conflicting identifier rows are retained."""
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last', or False")
    if subset is not None:
        missing = sorted(set(subset) - set(frame.columns))
        if missing:
            raise ValueError(f"Duplicate subset columns are missing: {', '.join(missing)}")
    columns = list(frame.columns) if subset is None else list(subset)
    keys = frame[columns].apply(lambda column: column.map(_hashable_value))
    duplicate_mask = keys.duplicated(keep=keep)
    return frame.loc[~duplicate_mask].copy()


def _hashable_value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return ("__missing__",)
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _hashable_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_hashable_value(item) for item in value), key=repr))
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return ("__missing__",)
    except (TypeError, ValueError):
        pass
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def handle_missing_values(
    frame: pd.DataFrame, *, required_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Drop rows missing caller-declared critical fields and leave other nulls intact.

    No values are imputed. Feature nulls remain visible for validation or a later,
    explicitly documented modeling policy.
    """
    required = list(required_columns)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Required missing-value fields are absent: {', '.join(missing)}")
    if not required:
        return frame.copy()
    return frame.dropna(subset=required).copy()


def clean_synthetic_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic type and category normalization to user-month rows.

    Null feature values are preserved. Exact duplicate rows are removed; rows that
    conflict at the same user-month key remain for schema validation to flag.
    """
    result = frame.copy()
    numeric_columns = [column for column in SYNTHETIC_MONTHLY_NUMERIC if column in result]
    result = convert_numeric_columns(result, numeric_columns)
    if "month" in result:
        result = parse_date_columns(result, ["month"])
    if "plan_type" in result:
        result = normalize_categorical_columns(result, ["plan_type"], lowercase=True)
    return drop_duplicate_rows(result)
