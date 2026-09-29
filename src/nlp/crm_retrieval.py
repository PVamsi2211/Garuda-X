from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.datasets import get_dataset_path
from src.data.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.nlp.transformer import (
    DEFAULT_MODEL_PATH,
    encode_query,
    encode_texts,
    get_embedding_dimension,
    load_embedding_model,
    normalize_embedding_text,
)

DEFAULT_CRM_PATH = get_dataset_path("customer_churn_v2", "full")
APPROVED_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
FEATURE_QUERY_TERMS = {
    "mrr": "monthly recurring revenue billing price",
    "sessions": "sessions product usage engagement",
    "feature_usage_score": "feature usage product adoption",
    "support_tickets": "support tickets customer support",
    "payment_failures": "payment failures billing payments",
    "nps_score": "NPS customer satisfaction feedback",
    "product_incident": "product incident service issue",
    "active_seats": "active seats seat usage",
    "tenure_month": "customer tenure account age",
    "plan_type": "subscription plan",
}
_METADATA_COLUMNS = (
    "user_id",
    "account_id",
    "interaction_id",
    "date",
    "interaction_type",
    "source_row",
)


def _is_missing(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _conversation_text(value: object) -> str | None:
    if not isinstance(value, list):
        return None
    parts = [
        turn["text"]
        for turn in value
        if isinstance(turn, dict) and isinstance(turn.get("text"), str)
    ]
    return "\n".join(parts) if parts else None


def load_crm_records(path: str | Path = DEFAULT_CRM_PATH) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"CRM interaction file does not exist: {source_path}")
    if source_path.suffix.lower() != ".jsonl":
        raise ValueError("CRM loader currently supports the project's JSONL interaction source")

    schema_path = source_path.parent / "schema.json"
    schema: dict[str, Any] = {}
    schema_fields: set[str] = set()
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_fields = set(schema.get("fields", {}))
    identity_fields = [name for name in ("user_id", "account_id") if name in schema_fields]
    interaction_id_field = next(
        (name for name in ("interaction_id", "conversation_id") if name in schema_fields),
        None,
    )
    date_field = next(
        (
            name
            for name in (
                "interaction_date",
                "contact_date",
                "created_at",
                "date",
                "timestamp",
            )
            if name in schema_fields
        ),
        None,
    )
    type_field = next(
        (name for name in ("interaction_type", "channel", "type") if name in schema_fields),
        None,
    )
    text_field = next(
        (name for name in ("text", "original_text", "conversation") if name in schema_fields),
        None,
    )
    if text_field is None:
        raise ValueError(f"No interaction text field is described in {schema_path}")

    normalized: list[dict[str, object]] = []
    with source_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            source_record = json.loads(line)
            raw_text = source_record.get(text_field)
            text = _conversation_text(raw_text) if text_field == "conversation" else raw_text
            item: dict[str, object] = {
                "date": source_record.get(date_field) if date_field else None,
                "interaction_type": source_record.get(type_field) if type_field else None,
                "text": text,
                "source_row": line_number,
            }
            for name in identity_fields:
                item[name] = source_record.get(name)
            if interaction_id_field:
                item["interaction_id"] = source_record.get(interaction_id_field)
            normalized.append(item)
    return pd.DataFrame(normalized)


def _record_dicts(
    records: pd.DataFrame | Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[object]]:
    if isinstance(records, pd.DataFrame):
        return records.to_dict(orient="records"), records.index.tolist()
    if isinstance(records, (str, bytes)):
        raise TypeError("records must be a DataFrame or an iterable of record mappings")
    result: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each CRM record must be a mapping")
        result.append(dict(record))
    return result, list(range(len(result)))


def _normalize_records(
    records: pd.DataFrame | Iterable[Mapping[str, object]],
) -> tuple[pd.DataFrame, int]:
    source_records, source_indexes = _record_dicts(records)
    normalized: list[dict[str, object]] = []
    invalid_text_count = 0
    for record, source_index in zip(source_records, source_indexes):
        raw_text = record.get("text")
        if _is_missing(raw_text):
            invalid_text_count += 1
            continue
        if isinstance(raw_text, str):
            original_text = raw_text
        elif isinstance(raw_text, (int, float, np.number)) and not isinstance(
            raw_text, (bool, np.bool_)
        ):
            original_text = str(raw_text)
        else:
            invalid_text_count += 1
            continue
        embedding_text = normalize_embedding_text(original_text)
        if not embedding_text:
            invalid_text_count += 1
            continue

        item = {
            key: record[key]
            for key in _METADATA_COLUMNS
            if key in record
        }
        if "interaction_id" not in item:
            item["interaction_id"] = record.get("conversation_id")
        if "interaction_type" not in item:
            item["interaction_type"] = record.get("channel")
        if "date" not in item:
            item["date"] = record.get("interaction_date")
        if "source_row" not in item:
            item["source_row"] = source_index
        item["text"] = original_text
        item["embedding_text"] = embedding_text
        normalized.append(item)
    columns = list(_METADATA_COLUMNS) + ["text", "embedding_text"]
    return pd.DataFrame(normalized, columns=columns), invalid_text_count


def _scope_key(value: object) -> str:
    return str(value)


def _format_query_value(value: object) -> str:
    if _is_missing(value):
        return "missing"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if not math.isfinite(number):
            return "missing"
        return format(number, ".12g")
    return str(value)


def build_retrieval_query(structured_explanation: Mapping[str, Any]) -> str:
    if not isinstance(structured_explanation, Mapping):
        raise TypeError("structured_explanation must be a mapping")
    drivers = structured_explanation.get("top_risk_drivers", [])
    if not isinstance(drivers, list):
        raise TypeError("top_risk_drivers must be a list")
    if not drivers:
        drivers = structured_explanation.get("feature_explanations", [])
    normalized_drivers: list[dict[str, object]] = []
    for driver in drivers:
        if not isinstance(driver, Mapping):
            raise TypeError("each structured driver must be a mapping")
        feature = driver.get("feature")
        if feature not in APPROVED_FEATURES:
            continue
        value = driver.get("feature_value")
        shap_value = driver.get("shap_value")
        if shap_value is not None:
            try:
                numeric_shap = float(shap_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("SHAP driver value must be numeric") from error
            if not math.isfinite(numeric_shap):
                raise ValueError("SHAP driver value must be finite")
            if numeric_shap <= 0 and structured_explanation.get("top_risk_drivers"):
                continue
            importance = abs(numeric_shap)
        else:
            importance = 0.0
        normalized_drivers.append(
            {"feature": str(feature), "feature_value": value, "importance": importance}
        )
    normalized_drivers.sort(
        key=lambda item: (-float(item["importance"]), str(item["feature"]))
    )

    query_parts: list[str] = []
    seen: set[str] = set()
    for driver in normalized_drivers:
        feature = str(driver["feature"])
        if feature in seen:
            continue
        seen.add(feature)
        query_parts.append(
            f"{FEATURE_QUERY_TERMS[feature]} "
            f"(current {feature.replace('_', ' ')} value "
            f"{_format_query_value(driver['feature_value'])})"
        )
    return "; ".join(query_parts)


def _valid_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return int(top_k)


def _valid_min_similarity(min_similarity: float | None) -> float | None:
    if min_similarity is None:
        return None
    if isinstance(min_similarity, (bool, np.bool_)):
        raise TypeError("min_similarity must be numeric")
    try:
        value = float(min_similarity)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("min_similarity must be numeric") from error
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("min_similarity must be finite and within [-1, 1]")
    return value


def _valid_as_of(as_of: object | None) -> pd.Timestamp | None:
    if as_of is None:
        return None
    try:
        value = pd.to_datetime(as_of, errors="raise", utc=True)
    except (TypeError, ValueError) as error:
        raise ValueError("as_of must be a valid date or timestamp") from error
    return pd.Timestamp(value)


class CRMEmbeddingIndex:
    def __init__(
        self,
        records: pd.DataFrame | Iterable[Mapping[str, object]],
        *,
        model: object | None = None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scope_column: str | None = None,
        batch_size: int = 32,
    ) -> None:
        normalized, invalid_text_count = _normalize_records(records)
        self.original_record_count = len(normalized) + invalid_text_count
        self.invalid_text_count = invalid_text_count
        self.model = model if model is not None else load_embedding_model(model_path)
        self.embedding_dimension = get_embedding_dimension(self.model)
        if self.embedding_dimension is None:
            raise ValueError("MiniLM did not report its embedding dimension")

        available_scopes = [
            column for column in ("user_id", "account_id") if column in normalized.columns
        ]
        if scope_column is not None and scope_column not in available_scopes:
            raise ValueError(
                f"scope_column must be one of the record identifiers: {available_scopes}"
            )
        if scope_column is None:
            scope_column = next(
                (
                    column
                    for column in available_scopes
                    if normalized[column].notna().any()
                ),
                None,
            )
        self.scope_column = scope_column or next(
            (
                column
                for column in available_scopes
                if normalized[column].notna().any()
            ),
            None,
        )
        if available_scopes:
            scope_presence = normalized[available_scopes].notna().any(axis=1)
            self.unscoped_record_count = int((~scope_presence).sum())
            scoped = normalized.loc[scope_presence].copy()
        else:
            self.unscoped_record_count = len(normalized)
            scoped = normalized.iloc[0:0].copy()

        self.scope_record_counts: dict[str, int] = {}
        if self.scope_column is not None:
            counts = scoped[self.scope_column].dropna().map(_scope_key).value_counts()
            self.scope_record_counts = {
                str(value): int(count) for value, count in counts.items()
            }
        self.records = scoped.reset_index(drop=True)
        self.embeddings = encode_texts(
            self.records["embedding_text"].tolist(),
            model=self.model,
            batch_size=batch_size,
        )
        if self.embeddings.shape != (
            len(self.records),
            self.embedding_dimension,
        ):
            raise RuntimeError("CRM embedding matrix has an unexpected dimension")
        self.records["embedding"] = [
            vector.copy() for vector in self.embeddings
        ]
        self._scope_positions: dict[str, dict[str, list[int]]] = {
            column: {} for column in available_scopes
        }
        for position, record in enumerate(self.records.to_dict(orient="records")):
            for column in available_scopes:
                scope_id = record.get(column)
                if not _is_missing(scope_id):
                    self._scope_positions[column].setdefault(
                        _scope_key(scope_id), []
                    ).append(position)

    def retrieve(
        self,
        *,
        query: str,
        user_id: object | None = None,
        account_id: object | None = None,
        top_k: int = 3,
        min_similarity: float | None = None,
        as_of: object | None = None,
    ) -> dict[str, object]:
        limit = _valid_top_k(top_k)
        threshold = _valid_min_similarity(min_similarity)
        cutoff = _valid_as_of(as_of)
        cutoff_metadata = {"as_of": cutoff.isoformat()} if cutoff is not None else {}
        if (user_id is None) == (account_id is None):
            raise ValueError("provide exactly one of user_id or account_id")
        scope_column = "user_id" if user_id is not None else "account_id"
        scope_id = user_id if user_id is not None else account_id
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        retrieval_query = normalize_embedding_text(query)
        result_key = {scope_column: scope_id}

        if not retrieval_query:
            return {
                **result_key,
                "retrieval_query": retrieval_query,
                "crm_evidence": [],
                "evidence_status": "empty_query",
                **cutoff_metadata,
            }
        positions = (
            self._scope_positions.get(scope_column, {}).get(_scope_key(scope_id), [])
            if scope_column in self._scope_positions
            else []
        )
        if not positions:
            return {
                **result_key,
                "retrieval_query": retrieval_query,
                "crm_evidence": [],
                "evidence_status": "no_matching_crm_records",
                **cutoff_metadata,
            }
        if cutoff is not None:
            candidate_records = self.records.iloc[positions]
            if "date" not in candidate_records:
                positions = []
            else:
                candidate_dates = pd.to_datetime(
                    candidate_records["date"], errors="coerce", utc=True
                )
                positions = [
                    position
                    for position, eligible in zip(
                        positions, (candidate_dates.notna() & (candidate_dates <= cutoff))
                    )
                    if bool(eligible)
                ]
            if not positions:
                return {
                    **result_key,
                    "retrieval_query": retrieval_query,
                    "crm_evidence": [],
                    "evidence_status": "no_records_as_of",
                    "as_of": cutoff.isoformat(),
                }

        query_vector = encode_query(retrieval_query, model=self.model)
        candidate_embeddings = self.embeddings[positions]
        similarities = np.clip(candidate_embeddings @ query_vector, -1.0, 1.0)
        ranked_positions = sorted(
            range(len(positions)),
            key=lambda local_position: (
                -float(similarities[local_position]),
                str(
                    self.records.iloc[positions[local_position]].get("interaction_id")
                    or self.records.iloc[positions[local_position]].get("source_row")
                    or ""
                ),
                positions[local_position],
            ),
        )
        evidence: list[dict[str, object]] = []
        for local_position in ranked_positions:
            score = float(similarities[local_position])
            if threshold is not None and score < threshold:
                continue
            record = self.records.iloc[positions[local_position]]
            item: dict[str, object] = {
                **result_key,
                "date": record.get("date"),
                "interaction_type": record.get("interaction_type"),
                "text": record["text"],
                "similarity_score": score,
                "rank": len(evidence) + 1,
                "source": "crm_interaction",
            }
            interaction_id = record.get("interaction_id")
            if not _is_missing(interaction_id):
                item["interaction_id"] = interaction_id
            source_row = record.get("source_row")
            if not _is_missing(source_row):
                item["source_row"] = source_row
            evidence.append(item)
            if len(evidence) == limit:
                break
        status = (
            "retrieved"
            if evidence
            else "below_similarity_threshold"
        )
        return {
            **result_key,
            "retrieval_query": retrieval_query,
            "crm_evidence": evidence,
            "evidence_status": status,
            **cutoff_metadata,
        }

    def retrieve_many(
        self,
        queries: Mapping[object, str],
        *,
        scope_column: str | None = None,
        top_k: int = 3,
        min_similarity: float | None = None,
        as_of: object | None = None,
    ) -> list[dict[str, object]]:
        if not isinstance(queries, Mapping):
            raise TypeError("queries must map account identifiers to retrieval queries")
        selected_scope = scope_column or self.scope_column
        if selected_scope not in {"user_id", "account_id"}:
            raise ValueError("scope_column must be user_id or account_id")
        return [
            self.retrieve(
                query=query,
                **{selected_scope: scope_id},
                top_k=top_k,
                min_similarity=min_similarity,
                as_of=as_of,
            )
            for scope_id, query in queries.items()
        ]


def retrieve_account_evidence(
    index: CRMEmbeddingIndex,
    *,
    structured_explanation: Mapping[str, Any],
    user_id: object | None = None,
    account_id: object | None = None,
    top_k: int = 3,
    min_similarity: float | None = None,
    as_of: object | None = None,
) -> dict[str, object]:
    if (user_id is None) == (account_id is None):
        raise ValueError("provide exactly one of user_id or account_id")
    query = build_retrieval_query(structured_explanation)
    result = index.retrieve(
        query=query,
        user_id=user_id,
        account_id=account_id,
        top_k=top_k,
        min_similarity=min_similarity,
        as_of=as_of,
    )
    identity_key = "user_id" if user_id is not None else "account_id"
    return {
        identity_key: user_id if user_id is not None else account_id,
        "structured_evidence": list(structured_explanation.get("evidence", [])),
        "retrieval_query": query,
        "crm_evidence": result["crm_evidence"],
        "evidence_status": result["evidence_status"],
        **({"as_of": result["as_of"]} if "as_of" in result else {}),
    }


def retrieve_crm_evidence(
    query: str,
    index: CRMEmbeddingIndex,
    *,
    user_id: object | None = None,
    account_id: object | None = None,
    top_k: int = 3,
    min_similarity: float | None = None,
    as_of: object | None = None,
) -> dict[str, object]:
    return index.retrieve(
        query=query,
        user_id=user_id,
        account_id=account_id,
        top_k=top_k,
        min_similarity=min_similarity,
        as_of=as_of,
    )
