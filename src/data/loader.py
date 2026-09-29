from pathlib import Path
from typing import TypeAlias

import pandas as pd

DataPath: TypeAlias = str | Path
SUPPORTED_FORMATS = {".csv", ".jsonl", ".parquet"}


def _require_file(path: DataPath) -> Path:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    return source


def load_csv(path: DataPath) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    source = _require_file(path)
    try:
        return pd.read_csv(source)
    except Exception as error:
        raise ValueError(f"Could not read CSV dataset {source}: {error}") from error


def load_jsonl(path: DataPath) -> pd.DataFrame:
    """Load a JSON Lines file, preserving nested values as object columns."""
    source = _require_file(path)
    try:
        return pd.read_json(source, lines=True)
    except Exception as error:
        raise ValueError(f"Could not read JSONL dataset {source}: {error}") from error


def load_parquet(path: DataPath) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    source = _require_file(path)
    try:
        return pd.read_parquet(source)
    except Exception as error:
        raise ValueError(f"Could not read Parquet dataset {source}: {error}") from error


def load_data(path: DataPath) -> pd.DataFrame:
    """Load a supported dataset based on its file extension."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported dataset format '{suffix or '<none>'}' for {source}; "
            f"supported formats are: {supported}"
        )
    loaders = {".csv": load_csv, ".jsonl": load_jsonl, ".parquet": load_parquet}
    return loaders[suffix](source)
