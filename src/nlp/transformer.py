from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "transformer" / "all-MiniLM-L6-v2-deploy"


def normalize_embedding_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.split())


@lru_cache(maxsize=4)
def _load_cached_model(model_path: str, device: str) -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_path,
        device=device,
        local_files_only=True,
        model_kwargs={"local_files_only": True},
    )


def load_embedding_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    device: str = "cpu",
) -> object:
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"Local MiniLM model directory does not exist: {resolved_path}")
    if not device:
        raise ValueError("device must be a non-empty string")
    return _load_cached_model(str(resolved_path), device)


def get_embedding_dimension(model: object) -> int | None:
    dimension_method = getattr(model, "get_embedding_dimension", None)
    if not callable(dimension_method):
        dimension_method = getattr(model, "get_sentence_embedding_dimension", None)
    dimension = dimension_method() if callable(dimension_method) else None
    return None if dimension is None else int(dimension)


def encode_texts(
    texts: Sequence[str],
    *,
    model: object | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    batch_size: int = 32,
) -> np.ndarray:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not a single string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    normalized = [normalize_embedding_text(text) for text in texts]
    if not normalized:
        selected_model = model if model is not None else load_embedding_model(model_path)
        dimension = get_embedding_dimension(selected_model)
        if dimension is None:
            raise ValueError("The embedding model did not report its output dimension")
        return np.empty((0, dimension), dtype=np.float32)
    if any(not text for text in normalized):
        raise ValueError("texts must not contain empty or whitespace-only values")
    selected_model = model if model is not None else load_embedding_model(model_path)
    vectors = np.asarray(
        selected_model.encode(
            normalized,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(normalized):
        raise RuntimeError("MiniLM returned embeddings with an unexpected shape")
    if not np.isfinite(vectors).all():
        raise RuntimeError("MiniLM returned non-finite embedding values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise RuntimeError("MiniLM embeddings were not normalized")
    return vectors


def encode_query(
    query: str,
    *,
    model: object | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> np.ndarray:
    return encode_texts([query], model=model, model_path=model_path)[0]
