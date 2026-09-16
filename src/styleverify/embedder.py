"""ArcFace embedding extraction and cosine similarity helpers."""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "ArcFace"


def get_embedding(image: str | np.ndarray) -> np.ndarray:
    """Return a normalized 512-d ArcFace embedding.

    image may be a file path string or an HxWx3 uint8 numpy array.
    enforce_detection=False allows synthetic/noise images without faces.
    """
    from deepface import DeepFace  # lazy import — avoids 2-s startup at import time

    result = DeepFace.represent(
        img_path=image,
        model_name=_MODEL_NAME,
        enforce_detection=False,
    )
    embedding = np.array(result[0]["embedding"], dtype=np.float64)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1] between two embedding vectors.

    Assumes both vectors are already L2-normalized (as returned by get_embedding).
    """
    return float(np.dot(a, b))


def verify(
    img1: str | np.ndarray,
    img2: str | np.ndarray,
    threshold: float = 0.6,
) -> dict:
    """Return {"match": bool, "score": float} where score is cosine similarity."""
    a = get_embedding(img1)
    b = get_embedding(img2)
    score = cosine_similarity(a, b)
    return {"match": score >= threshold, "score": score}
