"""Unit tests for src/styleverify/embedder.py.

Uses synthetic 112x112 numpy arrays — no I/O, no face detection required.
First run may be slow (~70 MB weight download); subsequent runs use cache.
"""

import numpy as np
import pytest

from styleverify import embedder


def _make_face_image(seed: int) -> np.ndarray:
    """Return a 112x112 uint8 RGB array; pseudo-random pixel fill."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)


@pytest.mark.integration
def test_same_image_high_similarity():
    arr = _make_face_image(42)
    score = embedder.cosine_similarity(
        embedder.get_embedding(arr),
        embedder.get_embedding(arr),
    )
    assert score >= 0.99, f"Expected same-image score >= 0.99, got {score:.4f}"


@pytest.mark.integration
def test_different_seeds_score_below_threshold():
    a = embedder.get_embedding(_make_face_image(0))
    b = embedder.get_embedding(_make_face_image(999))
    score = embedder.cosine_similarity(a, b)
    assert score < 0.6, f"Expected different-image score < 0.6, got {score:.4f}"
