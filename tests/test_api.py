"""HTTP-layer tests for the StyleVerify FastAPI app.

DeepFace is monkeypatched so tests run without model weights.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)

UNIT_VEC = np.ones(512, dtype=np.float64) / np.sqrt(512)
ORTH_VEC = np.zeros(512, dtype=np.float64)
ORTH_VEC[0] = 1.0


def _make_png_bytes(seed: int = 0) -> bytes:
    """Create a tiny 16×16 RGB PNG in memory."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_verify_same_person(monkeypatch):
    monkeypatch.setattr("styleverify.embedder.get_embedding", lambda _: UNIT_VEC)

    png1 = _make_png_bytes(seed=1)
    png2 = _make_png_bytes(seed=2)

    resp = client.post(
        "/verify",
        files={
            "image1": ("img1.png", png1, "image/png"),
            "image2": ("img2.png", png2, "image/png"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["match"] is True
    assert abs(body["score"] - 1.0) < 1e-4
    assert body["style_detected"] in ("sketch", "pencil", "cartoon", "oil", "watercolor", "natural")
    assert body["elapsed_ms"] > 0


def test_verify_different_person(monkeypatch):
    call_count = {"n": 0}

    def _alt_embedding(_):
        call_count["n"] += 1
        return UNIT_VEC if call_count["n"] == 1 else ORTH_VEC

    monkeypatch.setattr("styleverify.embedder.get_embedding", _alt_embedding)

    png1 = _make_png_bytes(seed=3)
    png2 = _make_png_bytes(seed=4)

    resp = client.post(
        "/verify",
        files={
            "image1": ("img1.png", png1, "image/png"),
            "image2": ("img2.png", png2, "image/png"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["match"] is False
    assert abs(body["score"]) < 1e-4


def test_verify_missing_field():
    png1 = _make_png_bytes(seed=5)

    resp = client.post(
        "/verify",
        files={"image1": ("img1.png", png1, "image/png")},
    )
    assert resp.status_code == 422
