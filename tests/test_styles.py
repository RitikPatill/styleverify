"""Tests for src/styleverify/styles.py (M3)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from styleverify import styles


def _sample_image(seed: int = 0, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Parametrized tests over all five transforms
# ---------------------------------------------------------------------------

STYLE_NAMES_TRANSFORMS = ["sketch", "pencil", "cartoon", "oil", "watercolor"]


@pytest.mark.parametrize("name", STYLE_NAMES_TRANSFORMS)
def test_each_transform_returns_rgb_image(name: str) -> None:
    img = _sample_image()
    fn = styles.STYLE_FNS[name]
    result = fn(img)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == img.size


@pytest.mark.parametrize("name", STYLE_NAMES_TRANSFORMS)
def test_each_transform_modifies_pixels(name: str) -> None:
    img = _sample_image()
    fn = styles.STYLE_FNS[name]
    result = fn(img)
    assert not np.array_equal(np.array(result), np.array(img)), (
        f"Transform '{name}' did not change any pixels"
    )


# ---------------------------------------------------------------------------
# detect_style tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STYLE_NAMES_TRANSFORMS)
def test_detect_style_returns_valid_name(name: str) -> None:
    img = _sample_image()
    fn = styles.STYLE_FNS[name]
    styled = fn(img)
    detected = styles.detect_style(styled)
    assert detected in styles.STYLE_NAMES, (
        f"detect_style returned '{detected}' which is not in STYLE_NAMES"
    )


def test_detect_style_natural_on_photo() -> None:
    img = _sample_image()
    result = styles.detect_style(img)
    assert isinstance(result, str)
    assert result in styles.STYLE_NAMES


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_style_fns_dict_complete() -> None:
    assert set(styles.STYLE_FNS) == {"sketch", "pencil", "cartoon", "oil", "watercolor"}


def test_style_names_contains_natural() -> None:
    assert "natural" in styles.STYLE_NAMES
