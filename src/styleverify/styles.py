"""Deterministic image style transforms for StyleVerify (M3).

All public functions accept and return ``PIL.Image.Image`` in RGB mode.
No OpenCV dependency — only Pillow, NumPy, and scikit-image.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from skimage import color, feature

__all__ = [
    "STYLE_NAMES",
    "STYLE_FNS",
    "sketch",
    "pencil",
    "cartoon",
    "oil",
    "watercolor",
    "detect_style",
]

STYLE_NAMES: list[str] = ["sketch", "pencil", "cartoon", "oil", "watercolor", "natural"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
    """Ensure input is a PIL Image in RGB mode."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    return img.convert("RGB")


def _arr(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 numpy array in [0, 255]."""
    return np.array(img, dtype=np.float32)


def _from_arr(arr: np.ndarray) -> Image.Image:
    """Convert float32 numpy array to uint8 RGB PIL Image."""
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def sketch(img: Image.Image) -> Image.Image:
    """Pencil-sketch effect via dodge-blend (large blur radius=21)."""
    img = _to_pil(img)
    gray = img.convert("L")
    inv = ImageOps.invert(gray)
    blurred = inv.filter(ImageFilter.GaussianBlur(radius=21))

    gray_arr = np.array(gray, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)
    result = np.clip(gray_arr / (1.0 - blur_arr / 255.0 + 1e-6), 0, 255)

    gray_result = Image.fromarray(result.astype(np.uint8), mode="L")
    return gray_result.convert("RGB")


def pencil(img: Image.Image) -> Image.Image:
    """Grainy pencil effect — same dodge-blend but smaller blur (radius=10) + noise."""
    img = _to_pil(img)
    gray = img.convert("L")

    # Add subtle grain before blending
    rng = np.random.default_rng(seed=0)
    gray_arr = np.array(gray, dtype=np.float32)
    noise = rng.normal(0, 4, gray_arr.shape).astype(np.float32)
    gray_noisy = Image.fromarray(np.clip(gray_arr + noise, 0, 255).astype(np.uint8), mode="L")

    inv = ImageOps.invert(gray_noisy)
    blurred = inv.filter(ImageFilter.GaussianBlur(radius=10))

    noisy_arr = np.array(gray_noisy, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)
    result = np.clip(noisy_arr / (1.0 - blur_arr / 255.0 + 1e-6), 0, 255)

    gray_result = Image.fromarray(result.astype(np.uint8), mode="L")
    return gray_result.convert("RGB")


def cartoon(img: Image.Image) -> Image.Image:
    """Flat-colour regions with bold edges via quantization + edge overlay."""
    img = _to_pil(img)

    # Quantize to 8 colours (returns "P" mode palette image)
    quantized = img.quantize(colors=8).convert("RGB")

    # Compute edges on grayscale
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.uint8)

    # Binary mask: pixels with edge value > 20
    mask = edge_arr > 20

    # Overlay black where edges are detected
    q_arr = np.array(quantized, dtype=np.uint8)
    q_arr[mask] = 0

    return Image.fromarray(q_arr, mode="RGB")


def oil(img: Image.Image) -> Image.Image:
    """Oil-painting approximation via repeated ModeFilter."""
    img = _to_pil(img)
    # ModeFilter size must be odd; use 9 for compatibility
    mode_filter = ImageFilter.ModeFilter(size=9)
    result = img.filter(mode_filter).filter(mode_filter).filter(mode_filter)
    return result.convert("RGB")


def watercolor(img: Image.Image) -> Image.Image:
    """Soft watercolor look: blur + mild desaturation + second blur pass."""
    img = _to_pil(img)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
    desaturated = ImageEnhance.Color(blurred).enhance(0.75)
    soft = desaturated.filter(ImageFilter.GaussianBlur(radius=1))
    return soft.convert("RGB")


# ---------------------------------------------------------------------------
# Style detector
# ---------------------------------------------------------------------------


def detect_style(img: Image.Image) -> str:
    """Heuristic style detector based on edge density and colour saturation.

    Returns one of STYLE_NAMES.
    """
    img = _to_pil(img)
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = color.rgb2gray(arr).astype(np.float64)
    edges = feature.canny(gray, sigma=2.0)
    edge_density = float(edges.mean())

    hsv = color.rgb2hsv(arr)
    mean_sat = float(hsv[:, :, 1].mean())

    if mean_sat < 0.12:
        return "sketch" if edge_density >= 0.08 else "pencil"
    if edge_density >= 0.10:
        return "cartoon"
    if mean_sat < 0.35:
        return "watercolor"
    if mean_sat >= 0.35:
        return "oil"
    return "natural"


# ---------------------------------------------------------------------------
# Convenience mapping
# ---------------------------------------------------------------------------

STYLE_FNS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "sketch": sketch,
    "pencil": pencil,
    "cartoon": cartoon,
    "oil": oil,
    "watercolor": watercolor,
}
