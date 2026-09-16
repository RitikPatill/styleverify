"""Demo script: generate a 2×3 grid showing all five style transforms.

Usage:
    python scripts/demo_styles.py [--input path/to/face.jpg] [--output demo_grid.png]

Run from the project root with:
    PYTHONPATH=src python scripts/demo_styles.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from styleverify.styles import STYLE_FNS  # noqa: E402


def _make_gradient_image(size: int = 256) -> Image.Image:
    """Synthesize a colorful gradient so the script is self-contained."""
    import numpy as np

    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    r = (xx * 255).astype("uint8")
    g = (yy * 255).astype("uint8")
    b = ((1 - xx * yy) * 200).astype("uint8")

    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _add_label(img: Image.Image, label: str, font_size: int = 18) -> Image.Image:
    """Add a text label banner at the bottom of a cell image."""
    banner_h = font_size + 8
    cell_w, cell_h = img.size
    result = Image.new("RGB", (cell_w, cell_h + banner_h), color=(30, 30, 30))
    result.paste(img, (0, 0))
    draw = ImageDraw.Draw(result)
    # `anchor` requires a TrueType font in Pillow 10+; centre manually instead.
    text_w = draw.textlength(label)
    draw.text(((cell_w - text_w) // 2, cell_h + 4), label, fill=(240, 240, 240))
    return result


def build_grid(source: Image.Image) -> Image.Image:
    """Return a 2-row × 3-column PIL grid (original + 5 styled variants)."""
    CELL_SIZE = 256
    FONT_SIZE = 16
    BANNER_H = FONT_SIZE + 8

    source = source.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)

    cells_data = [("original", source)]
    for name, fn in STYLE_FNS.items():
        styled = fn(source)
        cells_data.append((name, styled))

    # 2 rows × 3 cols
    cols, rows = 3, 2
    cell_h = CELL_SIZE + BANNER_H
    grid = Image.new("RGB", (cols * CELL_SIZE, rows * cell_h), color=(15, 15, 15))

    for idx, (label, img) in enumerate(cells_data):
        col = idx % cols
        row = idx // cols
        labelled = _add_label(img.resize((CELL_SIZE, CELL_SIZE)), label, FONT_SIZE)
        grid.paste(labelled, (col * CELL_SIZE, row * cell_h))

    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="StyleVerify M3 — style demo grid")
    parser.add_argument("--input", default=None, help="Path to source image")
    parser.add_argument("--output", default="demo_grid.png", help="Output path")
    args = parser.parse_args()

    if args.input:
        source = Image.open(args.input).convert("RGB")
    else:
        print("No --input provided; using synthetic gradient image.")
        source = _make_gradient_image()

    grid = build_grid(source)
    out_path = Path(args.output)
    grid.save(out_path)
    print(f"Saved demo grid → {out_path.resolve()}")


if __name__ == "__main__":
    main()
