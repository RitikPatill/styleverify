"""StyleVerify CLI — compare two face images for identity match."""

from __future__ import annotations

import argparse
import sys

from PIL import Image

from styleverify import embedder, styles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify whether two face images show the same person."
    )
    parser.add_argument("image1", help="Path to the reference image")
    parser.add_argument("image2", help="Path to the (possibly stylized) image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Cosine similarity threshold (default: 0.6)",
    )
    args = parser.parse_args()

    pil2 = Image.open(args.image2).convert("RGB")
    style = styles.detect_style(pil2)

    result = embedder.verify(args.image1, args.image2, args.threshold)
    score = round(result["score"], 2)

    if result["match"]:
        print(f"Same person (score={score}, style={style})")
        sys.exit(0)
    else:
        print(f"Different person (score={score}, style={style})")
        sys.exit(1)


if __name__ == "__main__":
    main()
