"""M5 evaluation script: benchmark StyleVerify on LFW pairs.

Usage:
    python scripts/eval.py [--data-dir data/lfw_eval] [--threshold 0.6] [--dry-run]

--dry-run generates 20 synthetic pairs (no network, no model needed) for CI smoke-testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from styleverify import embedder, styles  # noqa: E402

PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
LFW_TGZ_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
STYLES = list(styles.STYLE_FNS.keys())  # ["sketch", "pencil", "cartoon", "oil", "watercolor"]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Stream-download url → dest with tqdm progress bar. Skip if dest exists."""
    import requests
    from tqdm import tqdm

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    desc = desc or dest.name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


# ---------------------------------------------------------------------------
# LFW pairs parser
# ---------------------------------------------------------------------------


def parse_pairs(
    pairs_path: Path, n_per_class: int = 300
) -> tuple[list[tuple[str, str]], list[int]]:
    """Parse LFW pairs.txt fold-1 format.

    Returns (pairs, labels) where:
      pairs  = list of (rel_path1, rel_path2)
      labels = list of int, 1 = same person, 0 = different person

    Takes first n_per_class matched lines then first n_per_class mismatched lines.
    Line format:
      header     : "10\\t300"
      matched    : "Name\\tN1\\tN2"        (3 fields)
      mismatched : "Name_A\\tN1\\tName_B\\tN2"  (4 fields)
    """
    lines = pairs_path.read_text(encoding="utf-8").splitlines()
    # Skip header line
    data_lines = [l for l in lines[1:] if l.strip()]

    matched: list[tuple[str, str]] = []
    mismatched: list[tuple[str, str]] = []

    for line in data_lines:
        fields = line.split("\t")
        if len(fields) == 3:
            name, n1, n2 = fields
            p1 = f"{name}/{name}_{int(n1):04d}.jpg"
            p2 = f"{name}/{name}_{int(n2):04d}.jpg"
            matched.append((p1, p2))
        elif len(fields) == 4:
            name_a, n1, name_b, n2 = fields
            p1 = f"{name_a}/{name_a}_{int(n1):04d}.jpg"
            p2 = f"{name_b}/{name_b}_{int(n2):04d}.jpg"
            mismatched.append((p1, p2))

    matched = matched[:n_per_class]
    mismatched = mismatched[:n_per_class]

    pairs = matched + mismatched
    labels = [1] * len(matched) + [0] * len(mismatched)
    return pairs, labels


# ---------------------------------------------------------------------------
# Tarfile selective extraction
# ---------------------------------------------------------------------------


def collect_needed(pairs: list[tuple[str, str]]) -> set[str]:
    """Return set of all unique relative paths referenced by pairs."""
    needed: set[str] = set()
    for p1, p2 in pairs:
        needed.add(p1)
        needed.add(p2)
    return needed


def extract_images(tgz_path: Path, needed: set[str], img_dir: Path) -> None:
    """Extract only the needed images from lfw.tgz.

    Members inside the archive are named 'lfw/<rel_path>'.
    They are extracted to img_dir/<rel_path>.
    """
    import tarfile

    img_dir.mkdir(parents=True, exist_ok=True)
    needed_members = {f"lfw/{p}" for p in needed}

    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name not in needed_members:
                continue
            # Strip leading "lfw/" prefix
            rel = member.name[len("lfw/"):]
            dest = img_dir / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            dest.write_bytes(src.read())


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    pairs: list[tuple[str, str]],
    labels: list[int],
    img_dir: Path,
    threshold: float = 0.6,
) -> dict[str, dict]:
    """Run verification for each condition (clean + each style).

    Style is applied to img2 only (simulates stylized query vs clean reference).
    Returns {"clean": {"scores": [...], "labels": [...]}, "sketch": {...}, ...}
    """
    from PIL import Image
    from tqdm import tqdm

    conditions = ["clean"] + STYLES
    results: dict[str, dict] = {c: {"scores": [], "labels": []} for c in conditions}

    for (p1, p2), label in tqdm(zip(pairs, labels), total=len(pairs), desc="pairs"):
        path1 = img_dir / p1
        path2 = img_dir / p2

        # Load images once
        try:
            img1 = np.array(Image.open(path1).convert("RGB"))
            img2_pil = Image.open(path2).convert("RGB")
            img2 = np.array(img2_pil)
        except Exception as exc:
            log.warning("Skipping pair (%s, %s): %s", p1, p2, exc)
            continue

        for condition in conditions:
            try:
                if condition == "clean":
                    arr2 = img2
                else:
                    style_fn = styles.STYLE_FNS[condition]
                    arr2 = np.array(style_fn(img2_pil))

                result = embedder.verify(img1, arr2, threshold=threshold)
                results[condition]["scores"].append(result["score"])
                results[condition]["labels"].append(label)
            except Exception as exc:
                log.warning("Skipping condition %s for pair (%s, %s): %s", condition, p1, p2, exc)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    scores: list[float], labels: list[int], threshold: float = 0.6
) -> dict:
    """Return {"accuracy": float, "auc": float}.

    accuracy = fraction of pairs where (score >= threshold) == label.
    auc      = sklearn.metrics.roc_auc_score(labels, scores).
    """
    from sklearn.metrics import roc_auc_score  # lazy import

    if not scores:
        return {"accuracy": 0.0, "auc": 0.0}

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    predictions = (scores_arr >= threshold).astype(int)
    accuracy = float((predictions == labels_arr).mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc = float(roc_auc_score(labels_arr, scores_arr))
        except ValueError:
            auc = 0.0

    return {"accuracy": accuracy, "auc": auc}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_results(results: dict, out_path: Path) -> None:
    """Dump results dict as JSON to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def print_table(results: dict[str, dict], threshold: float = 0.6) -> None:
    """Print a markdown table of accuracy and AUC per condition."""
    label_map = {
        "clean": "Clean baseline",
        "sketch": "Sketch",
        "pencil": "Pencil",
        "cartoon": "Cartoon",
        "oil": "Oil painting",
        "watercolor": "Watercolor",
    }

    header = "| Style          | Accuracy | AUC   |"
    sep    = "|----------------|----------|-------|"
    print(header)
    print(sep)

    for condition in ["clean"] + STYLES:
        if condition not in results:
            continue
        data = results[condition]
        m = compute_metrics(data["scores"], data["labels"], threshold)
        name = label_map.get(condition, condition.capitalize())
        print(f"| {name:<14} | {m['accuracy']:.4f}   | {m['auc']:.4f} |")


# ---------------------------------------------------------------------------
# Dry-run synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_pairs(
    n_matched: int = 10, n_mismatched: int = 10
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """Generate synthetic image pairs for dry-run mode.

    Matched: same random image used for both slots (score ≈ 1.0).
    Mismatched: two independent random images (score ≈ 0.0 for noise).
    """
    rng = np.random.default_rng(seed=42)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[int] = []

    for _ in range(n_matched):
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        pairs.append((img, img.copy()))
        labels.append(1)

    for _ in range(n_mismatched):
        img1 = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        pairs.append((img1, img2))
        labels.append(0)

    return pairs, labels


def _run_eval_dry(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    labels: list[int],
    threshold: float = 0.6,
) -> dict[str, dict]:
    """Like run_eval but accepts pre-loaded numpy arrays (for dry-run)."""
    from PIL import Image
    from tqdm import tqdm

    conditions = ["clean"] + STYLES
    results: dict[str, dict] = {c: {"scores": [], "labels": []} for c in conditions}

    for (arr1, arr2), label in tqdm(zip(pairs, labels), total=len(pairs), desc="dry-run pairs"):
        img2_pil = Image.fromarray(arr2)

        for condition in conditions:
            try:
                if condition == "clean":
                    a2 = arr2
                else:
                    style_fn = styles.STYLE_FNS[condition]
                    a2 = np.array(style_fn(img2_pil))

                result = embedder.verify(arr1, a2, threshold=threshold)
                results[condition]["scores"].append(result["score"])
                results[condition]["labels"].append(label)
            except Exception as exc:
                log.warning("Skipping condition %s: %s", condition, exc)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="StyleVerify LFW evaluation benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lfw_eval"),
                        help="Directory to store downloaded data and results")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Cosine similarity threshold for match decision")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data (no network or model download needed)")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    threshold: float = args.threshold

    if args.dry_run:
        print("Dry-run mode: generating synthetic pairs…")
        pairs_raw, labels = _make_synthetic_pairs()
        results = _run_eval_dry(pairs_raw, labels, threshold=threshold)
    else:
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download pairs.txt
        pairs_path = download_file(PAIRS_URL, data_dir / "pairs.txt", desc="pairs.txt")

        # 2. Parse pairs
        pairs, labels = parse_pairs(pairs_path, n_per_class=300)
        print(f"Loaded {len(pairs)} pairs ({sum(labels)} matched, {len(labels)-sum(labels)} mismatched)")

        # 3. Download lfw.tgz
        tgz_path = download_file(LFW_TGZ_URL, data_dir / "lfw.tgz", desc="lfw.tgz (~170 MB)")

        # 4. Extract only needed images
        img_dir = data_dir / "images"
        needed = collect_needed(pairs)
        print(f"Extracting {len(needed)} images from archive…")
        extract_images(tgz_path, needed, img_dir)

        # 5. Run evaluation
        results = run_eval(pairs, labels, img_dir, threshold=threshold)

    # 6. Save and display
    out_path = args.data_dir / "results.json"
    # Store metrics in results for JSON (exclude raw scores to keep file small)
    metrics_only: dict[str, dict] = {}
    for condition, data in results.items():
        m = compute_metrics(data["scores"], data["labels"], threshold)
        metrics_only[condition] = {**m, "n_pairs": len(data["scores"])}
    save_results(metrics_only, out_path)
    print(f"\nResults saved to {out_path}\n")

    print_table(results, threshold)


if __name__ == "__main__":
    main()
