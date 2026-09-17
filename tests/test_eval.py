"""Tests for pure-Python helpers in scripts/eval.py.

No network access, no DeepFace, no model download required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts/ to sys.path so we can import eval directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval import compute_metrics, parse_pairs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for synthetic pairs.txt
# ---------------------------------------------------------------------------

PAIRS_HEADER = "10\t300\n"

MATCHED_LINES = [
    "Aaron_Eckhart\t1\t2",
    "Zach_Braff\t3\t4",
]

MISMATCHED_LINES = [
    "Aaron_Eckhart\t1\tZach_Braff\t2",
    "Alice_Smith\t5\tBob_Jones\t6",
]


def _write_pairs(tmp_path: Path, matched: list[str], mismatched: list[str]) -> Path:
    content = PAIRS_HEADER + "\n".join(matched + mismatched) + "\n"
    p = tmp_path / "pairs.txt"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_pairs tests
# ---------------------------------------------------------------------------


def test_parse_pairs_matched_structure(tmp_path):
    """2 matched + 2 mismatched → correct labels and path format."""
    pairs_path = _write_pairs(tmp_path, MATCHED_LINES, MISMATCHED_LINES)
    pairs, labels = parse_pairs(pairs_path, n_per_class=10)

    assert len(pairs) == 4
    assert labels == [1, 1, 0, 0]

    # Matched pair 1: Aaron_Eckhart 1 vs 2
    assert pairs[0] == (
        "Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
        "Aaron_Eckhart/Aaron_Eckhart_0002.jpg",
    )
    # Mismatched pair 1: Aaron_Eckhart 1 vs Zach_Braff 2
    assert pairs[2] == (
        "Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
        "Zach_Braff/Zach_Braff_0002.jpg",
    )


def test_parse_pairs_path_zero_padding(tmp_path):
    """Image number is zero-padded to 4 digits."""
    pairs_path = _write_pairs(tmp_path, ["SomeName\t1\t10"], [])
    pairs, _ = parse_pairs(pairs_path, n_per_class=10)
    assert pairs[0] == (
        "SomeName/SomeName_0001.jpg",
        "SomeName/SomeName_0010.jpg",
    )


def test_parse_pairs_respects_n(tmp_path):
    """5 matched + 5 mismatched, n_per_class=3 → 6 total pairs."""
    matched = [f"Person{i}\t1\t2" for i in range(5)]
    mismatched = [f"PersonA{i}\t1\tPersonB{i}\t2" for i in range(5)]
    pairs_path = _write_pairs(tmp_path, matched, mismatched)

    pairs, labels = parse_pairs(pairs_path, n_per_class=3)
    assert len(pairs) == 6
    assert sum(labels) == 3        # 3 matched
    assert labels.count(0) == 3   # 3 mismatched


def test_parse_pairs_only_matched(tmp_path):
    """File with only matched lines returns correct labels."""
    pairs_path = _write_pairs(tmp_path, MATCHED_LINES, [])
    pairs, labels = parse_pairs(pairs_path, n_per_class=10)
    assert all(l == 1 for l in labels)
    assert len(pairs) == 2


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect():
    """Perfect classifier: accuracy=1.0, auc=1.0."""
    scores = [1.0, 1.0, 0.0, 0.0]
    labels = [1, 1, 0, 0]
    m = compute_metrics(scores, labels, threshold=0.6)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)


def test_compute_metrics_inverted():
    """Inverted classifier: accuracy=0.0, auc=0.0."""
    scores = [0.0, 0.0, 1.0, 1.0]
    labels = [1, 1, 0, 0]
    m = compute_metrics(scores, labels, threshold=0.6)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["auc"] == pytest.approx(0.0)


def test_compute_metrics_random():
    """Random scores → auc near 0.5 (allow ±0.15)."""
    rng = np.random.default_rng(seed=7)
    scores = rng.uniform(0, 1, 50).tolist()
    labels = ([1, 0] * 25)  # alternating
    m = compute_metrics(scores, labels, threshold=0.6)
    assert abs(m["auc"] - 0.5) <= 0.15


def test_compute_metrics_empty():
    """Empty inputs return zeros without crashing."""
    m = compute_metrics([], [], threshold=0.6)
    assert m["accuracy"] == 0.0
    assert m["auc"] == 0.0


def test_compute_metrics_threshold_boundary():
    """Score exactly at threshold counts as a match."""
    scores = [0.6, 0.6, 0.6, 0.6]
    labels = [1, 1, 1, 1]
    m = compute_metrics(scores, labels, threshold=0.6)
    assert m["accuracy"] == pytest.approx(1.0)
