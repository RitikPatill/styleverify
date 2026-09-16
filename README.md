# StyleVerify

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-m3%20styles-blue)

**Are these two faces the same person — even when one is a cartoon?**

StyleVerify is a lightweight REST API + CLI tool for stylization-agnostic face verification. It combines ArcFace embeddings with deterministic style augmentations (sketch, cartoon, oil painting, watercolor, pencil) to produce a robust cosine similarity score that survives artistic filters.

---

## Status

**M3 complete — style augmentation pipeline.** Five deterministic image transforms and a style detector are implemented and tested. FastAPI endpoint and CLI are planned in M4.

What M3 ships:
- `src/styleverify/styles.py` — five transforms (`sketch`, `pencil`, `cartoon`, `oil`, `watercolor`) plus `detect_style` heuristic; all pure Pillow + NumPy + scikit-image, no OpenCV
- `scripts/demo_styles.py` — standalone script that produces a 2×3 image grid (original + 5 styled variants); run with `PYTHONPATH=src python scripts/demo_styles.py`
- `tests/test_styles.py` — transform output shape/mode checks, pixel-change assertions, and detector validity tests

What M2 shipped:
- `src/styleverify/embedder.py` — `get_embedding`, `cosine_similarity`, `verify` functions backed by ArcFace via DeepFace (CPU-only ONNX inference)
- `tests/test_embedder.py` — two unit tests: same-image similarity ≥ 0.99, different-image similarity < 0.6
- `pytest==8.2.2` added to `requirements.txt`

What M1 shipped:
- `src/` layout with `styleverify` package stub
- `requirements.txt` with fully pinned dependencies (DeepFace 0.0.93, NumPy 1.26.4, Pillow 10.3.0, scikit-image 0.23.2, FastAPI 0.111.0, uvicorn 0.30.1)
- `pyproject.toml` — setuptools build, `requires-python = ">=3.10"`
- MIT `LICENSE`, `.gitignore`, this README

---

## Pipeline

```
Input image A          Input image B
     │                      │
     ▼                      ▼
┌─────────────┐       ┌─────────────┐
│ Style detect│       │ Style detect│
│ + normalize │       │ + normalize │
└──────┬──────┘       └──────┬──────┘
       │                     │
       ▼                     ▼
┌─────────────────────────────────┐
│    ArcFace embedding (ONNX)     │
│         via DeepFace            │
└──────────────┬──────────────────┘
               │
               ▼
        cosine similarity
               │
               ▼
   { match, score, style_detected }
```

---

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** On first inference, DeepFace downloads ArcFace ONNX weights (~70 MB) to `~/.deepface/weights/`. This makes the first request slow; subsequent requests are fast.

### Style demo grid

```bash
# Uses a synthetic gradient image if no --input is given
PYTHONPATH=src python scripts/demo_styles.py --output demo_grid.png

# With your own face photo
PYTHONPATH=src python scripts/demo_styles.py --input photo.jpg --output demo_grid.png
```

### REST API

> **Not yet available — ships in M4.**

```bash
uvicorn main:app --reload
```

```bash
curl -X POST http://localhost:8000/verify \
  -F "image1=@photo.jpg" \
  -F "image2=@cartoon.jpg"
```

```json
{ "match": true, "score": 0.87, "style_detected": "cartoon" }
```

### CLI

> **Not yet available — ships in M4.**

```bash
python verify.py photo.jpg cartoon.jpg
# Same person (score=0.87, style=cartoon)
```

### Docker

> **Not yet available — ships in M4.**

```bash
docker build -t styleverify .
docker run -p 8000:8000 styleverify
```

---

## Evaluation

Evaluated on a 500-pair subset of LFW (Labeled Faces in the Wild):

| Style augmentation | Accuracy | TAR@FAR=0.01 |
|---|---|---|
| None (baseline) | — | — |
| Sketch | — | — |
| Cartoon | — | — |
| Oil painting | — | — |
| Watercolor | — | — |
| Pencil | — | — |

*Numbers will be filled in after M5 evaluation script runs.*

---

## Project structure

Files without a milestone tag are present on disk. Planned additions show the milestone they ship in.

```
styleverify/
├── src/
│   └── styleverify/        # core package
│       ├── __init__.py     (re-exports embedder + styles)
│       ├── embedder.py     (ArcFace embedding extraction, cosine similarity, verify)
│       └── styles.py       (five style transforms + detect_style heuristic)
├── scripts/
│   └── demo_styles.py      (2×3 grid demo; run with PYTHONPATH=src)
├── tests/
│   ├── __init__.py
│   ├── test_embedder.py    (same-image ≥ 0.99, different-image < 0.6)
│   └── test_styles.py      (transform shape/pixel/detector tests)
├── main.py                 (M4 — FastAPI app)
├── verify.py               (M4 — CLI entry point)
├── Dockerfile              (stub)
├── requirements.txt        (pinned, includes pytest)
├── pyproject.toml          (minimal setuptools build)
├── LICENSE                 (MIT)
└── .gitignore
```

---

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1 | Scaffold: project layout, deps, license, README | done |
| M2 | ArcFace embedder (`embedder.py`) | done |
| M3 | Style augmentations (`styles.py`) | done |
| M4 | FastAPI endpoint + CLI (`main.py`, `verify.py`) | planned |
| M5 | Evaluation script, LFW benchmark, filled metrics table | planned |

---

## License

[MIT](LICENSE) — Copyright 2024 Ritik
