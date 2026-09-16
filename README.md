# StyleVerify

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-scaffold-yellow)

**Are these two faces the same person — even when one is a cartoon?**

StyleVerify is a lightweight REST API + CLI tool for stylization-agnostic face verification. It combines ArcFace embeddings with deterministic style augmentations (sketch, cartoon, oil painting, watercolor, pencil) to produce a robust cosine similarity score that survives artistic filters.

---

## Status

**M1 complete — scaffold only.** The package skeleton, dependency pins, and project configuration are in place. Core logic (embedder, style augmentations, API, CLI) is not yet implemented; see the Roadmap below.

What M1 ships:
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

Files marked `(exists)` are present in M1. Everything else is a planned addition.

```
styleverify/
├── src/
│   └── styleverify/        # core package
│       ├── __init__.py     (exists — stub)
│       ├── embedder.py     (M2 — ArcFace embedding extraction)
│       └── styles.py       (M3 — style augmentations)
├── main.py                 (M4 — FastAPI app)
├── verify.py               (M4 — CLI entry point)
├── Dockerfile              (exists — stub)
├── requirements.txt        (exists — pinned)
├── pyproject.toml          (exists — minimal)
├── LICENSE                 (exists — MIT)
└── .gitignore              (exists)
```

---

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1 | Scaffold: project layout, deps, license, README | done |
| M2 | ArcFace embedder (`embedder.py`) | planned |
| M3 | Style augmentations (`styles.py`) | planned |
| M4 | FastAPI endpoint + CLI (`main.py`, `verify.py`) | planned |
| M5 | Evaluation script, LFW benchmark, filled metrics table | planned |

---

## License

[MIT](LICENSE) — Copyright 2024 Ritik
