# StyleVerify


> **Video walkthrough:** https://youtu.be/ZEK_jnDmXSs
> **60-second overview:** https://youtu.be/A9cibE3zidI

> Verify whether two face images show the same person even after artistic style transfer, using pre-trained embeddings.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/RitikPatill/styleverify/actions/workflows/ci.yml/badge.svg)

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

StyleVerify answers one question: are these two face images the same person? It is robust to artistic filters — sketch, cartoon, oil painting, watercolor, pencil — that break conventional pixel-level comparison. A reference photo and a stylized query are each passed through an ArcFace ONNX model to produce 512-dimensional embedding vectors; cosine similarity on those vectors is stable across the visual domain gap that confuses simpler approaches.

The project ships as both a REST API (`POST /verify`) and a CLI (`verify.py`). All inference runs on CPU via the weights bundled by DeepFace — no GPU, no cloud credits, no paid APIs.

## Quickstart

```bash
git clone https://github.com/RitikPatill/styleverify.git
cd styleverify
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start the API (first run downloads ArcFace weights ~70 MB to ~/.deepface/)
PYTHONPATH=src uvicorn main:app --reload
```

Or with Docker:

```bash
docker build -t styleverify .
docker run -p 8000:8000 styleverify
```

## Usage

**CLI**

```bash
PYTHONPATH=src python verify.py photo.jpg cartoon.jpg
# Same person (score=0.87, style=cartoon)

PYTHONPATH=src python verify.py photo.jpg other_person.jpg --threshold 0.7
# Different person (score=0.45, style=clean)
```

**REST API** — send two image files as multipart form data:

```bash
curl -X POST http://localhost:8000/verify \
  -F "image1=@photo.jpg" \
  -F "image2=@cartoon.jpg"
```

```json
{ "match": true, "score": 0.87, "style_detected": "cartoon", "elapsed_ms": 342.15 }
```

See [docs/api.md](docs/api.md) for the full endpoint reference.

## Architecture

```
image1 ──────────────────────────────┐
                                     ▼
                              ArcFace ONNX  ──►  cosine similarity  ──►  {match, score}
                                     ▲                   ▲
image2 ──► detect_style ──► style    │                   │
           (edge density,   label ───┘           threshold (default 0.6)
            saturation)
```

`detect_style` is a deterministic heuristic — no model, no neural style transfer — so the only learned component is the ArcFace backbone.

## Project structure

```
styleverify/
├── src/styleverify/        # core package
│   ├── embedder.py         # ArcFace extraction, cosine similarity, verify()
│   └── styles.py           # five style transforms + detect_style heuristic
├── scripts/
│   ├── demo_styles.py      # renders a 2x3 grid of all style variants
│   └── eval.py             # LFW benchmark: downloads data, runs all conditions
├── tests/
│   ├── fixtures/           # synthetic 64x64 face for smoke tests
│   ├── test_embedder.py    # same-image >= 0.99, different-image < 0.6
│   ├── test_styles.py      # shape/pixel/detector assertions
│   ├── test_api.py         # HTTP layer via TestClient, embedder monkeypatched
│   └── test_eval.py        # parse_pairs + compute_metrics, no network
├── docs/
│   ├── api.md              # endpoint reference
│   └── demo.tape           # VHS tape — regenerate GIF with `vhs docs/demo.tape`
├── main.py                 # FastAPI app — POST /verify, GET /health
├── verify.py               # CLI entry point
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## Evaluation

Evaluated on 600 pairs from LFW fold 1 (300 matched + 300 mismatched) at threshold 0.6. Style is applied to the query image only; the reference remains clean.

| Style          | Accuracy | AUC |
|----------------|----------|-----|
| Clean baseline | —        | —   |
| Sketch         | —        | —   |
| Pencil         | —        | —   |
| Cartoon        | —        | —   |
| Oil painting   | —        | —   |
| Watercolor     | —        | —   |

*Run the benchmark yourself to populate these numbers:*

```bash
# downloads lfw.tgz (~170 MB) and evaluates all conditions
PYTHONPATH=src python scripts/eval.py --data-dir data/lfw_eval

# smoke-test without network or model download (synthetic pairs)
PYTHONPATH=src python scripts/eval.py --dry-run
```

Results are written to `data/lfw_eval/results.json` (excluded from git).

## Roadmap

- [ ] Populate LFW benchmark table with real accuracy and AUC numbers
- [ ] Add a second embedding backend (e.g. FaceNet) for comparison
- [ ] Expose threshold as a query parameter on the `/verify` endpoint
- [ ] Add a batch endpoint that accepts a CSV of image-pair paths
- [ ] Publish a pre-built Docker image to GHCR via the CI workflow

## License

MIT — see [LICENSE](LICENSE).

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
