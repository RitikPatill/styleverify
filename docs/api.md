# StyleVerify API

Base URL: `http://localhost:8000`

---

## Endpoints

### GET /health

Health check. Returns `200 OK` when the service is running.

**Response**

```json
{"status": "ok"}
```

**curl**

```bash
curl http://localhost:8000/health
```

---

### POST /verify

Compare two face images and return a similarity score.

**Request** — multipart/form-data

| Field    | Type | Required | Description                          |
|----------|------|----------|--------------------------------------|
| image1   | file | yes      | Reference image (natural photo)      |
| image2   | file | yes      | Query image (may be stylized)        |
| threshold| float| no       | Match threshold, default `0.6`       |

Pass `threshold` as a query parameter:

```
POST /verify?threshold=0.7
```

**Response**

```json
{
  "match": true,
  "score": 0.87,
  "style_detected": "cartoon",
  "elapsed_ms": 342.15
}
```

| Field          | Type   | Description                                                  |
|----------------|--------|--------------------------------------------------------------|
| match          | bool   | `true` if `score >= threshold`                               |
| score          | float  | Cosine similarity in [-1, 1]; higher = more similar          |
| style_detected | string | Detected style of `image2`: sketch, pencil, cartoon, oil, watercolor, or natural |
| elapsed_ms     | float  | End-to-end latency in milliseconds                           |

**curl examples**

```bash
# Basic comparison
curl -X POST http://localhost:8000/verify \
  -F "image1=@photo.jpg" \
  -F "image2=@cartoon.jpg"

# Custom threshold
curl -X POST "http://localhost:8000/verify?threshold=0.7" \
  -F "image1=@photo.jpg" \
  -F "image2=@sketch.jpg"
```

**Error responses**

| Status | Condition                          |
|--------|------------------------------------|
| 422    | Missing required field or unreadable image file |

---

## Running the server

### Directly

```bash
PYTHONPATH=src uvicorn main:app --reload
```

### Docker

```bash
docker build -t styleverify .
docker run -p 8000:8000 styleverify
```

Interactive API docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).
