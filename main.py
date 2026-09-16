"""StyleVerify FastAPI application."""

from __future__ import annotations

import io
import time

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from styleverify import embedder, styles

app = FastAPI(title="StyleVerify", version="0.1.0")


class VerifyResponse(BaseModel):
    match: bool
    score: float
    style_detected: str
    elapsed_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(
    image1: UploadFile,
    image2: UploadFile,
    threshold: float = 0.6,
):
    t0 = time.perf_counter()

    try:
        contents1 = await image1.read()
        pil1 = Image.open(io.BytesIO(contents1)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot decode image1")

    try:
        contents2 = await image2.read()
        pil2 = Image.open(io.BytesIO(contents2)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot decode image2")

    arr1 = np.array(pil1, dtype=np.uint8)
    arr2 = np.array(pil2, dtype=np.uint8)

    result = embedder.verify(arr1, arr2, threshold)
    style_detected = styles.detect_style(pil2)

    t1 = time.perf_counter()
    elapsed_ms = round((t1 - t0) * 1000, 2)

    return VerifyResponse(
        match=result["match"],
        score=result["score"],
        style_detected=style_detected,
        elapsed_ms=elapsed_ms,
    )
