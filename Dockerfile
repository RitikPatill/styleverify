FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

# DeepFace downloads ArcFace weights (~70 MB) to ~/.deepface/ on first inference call.
# Weights are NOT baked into the image to keep the image lean.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
