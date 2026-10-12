# GlucoTwin research prototype image. NOTE: not yet built/verified locally (no Docker on dev box).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

RUN apt-get update && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

# Raw data is never baked into the image; mount it at runtime: -v <host>/data:/app/data
CMD ["make", "all", "PY=python"]
