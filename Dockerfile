# ══════════════════════════════════════════════════════════════════
# DocIQ – Multi-stage Dockerfile
# ══════════════════════════════════════════════════════════════════
# Produces a lean production image with Tesseract, Poppler, and
# the Python runtime.  Two entrypoints:
#   API  (default):  uvicorn src.api.app:app
#   CLI:             python -m src.main --input /data ...
# ══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps needed to compile wheels (numpy, opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Pierre Montanov <pierremontanov@gmail.com>"
LABEL org.opencontainers.image.title="DocIQ"
LABEL org.opencontainers.image.description="AI-powered medical document classification and extraction"

# System dependencies: Tesseract OCR + language packs, Poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-spa \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

# Application code
WORKDIR /app
COPY src/ src/
COPY config.yaml .

# Runtime directories
RUN mkdir -p /app/output /data

# ── Environment defaults ─────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCIQ_LOG_LEVEL=INFO \
    DOCIQ_LOG_FORMAT=json \
    DOCIQ_OUTPUT_DIR=/app/output \
    DOCIQ_API_HOST=0.0.0.0 \
    DOCIQ_API_PORT=8000 \
    DOCIQ_API_WORKERS=1

EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ───────────────────────────────────────────────────
# Default: run the API server.
# Override for CLI: docker run dociq python -m src.main --input /data
CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
