"""DocIQ FastAPI application.

Launch with::

    uvicorn src.api.app:app --reload

OpenAPI docs available at ``/docs`` (Swagger) and ``/redoc``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.api.models import (
    ErrorResponse,
    HealthResponse,
    ProcessingResponse,
    ProcessingResult,
    ReadinessCheck,
    ReadinessResponse,
)
from src.config import settings
from src.logging_config import setup_logging
from src.pipeline.core_engine import DocIQEngine
from src.pipeline.exceptions import DocIQError

# Configure structured logging before anything else logs.
setup_logging(level=settings.log_level, fmt=settings.log_format)

logger = logging.getLogger(__name__)

# ── Application factory ──────────────────────────────────────────

app = FastAPI(
    title="DocIQ API",
    description=(
        "AI-powered medical document classification and entity extraction. "
        "Upload PDF or image files and receive structured, validated JSON "
        "with optional FHIR resource mapping."
    ),
    version=settings.version,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# ── Process-level bookkeeping ────────────────────────────────────
_STARTUP_TIME: float = time.monotonic()
_STARTUP_UTC: datetime = datetime.now(timezone.utc)

# Singleton engine — created once, reused across requests.
_engine: DocIQEngine | None = None


def _get_engine() -> DocIQEngine:
    global _engine
    if _engine is None:
        _engine = DocIQEngine(run_inference=settings.run_inference)
        logger.info("DocIQEngine initialised for API.")
    return _engine


# ── Health & readiness helpers ───────────────────────────────────

def _timed_check(name: str, fn) -> ReadinessCheck:
    """Run *fn* and return a ReadinessCheck with elapsed time."""
    t0 = time.monotonic()
    try:
        available, detail = fn()
    except Exception as exc:
        available, detail = False, str(exc)
    elapsed = (time.monotonic() - t0) * 1000
    return ReadinessCheck(name=name, available=available, detail=detail, elapsed_ms=round(elapsed, 2))


def _check_tesseract():
    path = settings.tesseract_cmd or shutil.which("tesseract")
    if path and os.path.isfile(path):
        return True, path
    if shutil.which("tesseract"):
        return True, shutil.which("tesseract")
    return False, "not found on PATH"


def _check_poppler():
    """Verify that pdftoppm (Poppler) is reachable."""
    exe = "pdftoppm"
    if settings.poppler_path:
        candidate = os.path.join(settings.poppler_path, exe)
        if os.path.isfile(candidate):
            return True, candidate
    found = shutil.which(exe)
    if found:
        return True, found
    return False, "pdftoppm not found on PATH"


def _check_inference():
    engine = _get_engine()
    types = engine._engine.registered_types if engine._engine else []
    if types:
        return True, f"registered types: {', '.join(types)}"
    return False, "no models loaded"


def _check_config():
    """Validate that the current settings object loaded successfully."""
    try:
        # Accessing a few key fields forces any lazy validation errors.
        _ = settings.ocr_dpi, settings.export_format, settings.version
        return True, "settings loaded"
    except Exception as exc:
        return False, str(exc)


def _check_disk():
    """Ensure the output directory is writable and has >100 MB free."""
    out_dir = settings.output_dir or "output"
    try:
        os.makedirs(out_dir, exist_ok=True)
        usage = shutil.disk_usage(out_dir)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 100:
            return False, f"{free_mb:.0f} MB free (< 100 MB minimum)"
        return True, f"{free_mb:.0f} MB free"
    except Exception as exc:
        return False, str(exc)


# ── Health & readiness endpoints ─────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe",
)
async def health():
    """Returns 200 if the service is alive.

    Includes uptime and a UTC timestamp so monitoring dashboards can
    detect stale instances.
    """
    uptime = time.monotonic() - _STARTUP_TIME
    now = datetime.now(timezone.utc).isoformat()
    return HealthResponse(
        status="ok",
        version=settings.version,
        uptime_seconds=round(uptime, 2),
        timestamp=now,
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe",
    responses={503: {"model": ReadinessResponse, "description": "Not ready"}},
)
async def ready():
    """Check that runtime dependencies are available.

    Returns **200** when all checks pass or **503** when at least one
    dependency is unavailable — compatible with Kubernetes readiness
    probes.
    """
    t0 = time.monotonic()

    checks: List[ReadinessCheck] = [
        _timed_check("tesseract", _check_tesseract),
        _timed_check("poppler", _check_poppler),
        _timed_check("inference_engine", _check_inference),
        _timed_check("config", _check_config),
        _timed_check("disk", _check_disk),
    ]

    total_ms = round((time.monotonic() - t0) * 1000, 2)
    all_ready = all(c.available for c in checks)

    payload = ReadinessResponse(
        ready=all_ready,
        checks=checks,
        total_elapsed_ms=total_ms,
    )

    if all_ready:
        return payload
    return JSONResponse(status_code=503, content=payload.model_dump())


# ── Document processing ──────────────────────────────────────────

@app.post(
    "/process",
    response_model=ProcessingResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Processing"],
    summary="Process uploaded documents",
)
async def process_documents(
    files: List[UploadFile] = File(..., description="PDF or image files to process."),
    format: str = Query(
        "json",
        description="Response payload format. 'json' returns extracted data inline; "
                    "'fhir' returns FHIR-mapped resources in extracted_data.",
        enum=["json", "fhir"],
    ),
):
    """Upload one or more medical documents for classification and extraction.

    Supported file types: ``.pdf``, ``.png``, ``.jpg``, ``.jpeg``.

    Each file goes through:
    1. **Text extraction** (direct for text-based PDFs, OCR for scanned/images)
    2. **Language detection** (English / Spanish)
    3. **Document classification** (prescription, lab result, clinical history)
    4. **NER extraction** (rule-based entity extraction)
    5. **Schema validation** (Pydantic)
    6. Optionally **FHIR mapping** when ``format=fhir``
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    engine = _get_engine()
    results: List[dict] = []

    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[1].lower()
        if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
            results.append(
                {
                    "file": upload.filename or "unknown",
                    "status": "unsupported_type",
                    "language": "unknown",
                    "method": "",
                    "document_type": None,
                    "extracted_data": None,
                    "validated": False,
                    "error": f"Unsupported file type: {suffix}",
                    "elapsed_ms": 0,
                }
            )
            continue

        # Write upload to a temp file so the extraction pipeline can read it.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            content = await upload.read()
            tmp.write(content)
            tmp.flush()
            tmp.close()

            result = engine.process_file(tmp.name)
            result["file"] = upload.filename or result["file"]

            # FHIR mapping pass
            if format == "fhir" and result.get("extracted_data") and result.get("document_type"):
                try:
                    from src.pipeline.fhir_mapper import map_to_fhir_loose
                    from src.pipeline.validation import (
                        ClinicalHistorySchema,
                        Prescription,
                        ResultSchema,
                        SCHEMA_REGISTRY,
                    )

                    _SCHEMA_MAP = SCHEMA_REGISTRY
                    doc_type = result["document_type"]
                    if doc_type in _SCHEMA_MAP:
                        model = _SCHEMA_MAP[doc_type](**result["extracted_data"])
                        result["extracted_data"] = map_to_fhir_loose(model)
                except Exception as exc:
                    logger.warning("FHIR mapping failed for %s: %s", upload.filename, exc)

            results.append(result)

        except Exception as exc:
            logger.error("Processing failed for %s: %s", upload.filename, exc)
            results.append(
                {
                    "file": upload.filename or "unknown",
                    "status": "extraction_error",
                    "language": "unknown",
                    "method": "",
                    "document_type": None,
                    "extracted_data": None,
                    "validated": False,
                    "error": str(exc),
                    "elapsed_ms": 0,
                }
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Build summary
    summary: dict = {}
    for r in results:
        summary[r["status"]] = summary.get(r["status"], 0) + 1

    return ProcessingResponse(
        results=[ProcessingResult(**r) for r in results],
        summary=summary,
    )


# ── Error handlers ────────────────────────────────────────────────

_DOCIQ_STATUS_MAP = {
    "UnsupportedFileError": 400,
    "ConfigurationError": 500,
    "ConfigFileNotFoundError": 500,
    "ConfigParseError": 500,
    "DocumentExtractionError": 422,
    "PDFOpenError": 422,
    "PDFExtractionError": 422,
    "PageTimeoutError": 504,
    "OCRError": 422,
    "ImageLoadError": 422,
    "TesseractError": 422,
    "ClassificationError": 422,
    "NERExtractionError": 422,
    "ModelError": 500,
    "ModelLoadError": 500,
    "ModelExecutionError": 500,
    "SchemaValidationError": 422,
    "ExportError": 500,
    "FHIRMappingError": 422,
}


@app.exception_handler(DocIQError)
async def dociq_exception_handler(request, exc: DocIQError):
    """Map DocIQ custom exceptions to appropriate HTTP status codes."""
    exc_name = type(exc).__name__
    status_code = _DOCIQ_STATUS_MAP.get(exc_name, 500)
    logger.error("DocIQ error (%s): %s", exc_name, exc)
    return JSONResponse(
        status_code=status_code,
        content={"error": exc_name, "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )
