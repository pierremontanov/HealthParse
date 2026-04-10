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
from src.pipeline.core_engine import DocIQEngine

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

# Singleton engine — created once, reused across requests.
_engine: DocIQEngine | None = None


def _get_engine() -> DocIQEngine:
    global _engine
    if _engine is None:
        _engine = DocIQEngine(run_inference=settings.run_inference)
        logger.info("DocIQEngine initialised for API.")
    return _engine


# ── Health & readiness ────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe",
)
async def health():
    """Returns 200 if the service is alive."""
    return HealthResponse(status="ok", version=settings.version)


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe",
)
async def ready():
    """Check that runtime dependencies (Tesseract, models) are available."""
    checks: List[ReadinessCheck] = []

    # Tesseract
    tesseract_available = shutil.which("tesseract") is not None
    checks.append(
        ReadinessCheck(
            name="tesseract",
            available=tesseract_available,
            detail=shutil.which("tesseract") if tesseract_available else "not found on PATH",
        )
    )

    # Inference engine
    try:
        engine = _get_engine()
        types = engine._engine.registered_types if engine._engine else []
        checks.append(
            ReadinessCheck(
                name="inference_engine",
                available=len(types) > 0,
                detail=f"registered types: {', '.join(types)}" if types else "no models loaded",
            )
        )
    except Exception as exc:
        checks.append(
            ReadinessCheck(name="inference_engine", available=False, detail=str(exc))
        )

    all_ready = all(c.available for c in checks)
    return ReadinessResponse(ready=all_ready, checks=checks)


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
                    from src.pipeline.validation.schemas import ResultSchema
                    from src.pipeline.validation.prescription_schema import Prescription
                    from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema

                    _SCHEMA_MAP = {
                        "result": ResultSchema,
                        "prescription": Prescription,
                        "clinical_history": ClinicalHistorySchema,
                    }
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

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )
