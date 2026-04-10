"""Request and response models for the DocIQ REST API.

These Pydantic models define the API contract and are rendered
automatically in the OpenAPI / Swagger docs at ``/docs``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health / Readiness ────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(..., description="Service status.", examples=["ok"])
    version: str = Field(..., description="DocIQ version string.", examples=["1.0.0"])


class ReadinessCheck(BaseModel):
    """Individual readiness dependency check."""
    name: str
    available: bool
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Response for GET /ready."""
    ready: bool = Field(..., description="True when all dependencies are available.")
    checks: List[ReadinessCheck]


# ── Processing ────────────────────────────────────────────────────

class ProcessingResult(BaseModel):
    """Single-document processing result."""
    file: str = Field(..., description="Original filename.")
    status: str = Field(..., description="ok | extraction_error | inference_error")
    language: str = Field("unknown", description="Detected language.")
    method: str = Field("", description="Extraction method: direct | ocr | image.")
    document_type: Optional[str] = Field(None, description="Classified document type.")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Structured NER output.")
    validated: bool = Field(False, description="Whether the data passed schema validation.")
    error: Optional[str] = Field(None, description="Error message if processing failed.")
    elapsed_ms: int = Field(0, description="Processing time in milliseconds.")


class ProcessingResponse(BaseModel):
    """Response for POST /process."""
    results: List[ProcessingResult]
    summary: Dict[str, int] = Field(
        ..., description="Count of results by status (ok, extraction_error, etc.)."
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str = Field(..., description="Error type.")
    detail: str = Field(..., description="Human-readable description.")
