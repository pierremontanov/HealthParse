"""Custom exception hierarchy for DocIQ.

All pipeline exceptions inherit from :class:`DocIQError` so callers can
catch the base class for blanket error handling or individual subclasses
for targeted recovery.

Hierarchy
---------
::

    DocIQError
    ├── ConfigurationError
    │   ├── ConfigFileNotFoundError
    │   └── ConfigParseError
    ├── DocumentExtractionError
    │   ├── PDFOpenError
    │   ├── PDFExtractionError
    │   └── PageTimeoutError
    ├── OCRError
    │   ├── ImageLoadError
    │   └── TesseractError
    ├── ClassificationError
    ├── NERExtractionError
    ├── ModelError
    │   ├── ModelLoadError
    │   └── ModelExecutionError
    ├── SchemaValidationError
    ├── ExportError
    │   └── FHIRMappingError
    └── UnsupportedFileError
"""

from __future__ import annotations

from typing import Optional


# ── Base ────────────────────────────────────────────────────────

class DocIQError(Exception):
    """Base exception for all DocIQ pipeline errors."""


# ── Configuration ───────────────────────────────────────────────

class ConfigurationError(DocIQError):
    """Raised when configuration loading or validation fails."""

    def __init__(self, reason: str, path: Optional[str] = None) -> None:
        self.reason = reason
        self.path = path
        msg = f"Configuration error: {reason}"
        if path:
            msg += f" (file: {path})"
        super().__init__(msg)


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a config file does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Config file not found: {path}", path=path)


class ConfigParseError(ConfigurationError):
    """Raised when a config file cannot be parsed."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to parse config: {reason}", path=path)


# ── Document extraction ─────────────────────────────────────────

class DocumentExtractionError(DocIQError):
    """Raised when text extraction (direct or OCR) fails for a document."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Extraction failed for '{filename}': {reason}")


class PDFOpenError(DocumentExtractionError):
    """Raised when a PDF file cannot be opened (corrupted, encrypted, etc.)."""

    def __init__(self, filename: str, reason: str = "cannot open PDF") -> None:
        super().__init__(filename, reason)


class PDFExtractionError(DocumentExtractionError):
    """Raised when text extraction from a specific PDF page fails."""

    def __init__(self, filename: str, page: int, reason: str) -> None:
        self.page = page
        super().__init__(filename, f"page {page}: {reason}")


class PageTimeoutError(DocumentExtractionError):
    """Raised when a page extraction exceeds the configured timeout."""

    def __init__(self, filename: str, page: int, timeout_seconds: int) -> None:
        self.page = page
        self.timeout_seconds = timeout_seconds
        super().__init__(
            filename, f"page {page} timed out after {timeout_seconds}s"
        )


# ── OCR ─────────────────────────────────────────────────────────

class OCRError(DocIQError):
    """Raised when Tesseract OCR processing fails."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"OCR failed for '{filename}': {reason}")


class ImageLoadError(OCRError):
    """Raised when an image file cannot be loaded."""

    def __init__(self, filename: str) -> None:
        super().__init__(filename, "could not load image")


class TesseractError(OCRError):
    """Raised when Tesseract execution fails."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(filename, f"Tesseract error: {reason}")


# ── Classification / NER ────────────────────────────────────────

class ClassificationError(DocIQError):
    """Raised when document type classification fails or returns no match."""

    def __init__(self, filename: str, reason: str = "unrecognised document type") -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Classification failed for '{filename}': {reason}")


class NERExtractionError(DocIQError):
    """Raised when named-entity extraction fails for a classified document."""

    def __init__(self, filename: str, document_type: str, reason: str) -> None:
        self.filename = filename
        self.document_type = document_type
        self.reason = reason
        super().__init__(
            f"NER extraction failed for '{filename}' (type={document_type}): {reason}"
        )


# ── Model errors ────────────────────────────────────────────────

class ModelError(DocIQError):
    """Base class for ML model errors."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Model '{model_name}': {reason}")


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded."""

    def __init__(self, model_name: str, reason: str = "failed to load") -> None:
        super().__init__(model_name, f"load failed: {reason}")


class ModelExecutionError(ModelError):
    """Raised when a model fails during inference."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(model_name, f"execution failed: {reason}")


# ── Validation ──────────────────────────────────────────────────

class SchemaValidationError(DocIQError):
    """Raised when Pydantic schema validation fails on extracted data.

    Wraps the underlying Pydantic ``ValidationError`` with DocIQ context
    (filename, document type) while keeping the original error accessible.
    """

    def __init__(
        self,
        filename: str,
        document_type: str,
        reason: str,
        *,
        pydantic_error: Optional[Exception] = None,
    ) -> None:
        self.filename = filename
        self.document_type = document_type
        self.reason = reason
        self.pydantic_error = pydantic_error
        super().__init__(
            f"Validation failed for '{filename}' (type={document_type}): {reason}"
        )


# Keep the old name as an alias for backward compatibility.
ValidationError = SchemaValidationError


# ── Export ──────────────────────────────────────────────────────

class ExportError(DocIQError):
    """Raised when output export fails."""

    def __init__(self, fmt: str, reason: str) -> None:
        self.fmt = fmt
        self.reason = reason
        super().__init__(f"Export ({fmt}) failed: {reason}")


class FHIRMappingError(ExportError):
    """Raised when FHIR resource mapping fails."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        super().__init__("fhir", f"'{filename}': {reason}")


# ── File handling ───────────────────────────────────────────────

class UnsupportedFileError(DocIQError):
    """Raised when a file has an unsupported extension."""

    def __init__(self, filename: str, extension: str) -> None:
        self.filename = filename
        self.extension = extension
        super().__init__(
            f"Unsupported file type '{extension}' for '{filename}'"
        )
