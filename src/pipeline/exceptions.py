"""Custom exception hierarchy for DocIQ.

All pipeline exceptions inherit from :class:`DocIQError` so callers can
catch the base class for blanket error handling or individual subclasses
for targeted recovery.
"""

from __future__ import annotations


class DocIQError(Exception):
    """Base exception for all DocIQ pipeline errors."""


class DocumentExtractionError(DocIQError):
    """Raised when text extraction (direct or OCR) fails for a document."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Extraction failed for '{filename}': {reason}")


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


class ValidationError(DocIQError):
    """Raised when Pydantic schema validation fails on extracted data."""

    def __init__(self, filename: str, document_type: str, reason: str) -> None:
        self.filename = filename
        self.document_type = document_type
        self.reason = reason
        super().__init__(
            f"Validation failed for '{filename}' (type={document_type}): {reason}"
        )


class OCRError(DocIQError):
    """Raised when Tesseract OCR processing fails."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"OCR failed for '{filename}': {reason}")


class UnsupportedFileError(DocIQError):
    """Raised when a file has an unsupported extension."""

    def __init__(self, filename: str, extension: str) -> None:
        self.filename = filename
        self.extension = extension
        super().__init__(
            f"Unsupported file type '{extension}' for '{filename}'"
        )
