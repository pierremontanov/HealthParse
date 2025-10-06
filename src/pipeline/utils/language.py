"""Backwards compatible language utilities built on top of the central
language detection module."""
from src.pipeline.language import (
    PDFLanguageDetection,
    detect_language,
    detect_pdf_language,
)


def is_english(text: str) -> bool:
    return detect_language(text) == "en"


def is_spanish(text: str) -> bool:
    return detect_language(text) == "es"


__all__ = [
    "detect_language",
    "detect_pdf_language",
    "PDFLanguageDetection",
    "is_english",
    "is_spanish",
]
