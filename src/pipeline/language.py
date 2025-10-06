"""Language detection utilities for routing PDF documents.

This module centralizes lightweight language detection logic that can be
reused by preprocessing and classification pipelines. It relies on the
`langdetect` package which provides fast inference suitable for the
initial triage of documents before OCR/NER processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # ensure deterministic predictions across runs

SUPPORTED_LANGUAGES = {"en", "es"}


def _normalise_language(code: str) -> Optional[str]:
    """Normalise the raw language code returned by langdetect."""
    if not code:
        return None

    code = code.lower()

    # `langdetect` might return longer codes such as ``en-us``.
    if code in SUPPORTED_LANGUAGES:
        return code

    if code.startswith("en"):
        return "en"
    if code.startswith("es"):
        return "es"

    return None


def detect_language(text: str) -> str:
    """Detect the language of a text snippet.

    Parameters
    ----------
    text:
        Raw text extracted from the document.

    Returns
    -------
    str
        ``"en"`` or ``"es"`` when the language could be determined,
        otherwise ``"unknown"``.
    """
    cleaned = text.strip()
    if not cleaned:
        return "unknown"

    try:
        raw_code = detect(cleaned)
    except LangDetectException:
        return "unknown"

    normalised = _normalise_language(raw_code)
    return normalised or "unknown"


@dataclass
class PDFLanguageDetection:
    """Container for the results of a PDF language detection run."""

    language: str
    text_sample: str


def detect_pdf_language(
    pdf_path: str,
    *,
    max_pages: int = 3,
    min_characters: int = 120,
) -> PDFLanguageDetection:
    """Detect the language of a PDF by sampling its text content.

    Parameters
    ----------
    pdf_path:
        Path to the PDF document to inspect.
    max_pages:
        Maximum number of pages to sample before stopping.
    min_characters:
        Minimum number of accumulated characters required before running the
        language detector. This reduces noise for very short documents.

    Returns
    -------
    PDFLanguageDetection
        A dataclass containing the predicted language as well as the sampled
        text used for detection. The language defaults to ``"unknown"`` when
        not enough text is available.
    """
    doc = fitz.open(pdf_path)
    try:
        sample_parts = []
        for index, page in enumerate(doc):
            sample_parts.append(page.get_text("text").strip())
            if len(" ".join(sample_parts)) >= min_characters or index + 1 >= max_pages:
                break
    finally:
        doc.close()

    sample_text = " ".join(part for part in sample_parts if part)

    return PDFLanguageDetection(
        language=detect_language(sample_text),
        text_sample=sample_text,
    )


__all__ = ["PDFLanguageDetection", "detect_language", "detect_pdf_language"]
