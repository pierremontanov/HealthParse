"""Language detection utilities for routing PDF documents.

This module centralizes lightweight language detection logic that can be
reused by preprocessing and classification pipelines.  It relies on the
`langdetect` package which provides fast inference suitable for the
initial triage of documents before OCR/NER processing.

Configuration is read from :mod:`src.config` at call-time so that the
module respects ``config.yaml``, environment variables, and CLI overrides.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set

import fitz  # PyMuPDF
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # ensure deterministic predictions across runs

logger = logging.getLogger(__name__)


def _get_supported_languages() -> Set[str]:
    """Return the configured set of supported language codes."""
    try:
        from src.config import settings
        return settings.supported_languages
    except Exception:
        return {"en", "es"}


def _normalise_language(code: str, supported: Set[str] | None = None) -> Optional[str]:
    """Normalise the raw language code returned by langdetect.

    Parameters
    ----------
    code : str
        Raw ISO-639-1 code (possibly with region suffix like ``en-us``).
    supported : set[str], optional
        Accepted language codes.  Defaults to the configured set.

    Returns
    -------
    str or None
        The normalised code if it matches a supported language, else ``None``.
    """
    if not code:
        return None

    if supported is None:
        supported = _get_supported_languages()

    code = code.lower()

    if code in supported:
        return code

    # Handle region suffixes: ``en-us`` → ``en``
    prefix = code.split("-")[0]
    if prefix in supported:
        return prefix

    return None


def detect_language(text: str, *, supported: Set[str] | None = None) -> str:
    """Detect the language of a text snippet.

    Parameters
    ----------
    text : str
        Raw text extracted from the document.
    supported : set[str], optional
        Override the configured set of supported languages.

    Returns
    -------
    str
        ``"en"``, ``"es"``, etc. when the language could be determined,
        otherwise ``"unknown"``.
    """
    cleaned = text.strip()
    if not cleaned:
        logger.debug("Language detection skipped: empty text.")
        return "unknown"

    try:
        raw_code = detect(cleaned)
    except LangDetectException as exc:
        logger.debug("langdetect failed: %s", exc)
        return "unknown"

    normalised = _normalise_language(raw_code, supported)
    if normalised is None:
        logger.debug(
            "Detected language '%s' is not in the supported set; returning 'unknown'.",
            raw_code,
        )
        return "unknown"

    logger.debug("Detected language: %s (raw=%s)", normalised, raw_code)
    return normalised


@dataclass
class PDFLanguageDetection:
    """Container for the results of a PDF language detection run."""

    language: str
    text_sample: str


def detect_pdf_language(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    min_characters: int | None = None,
) -> PDFLanguageDetection:
    """Detect the language of a PDF by sampling its text content.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF document to inspect.
    max_pages : int, optional
        Maximum number of pages to sample.  Falls back to config
        (``lang_detect_max_pages``, default 3).
    min_characters : int, optional
        Minimum accumulated characters before running detection.  Falls
        back to config (``lang_detect_min_chars``, default 120).

    Returns
    -------
    PDFLanguageDetection
        The predicted language and the sampled text.
    """
    # Resolve defaults from config
    if max_pages is None or min_characters is None:
        try:
            from src.config import settings
            if max_pages is None:
                max_pages = settings.lang_detect_max_pages
            if min_characters is None:
                min_characters = settings.lang_detect_min_chars
        except Exception:
            max_pages = max_pages or 3
            min_characters = min_characters or 120

    doc = fitz.open(pdf_path)
    try:
        sample_parts: list[str] = []
        for index, page in enumerate(doc):
            sample_parts.append(page.get_text("text").strip())
            if len(" ".join(sample_parts)) >= min_characters or index + 1 >= max_pages:
                break
    finally:
        doc.close()

    sample_text = " ".join(part for part in sample_parts if part)

    result = PDFLanguageDetection(
        language=detect_language(sample_text),
        text_sample=sample_text,
    )
    logger.debug(
        "PDF language detection for '%s': %s (sampled %d chars from %d page(s)).",
        pdf_path,
        result.language,
        len(sample_text),
        len(sample_parts),
    )
    return result


__all__ = ["PDFLanguageDetection", "detect_language", "detect_pdf_language"]
