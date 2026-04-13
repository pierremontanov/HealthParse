"""PDF type detection – text-based vs. scanned/image-based.

Determines whether a PDF contains selectable text or needs OCR by
sampling each page and checking whether extracted characters meet a
configurable threshold.

Configuration is read from :mod:`src.config` at call-time.
"""
from __future__ import annotations

import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def is_pdf_text_based(pdf_path: str, min_char_threshold: int | None = None) -> bool:
    """Check whether a PDF is text-based.

    Iterates over every page and returns ``True`` as soon as any page
    yields at least *min_char_threshold* characters of selectable text.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    min_char_threshold : int, optional
        Minimum stripped characters for a page to count as "text-based".
        Falls back to config (``min_char_threshold``, default 10).

    Returns
    -------
    bool
        ``True`` if the PDF has extractable text, ``False`` if it is
        likely scanned/image-based.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    RuntimeError
        If the PDF cannot be opened (e.g. corrupted file).
    """
    if min_char_threshold is None:
        try:
            from src.config import settings
            min_char_threshold = settings.min_char_threshold
        except Exception:
            min_char_threshold = 10

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        from src.pipeline.exceptions import PDFOpenError
        logger.error("Failed to open PDF '%s': %s", pdf_path, exc)
        raise PDFOpenError(pdf_path, str(exc)) from exc

    try:
        for page in doc:
            text = page.get_text()
            if len(text.strip()) >= min_char_threshold:
                logger.debug(
                    "PDF '%s' classified as text-based (page %d has %d chars).",
                    pdf_path,
                    page.number + 1,
                    len(text.strip()),
                )
                return True
    finally:
        doc.close()

    logger.debug(
        "PDF '%s' classified as scanned/image-based (no page reached %d chars).",
        pdf_path,
        min_char_threshold,
    )
    return False


__all__ = ["is_pdf_text_based"]
