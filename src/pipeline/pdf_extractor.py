"""PDF text extraction with per-page threading.

Provides two extraction strategies:

- **Direct extraction** for text-based PDFs (via PyMuPDF)
- **OCR extraction** for scanned/image PDFs (via pdf2image + Tesseract)

Both strategies parallelise work across pages using a
:class:`~concurrent.futures.ThreadPoolExecutor`.  Configuration values
(DPI, thread-pool size) are read from :mod:`src.config` at call-time.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pdf2image import convert_from_path

from src.pipeline.pdf_type_detector import is_pdf_text_based
from src.pipeline.preprocess import preprocess_image

logger = logging.getLogger(__name__)

PageClassifier = Optional[Callable[[str, int], Any]]


def _get_page_timeout() -> int:
    """Return the configured page-level timeout in seconds (default 300)."""
    try:
        from src.config import settings
        return settings.page_timeout
    except Exception:
        return 300


# ── Helpers ──────────────────────────────────────────────────────

def _get_page_workers(n_pages: int) -> int:
    """Calculate a sensible thread-pool size for *n_pages*.

    Falls back to ``settings.max_workers`` when configured, otherwise
    uses ``min(n_pages, cpu_count * 2)``.
    """
    try:
        from src.config import settings
        if settings.max_workers is not None:
            return min(n_pages, settings.max_workers)
    except Exception:
        pass
    return min(n_pages, (os.cpu_count() or 1) * 2)


def _get_ocr_dpi() -> int:
    """Return the configured OCR DPI (default 300)."""
    try:
        from src.config import settings
        return settings.ocr_dpi
    except Exception:
        return 300


def _run_classifier(text: str, page_index: int, classifier: PageClassifier) -> Any:
    if classifier is None:
        return None
    return classifier(text, page_index)


def _sorted_page_results(
    results: Iterable[Tuple[int, str, Any]],
) -> List[Tuple[int, str, Any]]:
    return sorted(results, key=lambda item: item[0])


def _assemble_text(ordered: List[Tuple[int, str, Any]]) -> str:
    """Combine per-page text into a single string."""
    return "".join(
        f"\n--- Page {index + 1} ---\n{text}" for index, text, _ in ordered
    )


def _build_page_results(
    ordered: List[Tuple[int, str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {"page": index + 1, "text": text, "classification": classification}
        for index, text, classification in ordered
    ]


# ── Per-page workers ─────────────────────────────────────────────

def _process_ocr_page(
    page_index: int,
    page_image: Any,
    classifier: PageClassifier,
) -> Tuple[int, str, Any]:
    """OCR a single page image and optionally classify it."""
    img = page_image.convert("RGB")
    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess_image(open_cv_image)
    text = pytesseract.image_to_string(preprocessed)
    classification = _run_classifier(text, page_index, classifier)
    return page_index, text, classification


def _process_direct_page(
    doc: fitz.Document,
    page_index: int,
    classifier: PageClassifier,
) -> Tuple[int, str, Any]:
    """Extract text from a single page of an already-opened PDF."""
    page = doc.load_page(page_index)
    text = page.get_text()
    classification = _run_classifier(text, page_index, classifier)
    return page_index, text, classification


# ── OCR extraction ───────────────────────────────────────────────

def extract_text_from_pdf_ocr(
    pdf_path: str,
    *,
    page_classifier: PageClassifier = None,
    return_page_results: bool = False,
):
    """Convert each page of a scanned PDF to an image, then OCR.

    Pages are processed in parallel via a thread pool.  Per-page
    failures are isolated — a failing page contributes empty text
    rather than crashing the entire document.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_classifier : callable, optional
        Receives ``(text, page_index)``; return value is stored as the
        page's classification payload.
    return_page_results : bool
        When ``True``, return ``(text, page_results)`` instead of just
        ``text``.
    """
    dpi = _get_ocr_dpi()
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        return ("", []) if return_page_results else ""

    max_workers = _get_page_workers(len(pages))
    results: List[Tuple[int, str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page: Dict[Future, int] = {}
        for index, page in enumerate(pages):
            fut = executor.submit(_process_ocr_page, index, page, page_classifier)
            future_to_page[fut] = index

        for future in as_completed(future_to_page):
            page_idx = future_to_page[future]
            try:
                results.append(future.result(timeout=_get_page_timeout()))
            except Exception as exc:
                logger.error(
                    "OCR failed for page %d of '%s': %s", page_idx + 1, pdf_path, exc,
                )
                results.append((page_idx, "", None))

    ordered = _sorted_page_results(results)
    full_text = _assemble_text(ordered)

    if return_page_results:
        return full_text, _build_page_results(ordered)
    return full_text


# ── Direct text extraction ───────────────────────────────────────

def extract_text_directly(
    pdf_path: str,
    *,
    page_classifier: PageClassifier = None,
    return_page_results: bool = False,
):
    """Extract text from a text-based PDF using PyMuPDF.

    The PDF is opened **once** in the calling thread; each worker
    receives the open :class:`fitz.Document` handle and loads only its
    assigned page, avoiding the N-opens-per-N-pages resource leak.

    Per-page failures are isolated — a failing page contributes empty
    text rather than crashing the entire document.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_classifier : callable, optional
        Receives ``(text, page_index)``; return value stored per-page.
    return_page_results : bool
        When ``True``, return ``(text, page_results)`` tuple.
    """
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        if page_count == 0:
            return ("", []) if return_page_results else ""

        max_workers = _get_page_workers(page_count)
        results: List[Tuple[int, str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page: Dict[Future, int] = {}
            for index in range(page_count):
                fut = executor.submit(
                    _process_direct_page, doc, index, page_classifier,
                )
                future_to_page[fut] = index

            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    results.append(future.result(timeout=_get_page_timeout()))
                except Exception as exc:
                    logger.error(
                        "Direct extraction failed for page %d of '%s': %s",
                        page_idx + 1,
                        pdf_path,
                        exc,
                    )
                    results.append((page_idx, "", None))

    ordered = _sorted_page_results(results)
    full_text = _assemble_text(ordered)

    if return_page_results:
        return full_text, _build_page_results(ordered)
    return full_text


# ── Auto-detect convenience ──────────────────────────────────────

def extract_text_from_pdf(
    pdf_path: str,
    *,
    page_classifier: PageClassifier = None,
    return_page_results: bool = False,
):
    """Detect PDF type and extract text using the appropriate method.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_classifier : callable, optional
        Executed per page after extraction.
    return_page_results : bool
        When ``True``, also return per-page metadata.
    """
    if is_pdf_text_based(pdf_path):
        return extract_text_directly(
            pdf_path,
            page_classifier=page_classifier,
            return_page_results=return_page_results,
        )
    else:
        return extract_text_from_pdf_ocr(
            pdf_path,
            page_classifier=page_classifier,
            return_page_results=return_page_results,
        )
