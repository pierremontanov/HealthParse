from __future__ import annotations

import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from src.pipeline.preprocess import preprocess_image
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Tuple

PageClassifier = Optional[Callable[[str, int], Any]]


def _run_classifier(text: str, page_index: int, classifier: PageClassifier) -> Any:
    if classifier is None:
        return None
    return classifier(text, page_index)


def _sorted_page_results(results: Iterable[Tuple[int, str, Any]]) -> List[Tuple[int, str, Any]]:
    return sorted(results, key=lambda item: item[0])

def _process_ocr_page(
    page_index: int,
    page_image,
    classifier: PageClassifier,
):
    img = page_image.convert("RGB")  # PIL to RGB
    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL to OpenCV
    preprocessed = preprocess_image(open_cv_image)
    text = pytesseract.image_to_string(preprocessed)
    classification = _run_classifier(text, page_index, classifier)
    return page_index, text, classification


def extract_text_from_pdf_ocr(
    pdf_path: str,
    *,
    page_classifier: PageClassifier = None,
    return_page_results: bool = False,
):
    """
    Convert each page of a scanned PDF to an image, then extract and combine text using OCR.

    The heavy-weight OCR preprocessing and classification are executed in a
    thread pool so that multi-page documents can be processed faster.

    Args:
        pdf_path (str): Path to the PDF file.
        page_classifier: Optional callable that receives the extracted text and
            zero-based page index, returning any classification payload. The
            callable is executed inside the worker thread.
        return_page_results: When ``True`` the function returns both the
            combined text and a list of per-page dictionaries containing the
            page number, raw text, and classification result.

    Returns:
        str or Tuple[str, List[dict]]: Combined text extracted from all pages.
            When ``return_page_results`` is enabled, the tuple also contains the
            per-page metadata.
    """
    pages = convert_from_path(pdf_path, dpi=300)
    if not pages:
        return ("", []) if return_page_results else ""

    max_workers = min(len(pages), (os.cpu_count() or 1) * 2)
    results: List[Tuple[int, str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_ocr_page, index, page, page_classifier)
            for index, page in enumerate(pages)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    ordered = _sorted_page_results(results)
    full_text = "".join(
        f"\n--- Page {index + 1} ---\n{text}" for index, text, _ in ordered
    )

    if return_page_results:
        page_results = [
            {
                "page": index + 1,
                "text": text,
                "classification": classification,
            }
            for index, text, classification in ordered
        ]
        return full_text, page_results

    return full_text

def _process_direct_page(
    pdf_path: str,
    page_index: int,
    classifier: PageClassifier,
):
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        text = page.get_text()
    classification = _run_classifier(text, page_index, classifier)
    return page_index, text, classification


def extract_text_directly(
    pdf_path: str,
    *,
    page_classifier: PageClassifier = None,
    return_page_results: bool = False,
):
    """
    Extract text directly from a text-based PDF using PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.
        page_classifier: Optional callable executed per page inside the worker
            thread. It receives the extracted text and zero-based page index.
        return_page_results: When ``True`` the function returns the combined
            text together with per-page metadata containing classification
            results.

    Returns:
        str or Tuple[str, List[dict]]: Combined extracted text from all pages.
            When ``return_page_results`` is enabled, the tuple also contains the
            per-page metadata.
    """
    with fitz.open(pdf_path) as doc:
        page_indices = list(range(doc.page_count))

    if not page_indices:
        return ("", []) if return_page_results else ""

    max_workers = min(len(page_indices), (os.cpu_count() or 1) * 2)
    results: List[Tuple[int, str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_direct_page, pdf_path, index, page_classifier)
            for index in page_indices
        ]

        for future in as_completed(futures):
            results.append(future.result())

    ordered = _sorted_page_results(results)
    full_text = "".join(
        f"\n--- Page {index + 1} ---\n{text}" for index, text, _ in ordered
    )

    if return_page_results:
        page_results = [
            {
                "page": index + 1,
                "text": text,
                "classification": classification,
            }
            for index, text, classification in ordered
        ]
        return full_text, page_results

    return full_text

def is_pdf_text_based(pdf_path, min_char_threshold=10):
    """
    Check if a PDF is text-based by attempting to extract text from its pages.

    Args:
        pdf_path (str): Path to the PDF file.
        min_char_threshold (int): Minimum number of characters to consider the page 'text-based'.

    Returns:
        bool: True if the PDF has extractable text, False if it's likely scanned.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        if len(text.strip()) >= min_char_threshold:
            return True
    return False


def extract_text_from_pdf(
    pdf_path: str,
    *,
    page_classifier: PageClassifier = None,
    return_page_results: bool = False,
):
    """
    Automatically detect PDF type and extract text using the appropriate method.

    Args:
        pdf_path (str): Path to the PDF file.
        page_classifier: Optional callable executed per page after extraction.
        return_page_results: When ``True`` return both the aggregated text and
            the per-page metadata.

    Returns:
        str or Tuple[str, List[dict]]: Extracted text. When
            ``return_page_results`` is enabled, the tuple also contains the
            per-page metadata.
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
