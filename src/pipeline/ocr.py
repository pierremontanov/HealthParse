"""Standalone OCR module for image files.

Handles PNG/JPEG image extraction via Tesseract OCR with configurable
DPI, language packs, preprocessing threshold, and Tesseract/Poppler paths
read from :mod:`src.config`.

Usage
-----
    from src.pipeline.ocr import extract_text_from_image

    text = extract_text_from_image("scan.png")
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from src.pipeline.preprocess import preprocess_image

logger = logging.getLogger(__name__)


# ── Config helpers ──────────────────────────────────────────────

def _get_ocr_lang() -> str:
    """Return the configured Tesseract language pack string."""
    try:
        from src.config import settings
        return settings.ocr_lang
    except Exception:
        return "eng+spa"


def _apply_tesseract_cmd() -> None:
    """Point pytesseract at a custom binary if configured."""
    try:
        from src.config import settings
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            logger.debug("Tesseract cmd set to: %s", settings.tesseract_cmd)
    except Exception:
        pass


# ── Public API ──────────────────────────────────────────────────

def extract_text_from_image(image_path: str, *, lang: Optional[str] = None) -> str:
    """Load an image file, preprocess it, and extract text via Tesseract.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, or other image file readable by OpenCV.
    lang : str, optional
        Tesseract language pack override (e.g. ``"eng+spa"``).
        When ``None``, the value from ``settings.ocr_lang`` is used.

    Returns
    -------
    str
        Extracted text, or an empty string on failure.
    """
    _apply_tesseract_cmd()
    ocr_lang = lang or _get_ocr_lang()

    image = cv2.imread(image_path)
    if image is None:
        logger.error("Could not load image: %s", image_path)
        return ""

    preprocessed = preprocess_image(image)
    pil_image = Image.fromarray(preprocessed)
    text = pytesseract.image_to_string(pil_image, lang=ocr_lang)
    logger.debug(
        "OCR extracted %d chars from %s (lang=%s)",
        len(text),
        image_path,
        ocr_lang,
    )
    return text


def ocr_pil_image(pil_image: Image.Image, *, lang: Optional[str] = None) -> str:
    """Run OCR on an already-loaded PIL image.

    Useful when the caller has already converted a PDF page to an image
    (e.g. via ``pdf2image``).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        An RGB or grayscale PIL image.
    lang : str, optional
        Tesseract language pack override.

    Returns
    -------
    str
        Extracted text.
    """
    _apply_tesseract_cmd()
    ocr_lang = lang or _get_ocr_lang()

    rgb = pil_image.convert("RGB")
    open_cv_image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess_image(open_cv_image)
    result_image = Image.fromarray(preprocessed)
    text = pytesseract.image_to_string(result_image, lang=ocr_lang)
    return text
