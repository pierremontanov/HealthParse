"""Image and text preprocessing for the DocIQ pipeline.

Image preprocessing converts BGR images to grayscale and applies binary
thresholding (configurable via ``settings.preprocessing_threshold``) to
produce clean input for Tesseract OCR.

Text preprocessing applies Unicode normalisation, whitespace collapsing,
special-character removal, and lowercasing for downstream NER and
classification models.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.pipeline.utils.text_utils import clean_text, lowercase

logger = logging.getLogger(__name__)


# ── Config helper ───────────────────────────────────────────────

def _get_threshold() -> int:
    """Return the configured binarisation threshold (default 120)."""
    try:
        from src.config import settings
        return settings.preprocessing_threshold
    except Exception:
        return 120


# ── Image preprocessing ─────────────────────────────────────────

def preprocess_image(
    image: np.ndarray,
    *,
    threshold: Optional[int] = None,
) -> np.ndarray:
    """Convert a BGR image to a binary (black-and-white) image.

    Steps:
    1. Convert BGR → grayscale.
    2. Apply global binary thresholding at the configured threshold
       (default 120, configurable via ``settings.preprocessing_threshold``
       or the *threshold* parameter).

    Parameters
    ----------
    image : np.ndarray
        An OpenCV BGR image.
    threshold : int, optional
        Override the binarisation threshold.  When ``None``, the value
        from ``settings.preprocessing_threshold`` is used.

    Returns
    -------
    np.ndarray
        A single-channel binary (0/255) image.
    """
    thresh_val = threshold if threshold is not None else _get_threshold()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # already grayscale

    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    logger.debug("Preprocessed image: shape=%s threshold=%d", binary.shape, thresh_val)
    return binary


# ── Text preprocessing ──────────────────────────────────────────

def preprocess_text(raw_text: str) -> str:
    """Clean and normalise extracted text for NER/classification.

    Steps:
    1. Detect language (logged for traceability).
    2. Unicode NFKD normalisation + whitespace collapsing + special-char
       removal (via :func:`clean_text`).
    3. Lowercase conversion.

    Parameters
    ----------
    raw_text : str
        Raw OCR or direct-extraction output.

    Returns
    -------
    str
        Cleaned, lowercased text.
    """
    from src.pipeline.utils.language import detect_language

    lang = detect_language(raw_text)
    logger.debug("Language detected during preprocessing: %s", lang)

    text = clean_text(raw_text)
    text = lowercase(text)
    return text
