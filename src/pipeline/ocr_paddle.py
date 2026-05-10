"""PaddleOCR-based text extraction module (PaddleOCR 3.4.0+).

Drop-in replacement for the Tesseract-based :mod:`src.pipeline.ocr` module.
Uses a singleton :class:`PaddleOCR` instance (CPU, PP-OCRv4, Spanish
language) so that model weights are loaded only once.

PaddleOCR 3.4.0 uses PaddleX as its inference backend.  The critical
fix for CPU inference is ``enable_mkldnn=False``, which forces
``run_mode="paddle"`` instead of the broken ``run_mode="mkldnn"``
path in PaddlePaddle 3.x.

Pre-download models
-------------------
Run as a script to download model weights ahead of time
(useful for Docker builds or CI):

    python -m src.pipeline.ocr_paddle --download

Usage
-----
    from src.pipeline.ocr_paddle import extract_text_from_image

    text = extract_text_from_image("scan.png")
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional

# Disable oneDNN as additional insurance (PaddleX controls the real
# toggle via enable_mkldnn, but belt-and-suspenders doesn't hurt).
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
# Skip slow HuggingFace/ModelScope connectivity check on startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Singleton management ──────────────────────────────────────────

_ocr_instance = None
_ocr_lock = threading.Lock()


def _get_ocr_instance():
    """Return the singleton PaddleOCR instance, creating it on first call.

    Thread-safe via a lock.  The instance is reused for all subsequent
    calls to avoid reloading model weights each time.
    """
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    with _ocr_lock:
        # Double-check after acquiring the lock.
        if _ocr_instance is not None:
            return _ocr_instance

        from paddleocr import PaddleOCR

        logger.info("Initialising PaddleOCR 3.4 (CPU, PP-OCRv4, lang=en) …")
        _ocr_instance = PaddleOCR(
            ocr_version="PP-OCRv4",     # PP-OCRv4 models
            lang="en",                   # English/Latin — handles Spanish text
            enable_mkldnn=False,         # CRITICAL: bypass broken oneDNN/PIR path
        )
        logger.info("PaddleOCR initialised successfully.")
        return _ocr_instance


# ── Internal helpers ──────────────────────────────────────────────

def _predict_result_to_text(results) -> str:
    """Convert PaddleOCR 3.4 predict() results into a plain string.

    PaddleOCR 3.4's ``predict()`` returns an iterator of result objects.
    Each result has a ``rec_texts`` list and ``rec_scores`` list.

    Falls back to checking for older-style dict/list structures.
    """
    lines: list[str] = []

    for result in results:
        # PaddleOCR 3.4 result objects have rec_texts attribute
        if hasattr(result, "rec_texts"):
            lines.extend(result.rec_texts)
        # Fallback: dict-style results
        elif isinstance(result, dict):
            if "rec_texts" in result:
                lines.extend(result["rec_texts"])
            elif "rec_text" in result:
                lines.append(result["rec_text"])
        # Fallback: old nested list format
        elif isinstance(result, (list, tuple)):
            for detection in result:
                if isinstance(detection, (list, tuple)) and len(detection) >= 2:
                    text_info = detection[1]
                    if isinstance(text_info, (list, tuple)):
                        lines.append(str(text_info[0]))
                    else:
                        lines.append(str(text_info))

    return "\n".join(lines)


# ── Public API (same signatures as ocr.py) ────────────────────────

def extract_text_from_image(
    image_path: str,
    *,
    lang: Optional[str] = None,
) -> str:
    """Load an image file and extract text via PaddleOCR.

    Parameters
    ----------
    image_path : str
        Path to a PNG, JPEG, or other common image file.
    lang : str, optional
        Language override.  Ignored — the singleton is initialised
        with Spanish + English.  Kept for API compatibility.

    Returns
    -------
    str
        Extracted text, or an empty string on failure.
    """
    ocr = _get_ocr_instance()

    try:
        results = list(ocr.predict(image_path))
        text = _predict_result_to_text(results)
        logger.debug(
            "PaddleOCR extracted %d chars from %s",
            len(text),
            image_path,
        )
        return text
    except Exception as exc:
        logger.error("PaddleOCR failed on %s: %s", image_path, exc)
        return ""


def ocr_pil_image(
    pil_image: Image.Image,
    *,
    lang: Optional[str] = None,
) -> str:
    """Run PaddleOCR on an already-loaded PIL image.

    Useful when the caller has already converted a PDF page to an
    image (e.g. via ``pdf2image``).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        An RGB or grayscale PIL image.
    lang : str, optional
        Language override (kept for API compatibility; see above).

    Returns
    -------
    str
        Extracted text.
    """
    ocr = _get_ocr_instance()

    # PaddleOCR accepts numpy arrays (RGB).
    rgb = pil_image.convert("RGB")
    img_array = np.array(rgb)

    try:
        results = list(ocr.predict(img_array))
        text = _predict_result_to_text(results)
        logger.debug(
            "PaddleOCR extracted %d chars from PIL image (%dx%d)",
            len(text),
            pil_image.width,
            pil_image.height,
        )
        return text
    except Exception as exc:
        logger.error("PaddleOCR failed on PIL image: %s", exc)
        return ""


# ── Model download / warm-up ─────────────────────────────────────

def download_models(lang: str = "en") -> None:
    """Pre-download PaddleOCR model weights.

    Instantiating :class:`PaddleOCR` automatically downloads the
    detection, recognition, and angle-classification models if they
    are not already cached locally.

    Call this during Docker build, CI setup, or first-time install
    so the first real OCR request doesn't block on a download.

    Parameters
    ----------
    lang : str
        PaddleOCR language code (default ``"es"``).
    """
    from paddleocr import PaddleOCR

    print(f"Downloading PaddleOCR models (lang={lang}) …")
    t0 = time.time()
    PaddleOCR(
        ocr_version="PP-OCRv4",
        lang=lang,
        enable_mkldnn=False,
    )
    elapsed = time.time() - t0
    print(f"PaddleOCR models ready ({elapsed:.1f}s).")


def warmup() -> bool:
    """Eagerly initialise the singleton and confirm models are loaded.

    Returns ``True`` when the engine is ready, ``False`` on error.
    Intended as a startup health-check (e.g. in FastAPI ``lifespan``).
    """
    try:
        _get_ocr_instance()
        return True
    except Exception as exc:
        logger.error("PaddleOCR warm-up failed: %s", exc)
        return False


# ── CLI entry-point ──────────────────────────────────────────────

def _cli() -> None:
    """Minimal CLI for model management and quick OCR tests."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.ocr_paddle",
        description="PaddleOCR model management and quick OCR test.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model weights and exit.",
    )
    parser.add_argument(
        "--lang",
        default="es",
        help="PaddleOCR language code (default: es).",
    )
    parser.add_argument(
        "--test",
        metavar="IMAGE",
        help="Run OCR on a single image and print the extracted text.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.download:
        download_models(lang=args.lang)
        return

    if args.test:
        text = extract_text_from_image(args.test)
        if text:
            print(f"\n── Extracted text ({len(text)} chars) ──\n")
            print(text)
        else:
            print("No text extracted.", file=sys.stderr)
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
