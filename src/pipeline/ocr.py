"""Standalone OCR module for image files.

Delegates to PaddleOCR (3.4+, PP-OCRv4) via :mod:`src.pipeline.ocr_paddle`.

All existing import paths continue to work:

    from src.pipeline.ocr import extract_text_from_image

    text = extract_text_from_image("scan.png")
"""

from src.pipeline.ocr_paddle import extract_text_from_image, ocr_pil_image

__all__ = ["extract_text_from_image", "ocr_pil_image"]
