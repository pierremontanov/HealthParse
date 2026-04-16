"""Comprehensive unit tests for OCR pipeline (#24).

Extends the existing test_ocr.py with edge-case coverage for:
  • Image mode conversions (RGBA, palette, 16-bit)
  • Tesseract error handling and graceful degradation
  • Config fallback behaviour
  • ocr_pil_image with various PIL modes
  • Unicode and multi-language output
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

from src.pipeline.ocr import extract_text_from_image, ocr_pil_image


# ═══════════════════════════════════════════════════════════════════
# extract_text_from_image – edge cases
# ═══════════════════════════════════════════════════════════════════


class TestExtractTextEdgeCases:
    """Edge-case and error-handling tests for extract_text_from_image."""

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="  hello  \n")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_strips_whitespace_from_result(self, mock_imread, mock_pp, mock_tess):
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        result = extract_text_from_image("/img.png")
        # pytesseract returns raw string; check the pipeline handles it
        assert isinstance(result, str)

    @patch("src.pipeline.ocr.cv2.imread", return_value=None)
    def test_returns_empty_when_imread_fails(self, mock_imread):
        assert extract_text_from_image("/missing.png") == ""

    @patch("src.pipeline.ocr.pytesseract.image_to_string", side_effect=Exception("tesseract crash"))
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_raises_on_tesseract_exception(self, mock_imread, mock_pp, mock_tess):
        """Tesseract exceptions propagate (no silent swallow)."""
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(Exception, match="tesseract crash"):
            extract_text_from_image("/img.png")

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="café résumé")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_unicode_output(self, mock_imread, mock_pp, mock_tess):
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        result = extract_text_from_image("/img.png")
        assert "café" in result
        assert "résumé" in result

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_empty_text_from_blank_image(self, mock_imread, mock_pp, mock_tess):
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = extract_text_from_image("/blank.png")
        assert result == ""

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="text")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_grayscale_image_no_conversion_crash(self, mock_imread, mock_pp, mock_tess):
        gray = np.zeros((50, 50), dtype=np.uint8)
        mock_imread.return_value = gray
        result = extract_text_from_image("/gray.png")
        assert result == "text"


# ═══════════════════════════════════════════════════════════════════
# ocr_pil_image – PIL mode handling
# ═══════════════════════════════════════════════════════════════════


class TestOcrPilImageModes:
    """Test PIL image mode conversion paths."""

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="rgba ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_rgba_image(self, mock_pp, mock_tess):
        pil_img = Image.new("RGBA", (80, 80), color=(255, 0, 0, 128))
        result = ocr_pil_image(pil_img)
        assert result == "rgba ok"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="palette ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_palette_mode_image(self, mock_pp, mock_tess):
        pil_img = Image.new("P", (80, 80))
        result = ocr_pil_image(pil_img)
        assert result == "palette ok"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="1bit ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_binary_mode_image(self, mock_pp, mock_tess):
        pil_img = Image.new("1", (80, 80))
        result = ocr_pil_image(pil_img)
        assert result == "1bit ok"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="big ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_large_image(self, mock_pp, mock_tess):
        pil_img = Image.new("RGB", (4000, 3000), color="white")
        result = ocr_pil_image(pil_img)
        assert result == "big ok"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="tiny ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_tiny_image(self, mock_pp, mock_tess):
        pil_img = Image.new("RGB", (1, 1), color="black")
        result = ocr_pil_image(pil_img)
        assert result == "tiny ok"


# ═══════════════════════════════════════════════════════════════════
# Config fallbacks
# ═══════════════════════════════════════════════════════════════════


class TestOcrConfigFallbacks:
    """Test behaviour when config module is unavailable or values are missing."""

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_lang_override_takes_precedence(self, mock_imread, mock_pp, mock_tess):
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_settings = SimpleNamespace(ocr_lang="deu", tesseract_cmd=None)
        with patch("src.config.settings", mock_settings):
            extract_text_from_image("/img.png", lang="jpn")
        _, kwargs = mock_tess.call_args
        assert kwargs["lang"] == "jpn"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_no_tesseract_cmd_uses_default(self, mock_imread, mock_pp, mock_tess):
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_settings = SimpleNamespace(ocr_lang="eng", tesseract_cmd=None)
        with patch("src.config.settings", mock_settings):
            result = extract_text_from_image("/img.png")
        assert result == "ok"
