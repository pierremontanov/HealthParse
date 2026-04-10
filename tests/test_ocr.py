"""Tests for src.pipeline.ocr – image-based text extraction."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.pipeline.ocr import extract_text_from_image, ocr_pil_image


# ── extract_text_from_image ─────────────────────────────────────

class TestExtractTextFromImage:
    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="hello world")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_returns_extracted_text(self, mock_imread, mock_preprocess, mock_tess):
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = fake_img
        result = extract_text_from_image("/fake/image.png")
        assert result == "hello world"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="hola mundo")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_passes_lang_to_tesseract(self, mock_imread, mock_preprocess, mock_tess):
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = fake_img
        extract_text_from_image("/fake/image.png", lang="spa")
        _, kwargs = mock_tess.call_args
        assert kwargs["lang"] == "spa"

    @patch("src.pipeline.ocr.cv2.imread", return_value=None)
    def test_returns_empty_on_bad_image(self, mock_imread):
        result = extract_text_from_image("/fake/nonexistent.png")
        assert result == ""

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="text")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_calls_preprocess(self, mock_imread, mock_preprocess, mock_tess):
        fake_img = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_imread.return_value = fake_img
        extract_text_from_image("/fake/image.jpg")
        mock_preprocess.assert_called_once()

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="text")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_default_lang_from_config(self, mock_imread, mock_preprocess, mock_tess):
        """When no lang override, _get_ocr_lang reads from config."""
        fake_img = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_imread.return_value = fake_img
        mock_settings = SimpleNamespace(ocr_lang="fra", tesseract_cmd=None)
        with patch("src.config.settings", mock_settings):
            extract_text_from_image("/fake/image.jpg")
        _, kwargs = mock_tess.call_args
        assert kwargs["lang"] == "fra"


# ── ocr_pil_image ───────────────────────────────────────────────

class TestOcrPilImage:
    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="pil text")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_returns_text_from_pil(self, mock_preprocess, mock_tess):
        pil_img = Image.new("RGB", (100, 100), color="white")
        result = ocr_pil_image(pil_img)
        assert result == "pil text"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="lang test")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_passes_lang(self, mock_preprocess, mock_tess):
        pil_img = Image.new("RGB", (100, 100))
        ocr_pil_image(pil_img, lang="deu")
        _, kwargs = mock_tess.call_args
        assert kwargs["lang"] == "deu"

    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="gray ok")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    def test_handles_grayscale_input(self, mock_preprocess, mock_tess):
        pil_img = Image.new("L", (100, 100), color=128)
        result = ocr_pil_image(pil_img)
        assert result == "gray ok"


# ── tesseract_cmd config ────────────────────────────────────────

class TestTesseractCmdConfig:
    @patch("src.pipeline.ocr.pytesseract.pytesseract")
    @patch("src.pipeline.ocr.pytesseract.image_to_string", return_value="")
    @patch("src.pipeline.ocr.preprocess_image", side_effect=lambda img: img)
    @patch("src.pipeline.ocr.cv2.imread")
    def test_applies_tesseract_cmd(self, mock_imread, mock_pp, mock_tess, mock_pyt):
        fake_img = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_imread.return_value = fake_img
        mock_settings = SimpleNamespace(
            ocr_lang="eng", tesseract_cmd="/custom/tesseract"
        )
        with patch("src.config.settings", mock_settings):
            extract_text_from_image("/fake/image.png")
        # Verify tesseract_cmd was set on the module
        assert mock_pyt.tesseract_cmd == "/custom/tesseract"
