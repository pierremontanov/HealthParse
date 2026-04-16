"""Tests for src.pipeline.preprocess – image & text preprocessing."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.pipeline.preprocess import preprocess_image, preprocess_text


# ── preprocess_image ────────────────────────────────────────────

class TestPreprocessImage:
    def test_returns_single_channel(self):
        """Output should be a 2D grayscale array."""
        bgr = np.zeros((100, 80, 3), dtype=np.uint8)
        result = preprocess_image(bgr)
        assert len(result.shape) == 2

    def test_output_shape_matches_input(self):
        bgr = np.zeros((120, 200, 3), dtype=np.uint8)
        result = preprocess_image(bgr)
        assert result.shape == (120, 200)

    def test_binary_output_values(self):
        """All pixel values should be either 0 or 255."""
        bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = preprocess_image(bgr)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_threshold_override(self):
        """Explicit threshold param should take effect."""
        # Create an image with mid-gray pixels (128)
        gray_val = 128
        bgr = np.full((10, 10, 3), gray_val, dtype=np.uint8)

        # threshold=100 → 128 > 100 → white (255)
        high = preprocess_image(bgr, threshold=100)
        assert np.all(high == 255)

        # threshold=200 → 128 < 200 → black (0)
        low = preprocess_image(bgr, threshold=200)
        assert np.all(low == 0)

    def test_reads_config_threshold(self):
        """When no override, threshold comes from config."""
        bgr = np.full((10, 10, 3), 150, dtype=np.uint8)
        mock_settings = SimpleNamespace(preprocessing_threshold=200)
        with patch("src.config.settings", mock_settings):
            result = preprocess_image(bgr)
        # 150 < 200 → black
        assert np.all(result == 0)

    def test_already_grayscale_input(self):
        """Single-channel input should not crash."""
        gray = np.zeros((50, 50), dtype=np.uint8)
        result = preprocess_image(gray)
        assert result.shape == (50, 50)

    def test_white_image(self):
        bgr = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = preprocess_image(bgr)
        assert np.all(result == 255)

    def test_black_image(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = preprocess_image(bgr)
        assert np.all(result == 0)


# ── preprocess_text ─────────────────────────────────────────────

class TestPreprocessText:
    @patch("src.pipeline.preprocess.detect_language", return_value="en")
    def test_lowercases_text(self, mock_lang):
        result = preprocess_text("Hello WORLD")
        assert result == result.lower()

    @patch("src.pipeline.preprocess.detect_language", return_value="en")
    def test_collapses_whitespace(self, mock_lang):
        result = preprocess_text("hello   world\n\nfoo")
        assert "  " not in result

    @patch("src.pipeline.preprocess.detect_language", return_value="en")
    def test_unicode_normalisation(self, mock_lang):
        # NFKD normalises ﬁ (U+FB01) to fi
        result = preprocess_text("ﬁnd")
        assert "fi" in result

    @patch("src.pipeline.preprocess.detect_language", return_value="en")
    def test_removes_special_chars(self, mock_lang):
        result = preprocess_text("hello @#$ world!")
        # @ and # should be removed, word chars + punctuation kept
        assert "@" not in result
        assert "#" not in result

    @patch("src.pipeline.preprocess.detect_language", return_value="es")
    def test_detects_language(self, mock_lang):
        preprocess_text("Hola mundo esto es texto en español")
        mock_lang.assert_called_once()

    @patch("src.pipeline.preprocess.detect_language", return_value="en")
    def test_empty_string(self, mock_lang):
        result = preprocess_text("")
        assert result == ""
