"""Tests for src.pipeline.language – language detection utilities."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.pipeline.language import (
    PDFLanguageDetection,
    _normalise_language,
    detect_language,
    detect_pdf_language,
)


# ── _normalise_language ──────────────────────────────────────────

class TestNormaliseLanguage:
    def test_exact_match(self):
        assert _normalise_language("en", {"en", "es"}) == "en"

    def test_region_suffix(self):
        assert _normalise_language("en-us", {"en", "es"}) == "en"

    def test_unsupported(self):
        assert _normalise_language("fr", {"en", "es"}) is None

    def test_empty_string(self):
        assert _normalise_language("", {"en", "es"}) is None

    def test_custom_supported_set(self):
        assert _normalise_language("fr", {"fr", "de"}) == "fr"


# ── detect_language ──────────────────────────────────────────────

class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello, this is a test.") == "en"

    def test_spanish(self):
        assert detect_language("Hola, esto es una prueba.") == "es"

    def test_empty_returns_unknown(self):
        assert detect_language("") == "unknown"

    def test_whitespace_only_returns_unknown(self):
        assert detect_language("   \n\t  ") == "unknown"

    def test_unsupported_language_returns_unknown(self):
        # Very short or ambiguous text may not be reliably detected,
        # but a clear French sentence should not match en/es.
        result = detect_language("Bonjour, comment allez-vous aujourd'hui mon ami?")
        assert result == "unknown"

    def test_custom_supported_set(self):
        # French text with French in supported set
        result = detect_language(
            "Bonjour, comment allez-vous aujourd'hui mon ami?",
            supported={"fr", "de"},
        )
        assert result == "fr"


# ── detect_pdf_language ──────────────────────────────────────────

class TestDetectPdfLanguage:
    def _make_mock_doc(self, page_texts):
        """Build a fake fitz Document."""

        class DummyDoc(list):
            def close(self):
                pass

        pages = [
            SimpleNamespace(get_text=lambda mode="text", t=t: t if mode == "text" else "")
            for t in page_texts
        ]
        return DummyDoc(pages)

    @patch("src.pipeline.language.fitz.open")
    def test_english_pdf(self, mock_open):
        mock_open.return_value = self._make_mock_doc(
            ["Hello world, this is an English document."]
        )
        result = detect_pdf_language("dummy.pdf", max_pages=1)
        assert isinstance(result, PDFLanguageDetection)
        assert result.language == "en"
        assert "Hello" in result.text_sample

    @patch("src.pipeline.language.fitz.open")
    def test_spanish_pdf(self, mock_open):
        mock_open.return_value = self._make_mock_doc(
            ["Hola, este es un documento en español con suficiente texto."]
        )
        result = detect_pdf_language("dummy.pdf", max_pages=1, min_characters=10)
        assert result.language == "es"

    @patch("src.pipeline.language.fitz.open")
    def test_empty_pdf_returns_unknown(self, mock_open):
        mock_open.return_value = self._make_mock_doc([""])
        result = detect_pdf_language("dummy.pdf", max_pages=1, min_characters=10)
        assert result.language == "unknown"

    @patch("src.pipeline.language.fitz.open")
    def test_respects_max_pages(self, mock_open):
        mock_open.return_value = self._make_mock_doc(
            ["Page one.", "Page two.", "Page three."]
        )
        result = detect_pdf_language("dummy.pdf", max_pages=2, min_characters=9999)
        # Should only sample 2 pages even though 3 exist
        assert result.text_sample.count("Page") <= 2

    @patch("src.pipeline.language.fitz.open")
    def test_stops_early_when_enough_chars(self, mock_open):
        mock_open.return_value = self._make_mock_doc(
            [
                "This is enough text to satisfy the minimum character requirement.",
                "This page should not be sampled.",
            ]
        )
        result = detect_pdf_language("dummy.pdf", max_pages=5, min_characters=10)
        assert "should not be sampled" not in result.text_sample

    @patch("src.pipeline.language.fitz.open")
    def test_defaults_from_config(self, mock_open):
        """When no kwargs are passed, config defaults are used (max_pages=3, min_chars=120)."""
        mock_open.return_value = self._make_mock_doc(
            ["Short text."]  # < 120 chars but only 1 page
        )
        result = detect_pdf_language("dummy.pdf")
        # Should still return a result (even if unknown) without crashing
        assert isinstance(result, PDFLanguageDetection)
