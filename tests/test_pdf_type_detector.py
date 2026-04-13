"""Tests for src.pipeline.pdf_type_detector."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.pdf_type_detector import is_pdf_text_based


class TestIsPdfTextBased:
    """Unit tests with mocked fitz.open."""

    def _mock_doc(self, page_texts):
        """Build a fake fitz.Document whose pages yield *page_texts*."""
        pages = []
        for i, text in enumerate(page_texts):
            page = MagicMock()
            page.get_text.return_value = text
            page.number = i
            pages.append(page)

        doc = MagicMock()
        doc.__iter__ = lambda self: iter(pages)
        doc.close = MagicMock()
        return doc

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_text_based_pdf(self, mock_open):
        mock_open.return_value = self._mock_doc(["Hello world, enough text here!"])
        assert is_pdf_text_based("dummy.pdf", min_char_threshold=5) is True

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_scanned_pdf(self, mock_open):
        mock_open.return_value = self._mock_doc(["", "  "])
        assert is_pdf_text_based("dummy.pdf", min_char_threshold=5) is False

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_threshold_boundary_below(self, mock_open):
        mock_open.return_value = self._mock_doc(["abcd"])  # 4 chars
        assert is_pdf_text_based("dummy.pdf", min_char_threshold=5) is False

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_threshold_boundary_equal(self, mock_open):
        mock_open.return_value = self._mock_doc(["abcde"])  # 5 chars
        assert is_pdf_text_based("dummy.pdf", min_char_threshold=5) is True

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_multi_page_first_empty(self, mock_open):
        mock_open.return_value = self._mock_doc(["", "enough text on page 2"])
        assert is_pdf_text_based("dummy.pdf", min_char_threshold=5) is True

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_closes_document(self, mock_open):
        doc = self._mock_doc(["text"])
        mock_open.return_value = doc
        is_pdf_text_based("dummy.pdf", min_char_threshold=3)
        doc.close.assert_called_once()

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_closes_document_on_scanned(self, mock_open):
        doc = self._mock_doc([""])
        mock_open.return_value = doc
        is_pdf_text_based("dummy.pdf", min_char_threshold=5)
        doc.close.assert_called_once()

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_corrupt_pdf_raises_pdf_open_error(self, mock_open):
        from src.pipeline.exceptions import PDFOpenError
        mock_open.side_effect = Exception("corrupt file")
        with pytest.raises(PDFOpenError):
            is_pdf_text_based("bad.pdf")

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_default_threshold_from_config(self, mock_open):
        """When no explicit threshold is given, config default (10) is used."""
        mock_open.return_value = self._mock_doc(["short"])  # 5 chars < 10
        assert is_pdf_text_based("dummy.pdf") is False

    @patch("src.pipeline.pdf_type_detector.fitz.open")
    def test_empty_pdf(self, mock_open):
        mock_open.return_value = self._mock_doc([])
        assert is_pdf_text_based("empty.pdf", min_char_threshold=1) is False
