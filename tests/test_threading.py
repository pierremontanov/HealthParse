"""Tests for #25+#26 – Threaded processing (per-file & per-page).

Covers:
- Per-page error isolation (direct + OCR)
- Single fitz.open for direct extraction (no N-opens leak)
- Config-driven thread pool sizing
- Timeout protection
- Page ordering after async completion
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF required")

from src.pipeline.pdf_extractor import (
    _get_page_workers,
    _process_direct_page,
    _assemble_text,
    _build_page_results,
    _sorted_page_results,
    extract_text_directly,
)


# ── Helper factories ─────────────────────────────────────────────

def _create_pdf(tmp_path, n_pages=3, text_template="Page {} content"):
    """Create a real PDF with *n_pages* pages."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), text_template.format(i + 1))
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


# ── _get_page_workers ────────────────────────────────────────────

class TestGetPageWorkers:
    def test_scales_with_pages(self):
        # Should never exceed page count
        w = _get_page_workers(2)
        assert w <= 2

    def test_at_least_one(self):
        assert _get_page_workers(1) >= 1

    def test_respects_config_cap(self):
        # Patch settings.max_workers inside the lazy import
        mock_settings = SimpleNamespace(max_workers=4)
        with patch("src.config.settings", mock_settings):
            w = _get_page_workers(100)
            assert w == 4


# ── _sorted_page_results / _assemble_text / _build_page_results ─

class TestPageResultHelpers:
    def test_sorts_by_index(self):
        unsorted = [(2, "c", None), (0, "a", None), (1, "b", None)]
        ordered = _sorted_page_results(unsorted)
        assert [i for i, _, _ in ordered] == [0, 1, 2]

    def test_assemble_text(self):
        ordered = [(0, "hello", None), (1, "world", None)]
        text = _assemble_text(ordered)
        assert "--- Page 1 ---" in text
        assert "hello" in text
        assert "--- Page 2 ---" in text
        assert "world" in text

    def test_build_page_results(self):
        ordered = [(0, "text-a", "label-a"), (1, "text-b", None)]
        results = _build_page_results(ordered)
        assert len(results) == 2
        assert results[0] == {"page": 1, "text": "text-a", "classification": "label-a"}
        assert results[1]["classification"] is None


# ── _process_direct_page ─────────────────────────────────────────

class TestProcessDirectPage:
    def test_extracts_text_from_open_doc(self, tmp_path):
        pdf_path = _create_pdf(tmp_path, n_pages=2)
        with fitz.open(pdf_path) as doc:
            idx, text, cls = _process_direct_page(doc, 0, None)
        assert idx == 0
        assert "Page 1" in text
        assert cls is None

    def test_runs_classifier(self, tmp_path):
        pdf_path = _create_pdf(tmp_path, n_pages=1)
        classifier = lambda t, i: f"label-{i}"
        with fitz.open(pdf_path) as doc:
            idx, text, cls = _process_direct_page(doc, 0, classifier)
        assert cls == "label-0"


# ── extract_text_directly (integration) ──────────────────────────

class TestExtractTextDirectly:
    def test_extracts_all_pages(self, tmp_path):
        pdf_path = _create_pdf(tmp_path, n_pages=3)
        text = extract_text_directly(pdf_path)
        assert "--- Page 1 ---" in text
        assert "--- Page 2 ---" in text
        assert "--- Page 3 ---" in text

    def test_page_results_returned(self, tmp_path):
        pdf_path = _create_pdf(tmp_path, n_pages=2)
        text, pages = extract_text_directly(pdf_path, return_page_results=True)
        assert len(pages) == 2
        assert pages[0]["page"] == 1
        assert pages[1]["page"] == 2

    def test_classifier_called_per_page(self, tmp_path):
        pdf_path = _create_pdf(tmp_path, n_pages=2)
        calls = []
        def classifier(text, idx):
            calls.append(idx)
            return f"l-{idx}"
        text, pages = extract_text_directly(
            pdf_path, page_classifier=classifier, return_page_results=True,
        )
        assert set(calls) == {0, 1}
        assert pages[0]["classification"] == "l-0"

    def test_zero_page_pdf(self, tmp_path):
        """Simulate a zero-page PDF by mocking fitz.open."""
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_doc.__enter__ = lambda s: s
        mock_doc.__exit__ = MagicMock(return_value=False)
        with patch("src.pipeline.pdf_extractor.fitz.open", return_value=mock_doc):
            text = extract_text_directly(str(tmp_path / "fake.pdf"))
        assert text == ""

    def test_zero_page_pdf_with_page_results(self, tmp_path):
        """Simulate a zero-page PDF returning (text, pages) tuple."""
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_doc.__enter__ = lambda s: s
        mock_doc.__exit__ = MagicMock(return_value=False)
        with patch("src.pipeline.pdf_extractor.fitz.open", return_value=mock_doc):
            text, pages = extract_text_directly(
                str(tmp_path / "fake.pdf"), return_page_results=True,
            )
        assert text == ""
        assert pages == []

    def test_single_fitz_open(self, tmp_path):
        """Verify the PDF is opened only once (not N times for N pages)."""
        pdf_path = _create_pdf(tmp_path, n_pages=3)
        open_count = 0
        original_open = fitz.open

        def counting_open(*args, **kwargs):
            nonlocal open_count
            open_count += 1
            return original_open(*args, **kwargs)

        with patch("src.pipeline.pdf_extractor.fitz.open", side_effect=counting_open):
            extract_text_directly(pdf_path)

        assert open_count == 1, f"Expected 1 fitz.open call, got {open_count}"


# ── Per-page error isolation ─────────────────────────────────────

class TestPerPageErrorIsolation:
    def test_failing_page_does_not_crash_extraction(self, tmp_path):
        """If one page's worker raises, other pages still succeed."""
        pdf_path = _create_pdf(tmp_path, n_pages=3)

        call_count = 0

        def exploding_classifier(text, idx):
            nonlocal call_count
            call_count += 1
            if idx == 1:
                raise RuntimeError("boom on page 2")
            return f"ok-{idx}"

        # The extraction itself should not raise
        text, pages = extract_text_directly(
            pdf_path,
            page_classifier=exploding_classifier,
            return_page_results=True,
        )
        # All 3 pages should be present (page 2 with empty text from error)
        assert len(pages) == 3
        # Pages 1 and 3 should have text
        assert "Page 1" in pages[0]["text"]
        assert "Page 3" in pages[2]["text"]


# ── Config integration ───────────────────────────────────────────

class TestConfigIntegration:
    def test_ocr_dpi_read_from_config(self):
        from src.pipeline.pdf_extractor import _get_ocr_dpi
        dpi = _get_ocr_dpi()
        assert isinstance(dpi, int)
        assert 72 <= dpi <= 1200

    def test_page_timeout_read_from_config(self):
        from src.pipeline.pdf_extractor import _get_page_timeout
        timeout = _get_page_timeout()
        assert isinstance(timeout, int)
        assert timeout >= 10
