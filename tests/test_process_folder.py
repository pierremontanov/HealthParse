"""Tests for the process_folder ingestion module."""

import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.pipeline.process_folder import (
    DocumentResult,
    process_folder,
    _finalise_language,
    SUPPORTED_EXTENSIONS,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def fake_folder(tmp_path):
    """Create a temp folder with representative dummy files."""
    for name in ["doc1.pdf", "image1.png", "note.txt", "scan.jpg"]:
        (tmp_path / name).write_text("fake content")
    return tmp_path


@pytest.fixture
def _mock_extraction():
    """Patch all low-level extraction and detection functions."""
    with (
        patch("src.pipeline.process_folder.detect_pdf_language") as mock_pdf_lang,
        patch("src.pipeline.process_folder.is_pdf_text_based") as mock_text_based,
        patch("src.pipeline.process_folder.extract_text_directly") as mock_direct,
        patch("src.pipeline.process_folder.extract_text_from_pdf_ocr") as mock_ocr,
        patch("src.pipeline.process_folder.extract_text_from_image") as mock_img,
        patch("src.pipeline.process_folder.detect_language") as mock_lang,
    ):
        mock_pdf_lang.return_value = SimpleNamespace(
            language="en", text_sample="sample text"
        )
        mock_text_based.return_value = True
        mock_direct.return_value = "  PDF extracted text  "
        mock_ocr.return_value = "OCR text"
        mock_img.return_value = "  Image OCR text  "
        mock_lang.return_value = "en"

        yield {
            "detect_pdf_language": mock_pdf_lang,
            "is_pdf_text_based": mock_text_based,
            "extract_text_directly": mock_direct,
            "extract_text_from_pdf_ocr": mock_ocr,
            "extract_text_from_image": mock_img,
            "detect_language": mock_lang,
        }


# ── DocumentResult tests ─────────────────────────────────────────

class TestDocumentResult:
    def test_as_dict_returns_all_fields(self):
        dr = DocumentResult(file="test.pdf", status="ok", text="hello")
        d = dr.as_dict()
        assert d["file"] == "test.pdf"
        assert d["status"] == "ok"
        assert d["text"] == "hello"
        assert "document_type" in d
        assert "extracted_data" in d

    def test_default_values(self):
        dr = DocumentResult(file="x.pdf", status="ok")
        assert dr.language == "unknown"
        assert dr.validated is False
        assert dr.error is None
        assert dr.elapsed_ms == 0


# ── _finalise_language tests ──────────────────────────────────────

class TestFinaliseLanguage:
    @patch("src.pipeline.process_folder.detect_language", return_value="es")
    def test_uses_detected_when_known(self, _mock):
        assert _finalise_language("algo de texto", "en") == "es"

    @patch("src.pipeline.process_folder.detect_language", return_value="unknown")
    def test_falls_back_to_hint(self, _mock):
        assert _finalise_language("", "en") == "en"


# ── Basic ingestion (backward-compatible mode) ────────────────────

class TestProcessFolderBasic:
    def test_skips_unsupported_extensions(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        filenames = {r["file"] for r in results}
        assert "note.txt" not in filenames

    def test_processes_supported_files(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        assert len(results) == 3  # doc1.pdf, image1.png, scan.jpg

    def test_pdf_has_correct_method(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["method"] == "direct"

    def test_image_has_correct_method(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        img_result = next(r for r in results if r["file"] == "image1.png")
        assert img_result["method"] == "image"

    def test_text_is_stripped(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["text"] == "PDF extracted text"
        img_result = next(r for r in results if r["file"] == "image1.png")
        assert img_result["text"] == "Image OCR text"

    def test_language_fields_populated(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["language"] == "en"
        assert pdf_result["language_hint"] == "en"
        assert pdf_result["language_sample"] == "sample text"

    def test_all_results_have_status_ok(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        assert all(r["status"] == "ok" for r in results)

    def test_results_sorted_alphabetically(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        filenames = [r["file"] for r in results]
        assert filenames == sorted(filenames)

    def test_no_inference_fields_by_default(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        for r in results:
            assert r["document_type"] is None
            assert r["extracted_data"] is None
            assert r["validated"] is False

    def test_elapsed_ms_is_positive(self, fake_folder, _mock_extraction):
        results = process_folder(str(fake_folder))
        for r in results:
            assert r["elapsed_ms"] >= 0


# ── OCR fallback path ────────────────────────────────────────────

class TestProcessFolderOCRPath:
    def test_scanned_pdf_uses_ocr(self, fake_folder, _mock_extraction):
        _mock_extraction["is_pdf_text_based"].return_value = False
        results = process_folder(str(fake_folder))
        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["method"] == "ocr"
        assert pdf_result["text"] == "OCR text"


# ── Error resilience ──────────────────────────────────────────────

class TestProcessFolderErrorHandling:
    def test_extraction_error_does_not_crash_batch(self, fake_folder, _mock_extraction):
        _mock_extraction["extract_text_directly"].side_effect = RuntimeError("corrupt")
        results = process_folder(str(fake_folder))
        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["status"] == "extraction_error"
        assert "corrupt" in pdf_result["error"]
        # Other files still processed
        ok_results = [r for r in results if r["status"] == "ok"]
        assert len(ok_results) == 2  # image1.png, scan.jpg

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            process_folder("/nonexistent/folder/path")

    def test_empty_folder_returns_empty(self, tmp_path):
        with patch("src.pipeline.process_folder.detect_pdf_language"):
            results = process_folder(str(tmp_path))
        assert results == []


# ── Inference integration ─────────────────────────────────────────

class TestProcessFolderWithInference:
    def test_inference_classifies_and_extracts(self, fake_folder, _mock_extraction):
        mock_engine = MagicMock()
        mock_engine.classify.return_value = "prescription"
        mock_inference_result = MagicMock()
        mock_inference_result.as_dict.return_value = {"patient_name": "Test Patient"}
        mock_inference_result.validated_data = True  # truthy = validated
        mock_engine.process_document.return_value = mock_inference_result

        results = process_folder(
            str(fake_folder), run_inference=True, engine=mock_engine
        )

        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["document_type"] == "prescription"
        assert pdf_result["extracted_data"] == {"patient_name": "Test Patient"}
        assert pdf_result["validated"] is True

    def test_inference_error_captured_gracefully(self, fake_folder, _mock_extraction):
        mock_engine = MagicMock()
        mock_engine.classify.return_value = "prescription"
        mock_engine.process_document.side_effect = ValueError("bad schema")

        results = process_folder(
            str(fake_folder), run_inference=True, engine=mock_engine
        )

        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["status"] == "inference_error"
        assert "bad schema" in pdf_result["error"]

    def test_unclassifiable_document_marked_unknown(
        self, fake_folder, _mock_extraction
    ):
        mock_engine = MagicMock()
        mock_engine.classify.return_value = None

        results = process_folder(
            str(fake_folder), run_inference=True, engine=mock_engine
        )

        pdf_result = next(r for r in results if r["file"] == "doc1.pdf")
        assert pdf_result["document_type"] == "unknown"
        assert pdf_result["status"] == "ok"
        mock_engine.process_document.assert_not_called()

    def test_creates_default_engine_when_none_provided(
        self, fake_folder, _mock_extraction
    ):
        with patch(
            "src.pipeline.inference.create_default_engine"
        ) as mock_factory:
            mock_engine = MagicMock()
            mock_engine.classify.return_value = None
            mock_factory.return_value = mock_engine

            results = process_folder(str(fake_folder), run_inference=True)
            mock_factory.assert_called_once()
