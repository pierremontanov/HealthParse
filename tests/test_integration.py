"""Full pipeline integration tests (NEW-6).

Exercises cross-cutting integration paths that span multiple pipeline
stages and verify correct behaviour at module boundaries.  Complements
the existing ``test_e2e_pipeline.py`` (which covers happy-path flows)
with error propagation, metrics integration, API endpoint testing,
validation failure handling, engine configuration modes, and export
edge cases.

Every test uses the **real** inference engine and validators — only
file-level I/O is mocked so the suite stays fast and deterministic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.core_engine import DocIQEngine, EngineResult
from src.pipeline.exceptions import (
    DocIQError,
    DocumentExtractionError,
    ExportError,
    UnsupportedFileError,
)
from src.pipeline.inference import InferenceEngine, InferenceResult, create_default_engine
from src.pipeline.output_formatter import export_results


# ═══════════════════════════════════════════════════════════════════
# Shared text fixtures
# ═══════════════════════════════════════════════════════════════════

PRESCRIPTION_TEXT = """\
Medical Prescription

Patient Name: Ana Sofia Vega
Patient ID: RX-001
Date of Prescription: 2025-04-10
Doctor: Dr. Luis Morales
Clinic: Clinica San Rafael

Prescription:
Metformin 500mg, take 1 tablet twice daily with meals
Lisinopril 10mg, take 1 tablet daily in the morning
"""

LAB_RESULT_TEXT = """\
Laboratory Test Results

Patient Name: Diego Ramirez
Patient ID: LR-042
Date of Birth: 1980-07-15
Exam Date: 2025-04-12
Clinic: Lab Central

Test Results:
- Glucose: 110 mg/dL (Ref: 70-100)
- Cholesterol: 195 mg/dL (Ref: 125-200)

Summary: Glucose slightly elevated.
"""

CLINICAL_HISTORY_TEXT = """\
Clinical History

Patient Name: Rosa Martinez
Patient ID: CH-015
Date of Birth: 1955-11-20
Clinic: Hospital del Sur
Doctor: Dr. Carmen Diaz

Annotations:
- 2025-03-10: Follow-up visit. Blood pressure 140/90 mmHg.
- 2025-02-01: Initial consultation for recurrent headaches.
"""

GIBBERISH_TEXT = """\
8374 xkcd random potato umbrella
lorem ipsum dolor sit amet.
"""


# ═══════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def inference_engine() -> InferenceEngine:
    """Real inference engine with built-in rule-based extractors."""
    return create_default_engine()


@pytest.fixture
def _mock_extraction():
    """Patch low-level file I/O so we can feed text directly."""
    with (
        patch("src.pipeline.process_folder.detect_pdf_language") as m_pdf_lang,
        patch("src.pipeline.process_folder.is_pdf_text_based") as m_text,
        patch("src.pipeline.process_folder.extract_text_directly") as m_direct,
        patch("src.pipeline.process_folder.extract_text_from_pdf_ocr") as m_ocr,
        patch("src.pipeline.process_folder.extract_text_from_image") as m_img,
        patch("src.pipeline.process_folder.detect_language") as m_lang,
    ):
        m_pdf_lang.return_value = SimpleNamespace(language="en", text_sample="sample")
        m_text.return_value = True
        m_direct.return_value = "placeholder"
        m_ocr.return_value = "placeholder"
        m_img.return_value = "placeholder"
        m_lang.return_value = "en"
        yield {
            "extract_text_directly": m_direct,
            "extract_text_from_image": m_img,
            "is_pdf_text_based": m_text,
            "extract_text_from_pdf_ocr": m_ocr,
        }


# ═══════════════════════════════════════════════════════════════════
# 1. Error propagation through the engine
# ═══════════════════════════════════════════════════════════════════

class TestErrorPropagation:
    """Verify errors at each stage surface correctly through the engine."""

    def test_file_not_found_raises(self):
        engine = DocIQEngine(run_inference=True)
        with pytest.raises(FileNotFoundError):
            engine.process_file("/nonexistent/document.pdf")

    def test_unsupported_extension_raises(self, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        engine = DocIQEngine(run_inference=True)
        with pytest.raises(UnsupportedFileError):
            engine.process_file(str(txt))

    def test_extraction_error_raises_document_extraction_error(self, tmp_path, _mock_extraction):
        """When extraction itself throws, the engine wraps it."""
        _mock_extraction["extract_text_directly"].side_effect = RuntimeError("corrupt PDF")
        pdf = tmp_path / "bad.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        with pytest.raises(DocumentExtractionError, match="corrupt PDF"):
            engine.process_file(str(pdf))

    def test_inference_error_captured_in_result(self, tmp_path, _mock_extraction):
        """When NER/classification raises, status becomes inference_error."""
        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        pdf = tmp_path / "ner_fail.pdf"
        pdf.write_text("fake")

        # Create an engine with a broken NER extractor
        broken_engine = create_default_engine()
        # Monkey-patch the process_document to raise
        original = broken_engine.process_document

        def _raise_on_process(doc_type, text):
            raise RuntimeError("NER model crashed")

        broken_engine.process_document = _raise_on_process

        engine = DocIQEngine(inference_engine=broken_engine, run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "inference_error"
        assert "NER model crashed" in result["error"]

    def test_unsupported_export_format_raises(self, tmp_path):
        items = [{"file": "test.pdf", "status": "ok", "extracted_data": None}]
        with pytest.raises(ExportError):
            export_results(items, output_dir=str(tmp_path), fmt="xml")

    def test_batch_folder_not_found_raises(self):
        engine = DocIQEngine(run_inference=True)
        with pytest.raises(FileNotFoundError):
            engine.process_batch("/nonexistent/folder/")


# ═══════════════════════════════════════════════════════════════════
# 2. Engine configuration modes
# ═══════════════════════════════════════════════════════════════════

class TestEngineConfigModes:
    """Verify different engine configurations produce correct output."""

    def test_inference_disabled_skips_classification(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "no_inf.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=False)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] is None
        assert result["extracted_data"] is None
        assert result["validated"] is False
        assert len(result["text"]) > 0

    def test_inference_disabled_batch_no_extraction(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        for name in ["a.pdf", "b.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=False)
        er = engine.process_batch(str(tmp_path))

        assert er.count == 2
        for r in er:
            assert r["document_type"] is None
            assert r["text"].strip() != ""

    def test_custom_engine_instance(self, tmp_path, _mock_extraction):
        """Can pass a pre-built inference engine to DocIQEngine."""
        custom = create_default_engine()
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "custom.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(inference_engine=custom, run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["document_type"] == "result"
        assert result["validated"] is True

    def test_empty_text_skips_inference(self, tmp_path, _mock_extraction):
        """When extraction returns empty text, inference is skipped."""
        _mock_extraction["extract_text_directly"].return_value = ""
        pdf = tmp_path / "empty.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] is None
        assert result["extracted_data"] is None


# ═══════════════════════════════════════════════════════════════════
# 3. Unclassifiable document handling
# ═══════════════════════════════════════════════════════════════════

class TestUnclassifiableDocuments:
    """Verify the pipeline handles documents it cannot classify."""

    def test_gibberish_text_returns_unknown(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = GIBBERISH_TEXT
        pdf = tmp_path / "gibberish.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] == "unknown"
        assert result["extracted_data"] is None

    def test_gibberish_in_batch_does_not_block_others(self, tmp_path, _mock_extraction):
        texts = iter([GIBBERISH_TEXT, LAB_RESULT_TEXT])
        _mock_extraction["extract_text_directly"].side_effect = lambda *a, **kw: next(texts)

        for name in ["junk.pdf", "lab.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        assert er.count == 2
        types = {r["document_type"] for r in er}
        assert "unknown" in types
        assert "result" in types
        # The valid one should be fully extracted
        valid = [r for r in er if r["document_type"] == "result"]
        assert valid[0]["validated"] is True


# ═══════════════════════════════════════════════════════════════════
# 4. Metrics integration
# ═══════════════════════════════════════════════════════════════════

class TestMetricsIntegration:
    """Verify metrics are recorded during pipeline processing."""

    def test_process_file_records_metrics(self, tmp_path, _mock_extraction):
        from src.pipeline.metrics import get_collector

        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "metrics.pdf"
        pdf.write_text("fake")

        collector = get_collector()
        collector.clear()

        engine = DocIQEngine(run_inference=True)
        engine.process_file(str(pdf))

        summary = collector.summary()
        assert "extraction" in summary
        assert "process_file" in summary
        assert summary["extraction"]["count"] >= 1
        assert summary["process_file"]["count"] >= 1

    def test_elapsed_ms_is_positive_and_consistent(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        pdf = tmp_path / "timed.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["elapsed_ms"] >= 0
        assert isinstance(result["elapsed_ms"], int)


# ═══════════════════════════════════════════════════════════════════
# 5. Export edge cases
# ═══════════════════════════════════════════════════════════════════

class TestExportEdgeCases:
    """Export functions handle boundary conditions."""

    def test_export_empty_results_json(self, tmp_path):
        path = export_results([], output_dir=str(tmp_path / "empty_json"), fmt="json")
        assert os.path.isdir(path)
        assert len(os.listdir(path)) == 0

    def test_export_empty_results_csv(self, tmp_path):
        path = export_results([], output_dir=str(tmp_path / "empty_csv"), fmt="csv")
        assert os.path.exists(path)

    def test_export_empty_results_fhir(self, tmp_path):
        path = export_results([], output_dir=str(tmp_path / "empty_fhir"), fmt="fhir")
        assert os.path.isdir(path)

    def test_export_with_custom_filename_json(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        path = export_results(
            [result], output_dir=str(tmp_path / "custom"), fmt="json",
            filename="my_results",
        )
        assert "my_results" in path

    def test_export_with_validation_disabled(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "noval.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        path = export_results(
            [result], output_dir=str(tmp_path / "noval"), fmt="json",
            validate=False,
        )
        assert os.path.isdir(path)
        files = os.listdir(path)
        assert len(files) == 1


# ═══════════════════════════════════════════════════════════════════
# 6. Data integrity across stages
# ═══════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Verify data flows correctly across pipeline stages."""

    def test_patient_name_preserved_through_all_stages(self, tmp_path, _mock_extraction):
        """Patient name from extraction matches what lands in JSON export."""
        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        pdf = tmp_path / "integrity.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        # Stage 1: engine result
        assert result["extracted_data"]["patient_name"] == "Ana Sofia Vega"

        # Stage 2: JSON export round-trip
        path = export_results([result], output_dir=str(tmp_path / "int_out"), fmt="json")
        with open(os.path.join(path, "integrity.json")) as fh:
            reloaded = json.load(fh)
        assert reloaded["extracted_data"]["patient_name"] == "Ana Sofia Vega"

    def test_all_doc_types_preserve_patient_info(self, tmp_path, _mock_extraction):
        """Each document type preserves its patient name through the pipeline."""
        cases = [
            ("rx.pdf", PRESCRIPTION_TEXT, "Ana Sofia Vega"),
            ("lab.pdf", LAB_RESULT_TEXT, "Diego Ramirez"),
            ("hc.pdf", CLINICAL_HISTORY_TEXT, "Rosa Martinez"),
        ]
        engine = DocIQEngine(run_inference=True)

        for fname, text, expected_name in cases:
            _mock_extraction["extract_text_directly"].return_value = text
            pdf = tmp_path / fname
            pdf.write_text("fake")
            result = engine.process_file(str(pdf))

            assert result["extracted_data"]["patient_name"] == expected_name, (
                f"Patient name mismatch for {fname}"
            )

    def test_inference_result_stages_are_populated(self, inference_engine):
        """InferenceResult exposes every intermediate stage for all types."""
        for doc_type, text in [
            ("prescription", PRESCRIPTION_TEXT),
            ("result", LAB_RESULT_TEXT),
            ("clinical_history", CLINICAL_HISTORY_TEXT),
        ]:
            ir = inference_engine.process_document(doc_type, text)
            assert ir.document_type == doc_type
            assert len(ir.raw_text) > 0
            assert len(ir.preprocessed_text) > 0
            assert isinstance(ir.classifier_output, dict)
            assert isinstance(ir.ner_output, dict)
            assert isinstance(ir.combined_output, dict)
            assert ir.validated_data is not None

    def test_validated_data_matches_combined_output_keys(self, inference_engine):
        """Validated model should cover the key fields from combined output."""
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        validated_dict = ir.as_dict()

        # Key fields from the combined output must be in the validated result
        assert "patient_name" in validated_dict
        assert "exam_date" in validated_dict or "exam_type" in validated_dict


# ═══════════════════════════════════════════════════════════════════
# 7. Image file processing integration
# ═══════════════════════════════════════════════════════════════════

class TestImageProcessing:
    """Verify image files go through the OCR path correctly."""

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg"])
    def test_image_file_uses_image_method(self, tmp_path, _mock_extraction, ext):
        _mock_extraction["extract_text_from_image"].return_value = LAB_RESULT_TEXT
        img = tmp_path / f"scan{ext}"
        img.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(img))

        assert result["method"] == "image"
        assert result["status"] == "ok"
        assert result["document_type"] == "result"

    def test_ocr_pdf_uses_ocr_method(self, tmp_path, _mock_extraction):
        """Non-text-based PDFs use OCR extraction."""
        _mock_extraction["is_pdf_text_based"].return_value = False
        _mock_extraction["extract_text_from_pdf_ocr"].return_value = PRESCRIPTION_TEXT
        pdf = tmp_path / "scanned.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["method"] == "ocr"
        assert result["status"] == "ok"
        assert result["document_type"] == "prescription"


# ═══════════════════════════════════════════════════════════════════
# 8. EngineResult wrapper
# ═══════════════════════════════════════════════════════════════════

class TestEngineResult:
    """Verify the EngineResult convenience wrapper."""

    def test_ok_and_errors_partitioning(self, tmp_path, _mock_extraction):
        texts = iter([PRESCRIPTION_TEXT, GIBBERISH_TEXT])
        _mock_extraction["extract_text_directly"].side_effect = lambda *a, **kw: next(texts)

        for name in ["rx.pdf", "junk.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        assert er.count == 2
        assert len(er.ok) == 2  # both are "ok" status (gibberish is ok but unknown)
        assert len(er.errors) == 0

    def test_summary_counts(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        summary = er.summary()
        assert summary["ok"] == 3

    def test_iteration_and_indexing(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        (tmp_path / "single.pdf").write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        assert len(er) == 1
        assert er[0]["status"] == "ok"
        for r in er:
            assert "file" in r

    def test_repr(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        (tmp_path / "repr.pdf").write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        r = repr(er)
        assert "EngineResult" in r
        assert "ok=" in r


# ═══════════════════════════════════════════════════════════════════
# 9. API endpoint integration (FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Integration tests for the FastAPI application."""

    @pytest.fixture(autouse=True)
    def _setup_client(self, _mock_extraction):
        from fastapi.testclient import TestClient
        from src.api.app import app

        self.client = TestClient(app)
        self._mocks = _mock_extraction

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_process_pdf_returns_200(self):
        self._mocks["extract_text_directly"].return_value = LAB_RESULT_TEXT
        resp = self.client.post(
            "/process",
            files=[("files", ("lab.pdf", b"fake-pdf", "application/pdf"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        result = data["results"][0]
        assert result["status"] == "ok"
        assert result["document_type"] == "result"
        assert result["extracted_data"]["patient_name"] == "Diego Ramirez"

    def test_process_image_returns_200(self):
        self._mocks["extract_text_from_image"].return_value = PRESCRIPTION_TEXT
        resp = self.client.post(
            "/process",
            files=[("files", ("scan.png", b"fake-png", "image/png"))],
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["document_type"] == "prescription"

    def test_process_unsupported_file_type(self):
        resp = self.client.post(
            "/process",
            files=[("files", ("readme.txt", b"hello", "text/plain"))],
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["status"] == "unsupported_type"

    def test_process_fhir_format(self):
        self._mocks["extract_text_directly"].return_value = LAB_RESULT_TEXT
        resp = self.client.post(
            "/process?format=fhir",
            files=[("files", ("lab.pdf", b"fake-pdf", "application/pdf"))],
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["status"] == "ok"
        # FHIR format replaces extracted_data with FHIR resource
        fhir = result["extracted_data"]
        assert fhir is not None
        assert "resourceType" in fhir

    def test_process_multiple_files(self):
        texts = iter([PRESCRIPTION_TEXT, LAB_RESULT_TEXT])
        self._mocks["extract_text_directly"].side_effect = lambda *a, **kw: next(texts)

        resp = self.client.post(
            "/process",
            files=[
                ("files", ("rx.pdf", b"fake-pdf", "application/pdf")),
                ("files", ("lab.pdf", b"fake-pdf", "application/pdf")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        types = {r["document_type"] for r in data["results"]}
        assert "prescription" in types
        assert "result" in types

    def test_no_files_returns_400(self):
        resp = self.client.post("/process")
        assert resp.status_code in (400, 422)


# ═══════════════════════════════════════════════════════════════════
# 10. Cross-format export consistency
# ═══════════════════════════════════════════════════════════════════

class TestCrossFormatConsistency:
    """All export formats should preserve the same core data."""

    @pytest.fixture
    def pipeline_result(self, tmp_path, _mock_extraction) -> Dict[str, Any]:
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "consistency.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        return engine.process_file(str(pdf))

    def test_json_preserves_patient_name(self, tmp_path, pipeline_result):
        path = export_results(
            [pipeline_result], output_dir=str(tmp_path / "j"), fmt="json"
        )
        with open(os.path.join(path, "consistency.json")) as fh:
            data = json.load(fh)
        assert data["extracted_data"]["patient_name"] == "Diego Ramirez"

    def test_csv_contains_patient_name(self, tmp_path, pipeline_result):
        path = export_results(
            [pipeline_result], output_dir=str(tmp_path / "c"), fmt="csv"
        )
        content = open(path).read()
        assert "Diego Ramirez" in content

    def test_fhir_preserves_patient_name(self, tmp_path, pipeline_result):
        path = export_results(
            [pipeline_result], output_dir=str(tmp_path / "f"), fmt="fhir"
        )
        # Find the individual FHIR file (not bundle)
        fhir_files = [
            f for f in os.listdir(path)
            if f.endswith(".json") and f != "bundle.json"
        ]
        assert len(fhir_files) >= 1
        with open(os.path.join(path, fhir_files[0])) as fh:
            fhir = json.load(fh)
        assert fhir["subject"]["display"] == "Diego Ramirez"


# ═══════════════════════════════════════════════════════════════════
# 11. Batch with extraction failure in one file
# ═══════════════════════════════════════════════════════════════════

class TestBatchPartialFailure:
    """Batch processing should continue when one file fails extraction."""

    def test_one_extraction_failure_others_succeed(self, tmp_path, _mock_extraction):
        call_count = {"n": 0}
        def _side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("corrupt file")
            return LAB_RESULT_TEXT

        _mock_extraction["extract_text_directly"].side_effect = _side_effect

        for name in ["bad.pdf", "good.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        # Use process_batch from process_folder (which catches extraction errors)
        from src.pipeline.process_folder import process_folder
        results = process_folder(
            str(tmp_path), run_inference=True, engine=engine._engine
        )

        statuses = {r["status"] for r in results}
        assert "extraction_error" in statuses
        assert "ok" in statuses
        assert len(results) == 2

    def test_export_handles_mixed_status_results(self, tmp_path):
        """Export should handle results with errors gracefully."""
        items = [
            {
                "file": "good.pdf",
                "status": "ok",
                "document_type": "result",
                "extracted_data": {"patient_name": "Test"},
                "validated": True,
                "text": "sample",
                "language": "en",
                "method": "direct",
                "elapsed_ms": 100,
            },
            {
                "file": "bad.pdf",
                "status": "extraction_error",
                "document_type": None,
                "extracted_data": None,
                "validated": False,
                "text": "",
                "language": "unknown",
                "method": "",
                "error": "corrupt",
                "elapsed_ms": 0,
            },
        ]
        path = export_results(
            items, output_dir=str(tmp_path / "mixed"), fmt="json"
        )
        files = os.listdir(path)
        assert len(files) == 2

        # The error result should still be exported
        with open(os.path.join(path, "bad.json")) as fh:
            data = json.load(fh)
        assert data["status"] == "extraction_error"


# ═══════════════════════════════════════════════════════════════════
# 12. Empty batch and edge cases
# ═══════════════════════════════════════════════════════════════════

class TestBatchEdgeCases:
    """Edge cases for batch processing."""

    def test_empty_folder_returns_empty(self, tmp_path, _mock_extraction):
        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))
        assert er.count == 0
        assert len(er.ok) == 0

    def test_folder_with_only_unsupported_files(self, tmp_path, _mock_extraction):
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "notes.md").write_text("# Notes")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))
        assert er.count == 0

    def test_batch_export_round_trip(self, tmp_path, _mock_extraction):
        """Batch → export → reload should preserve all results."""
        texts = iter([PRESCRIPTION_TEXT, LAB_RESULT_TEXT, CLINICAL_HISTORY_TEXT])
        _mock_extraction["extract_text_directly"].side_effect = lambda *a, **kw: next(texts)

        for name in ["rx.pdf", "lab.pdf", "hc.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        path = export_results(er.all, output_dir=str(tmp_path / "batch_out"), fmt="json")
        json_files = sorted(os.listdir(path))
        assert len(json_files) == 3

        reloaded_names = set()
        for f in json_files:
            with open(os.path.join(path, f)) as fh:
                data = json.load(fh)
            assert data["status"] == "ok"
            reloaded_names.add(data["extracted_data"]["patient_name"])

        assert "Ana Sofia Vega" in reloaded_names
        assert "Diego Ramirez" in reloaded_names
        assert "Rosa Martinez" in reloaded_names
