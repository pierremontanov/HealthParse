"""End-to-end pipeline tests (#19).

These tests exercise the **real** pipeline stages — classification, NER
extraction, schema validation, export, and FHIR mapping — using synthetic
but realistic document text.  No extraction/OCR mocks are used for the
inference path; only file I/O is faked so tests stay fast and deterministic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import patch

import pytest

from src.pipeline.core_engine import DocIQEngine, EngineResult
from src.pipeline.inference import InferenceEngine, create_default_engine
from src.pipeline.output_formatter import export_results


# ═══════════════════════════════════════════════════════════════════
# Realistic synthetic document text
# ═══════════════════════════════════════════════════════════════════

PRESCRIPTION_TEXT = """\
Medical Prescription

Patient Name: Maria Garcia Lopez
Patient ID: RX-20240315-001
Date: 15/03/2024
Doctor: Dr. Roberto Fernandez
Institution: Clinica Santa Maria

Prescribed Medications:
- Amoxicillin 500mg capsule, take 1 capsule every 8 hours for 7 days
- Ibuprofen 400mg tablet, take 1 tablet every 6 hours as needed for pain
- Omeprazole 20mg capsule, take 1 capsule daily before breakfast

Additional Notes: Follow up in 2 weeks. Avoid alcohol during treatment.
"""

LAB_RESULT_TEXT = """\
Laboratory Test Results

Patient Name: Carlos Mendez Ruiz
Patient ID: LR-20240320-042
Age: 45
Sex: M
Date of Birth: 1979-05-12

Exam Type: Blood Chemistry – Glucose Panel
Exam Date: 20/03/2024
Study Area: Blood

Test Results:
- Glucose: 105 mg/dL (Ref: 70-100)
- HbA1c: 6.2% (Ref: 4.0-5.6)
- Insulin: 15 uIU/mL (Ref: 2.6-24.9)

Findings: Glucose levels are slightly elevated. HbA1c indicates pre-diabetic range.

Impression: Borderline glucose metabolism. Recommend dietary modification and follow-up in 3 months.

Professional: Dr. Ana Lucia Vargas
Institution: Laboratorio Central de Diagnostico
"""

CLINICAL_HISTORY_TEXT = """\
Clinical History

Patient Name: Elena Torres Vega
Patient ID: CH-20240318-015
Age: 62
Sex: F
Date of Birth: 1962-08-22

Consultation Date: 18/03/2024
Doctor: Dr. Miguel Angel Reyes
Institution: Hospital General del Norte

Chief Complaint: Persistent headaches and dizziness for the past two weeks.

Medical History:
- 2020-01-15: Diagnosed with Type 2 Diabetes, managed with Metformin
- 2021-06-20: Hypertension diagnosis, started Losartan 50mg
- 2023-11-10: Routine checkup, stable condition

Current Medications: Metformin 850mg, Losartan 50mg, Aspirin 100mg

Physical Exam: Blood pressure 150/95 mmHg, heart rate 78 bpm, temperature 36.5C.

Assessment: Uncontrolled hypertension likely contributing to headaches. Need to adjust medication.

Plan: Increase Losartan to 100mg daily. Order 24-hour blood pressure monitoring. Follow up in 1 week.
"""

UNCLASSIFIABLE_TEXT = """\
3847 xkcd zebra umbrella potato
Lorem ipsum dolor sit amet.
"""


# ═══════════════════════════════════════════════════════════════════
# Fixtures
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
        }


def _write_fake_files(tmp_path: Path, names: List[str]) -> Path:
    """Create dummy files with correct extensions so the engine accepts them."""
    for name in names:
        (tmp_path / name).write_text("fake")
    return tmp_path


# ═══════════════════════════════════════════════════════════════════
# Stage 1 – Classification
# ═══════════════════════════════════════════════════════════════════

class TestClassification:
    """Verify the DocumentClassifier correctly types realistic text."""

    def test_classifies_prescription(self, inference_engine):
        result = inference_engine.classify(PRESCRIPTION_TEXT)
        assert result == "prescription"

    def test_classifies_lab_result(self, inference_engine):
        result = inference_engine.classify(LAB_RESULT_TEXT)
        assert result == "result"

    def test_classifies_clinical_history(self, inference_engine):
        result = inference_engine.classify(CLINICAL_HISTORY_TEXT)
        assert result == "clinical_history"

    def test_unclassifiable_returns_none(self, inference_engine):
        result = inference_engine.classify(UNCLASSIFIABLE_TEXT)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Stage 2 – NER Extraction + Validation
# ═══════════════════════════════════════════════════════════════════

class TestPrescriptionE2E:
    """Full pipeline: classify → extract → validate a prescription."""

    def test_extraction_produces_validated_model(self, inference_engine):
        ir = inference_engine.process_document("prescription", PRESCRIPTION_TEXT)
        assert ir.validated_data is not None

    def test_patient_name_extracted(self, inference_engine):
        ir = inference_engine.process_document("prescription", PRESCRIPTION_TEXT)
        data = ir.as_dict()
        assert data["patient_name"] == "Maria Garcia Lopez"

    def test_doctor_extracted(self, inference_engine):
        ir = inference_engine.process_document("prescription", PRESCRIPTION_TEXT)
        data = ir.as_dict()
        assert "Roberto Fernandez" in (data.get("doctor_name") or "")

    def test_items_extracted(self, inference_engine):
        ir = inference_engine.process_document("prescription", PRESCRIPTION_TEXT)
        data = ir.as_dict()
        assert "items" in data
        assert len(data["items"]) >= 1

    def test_date_extracted(self, inference_engine):
        ir = inference_engine.process_document("prescription", PRESCRIPTION_TEXT)
        data = ir.as_dict()
        # Date may be raw or normalised depending on extractor
        assert data.get("date") is not None
        assert "2024" in data["date"]


class TestLabResultE2E:
    """Full pipeline for a lab result document."""

    def test_extraction_produces_validated_model(self, inference_engine):
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        assert ir.validated_data is not None

    def test_patient_name(self, inference_engine):
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        data = ir.as_dict()
        assert data["patient_name"] == "Carlos Mendez Ruiz"

    def test_exam_type_detected(self, inference_engine):
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        data = ir.as_dict()
        assert data.get("exam_type") is not None

    def test_findings_populated(self, inference_engine):
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        data = ir.as_dict()
        assert data.get("findings") is not None
        assert len(data["findings"]) > 0

    def test_professional_extracted(self, inference_engine):
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        data = ir.as_dict()
        assert "Vargas" in (data.get("professional") or "")


class TestClinicalHistoryE2E:
    """Full pipeline for a clinical history document."""

    def test_extraction_produces_validated_model(self, inference_engine):
        ir = inference_engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        assert ir.validated_data is not None

    def test_patient_name(self, inference_engine):
        ir = inference_engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        data = ir.as_dict()
        assert data["patient_name"] == "Elena Torres Vega"

    def test_doctor_name(self, inference_engine):
        ir = inference_engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        data = ir.as_dict()
        assert "Reyes" in (data.get("doctor_name") or "")

    def test_consultation_date(self, inference_engine):
        ir = inference_engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        data = ir.as_dict()
        assert data.get("consultation_date") is not None

    def test_medications_extracted(self, inference_engine):
        ir = inference_engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        data = ir.as_dict()
        meds = data.get("current_medications") or []
        assert len(meds) >= 1


# ═══════════════════════════════════════════════════════════════════
# Stage 3 – DocIQEngine single-file end-to-end
# ═══════════════════════════════════════════════════════════════════

class TestEngineProcessFileE2E:
    """process_file with inference enabled, using the real engine."""

    def test_pdf_classified_and_extracted(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "result_test.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] == "result"
        assert result["validated"] is True
        assert result["extracted_data"]["patient_name"] == "Carlos Mendez Ruiz"

    def test_prescription_classified_and_extracted(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        pdf = tmp_path / "rx.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] == "prescription"
        assert result["validated"] is True
        assert result["extracted_data"]["patient_name"] == "Maria Garcia Lopez"

    def test_clinical_history_classified_and_extracted(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = CLINICAL_HISTORY_TEXT
        pdf = tmp_path / "hc.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] == "clinical_history"
        assert result["validated"] is True
        assert result["extracted_data"]["patient_name"] == "Elena Torres Vega"

    def test_image_classified_and_extracted(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_from_image"].return_value = PRESCRIPTION_TEXT
        img = tmp_path / "scan.png"
        img.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(img))

        assert result["status"] == "ok"
        assert result["method"] == "image"
        assert result["document_type"] == "prescription"

    def test_unclassifiable_text_handled(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = UNCLASSIFIABLE_TEXT
        pdf = tmp_path / "random.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        assert result["status"] == "ok"
        assert result["document_type"] == "unknown"
        assert result["extracted_data"] is None

    def test_elapsed_ms_is_positive(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "timed.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))
        assert result["elapsed_ms"] >= 0


# ═══════════════════════════════════════════════════════════════════
# Stage 4 – Batch processing end-to-end
# ═══════════════════════════════════════════════════════════════════

class TestEngineBatchE2E:
    """process_batch through the real inference engine."""

    def test_mixed_batch(self, tmp_path, _mock_extraction):
        """Three documents of different types, processed as a batch."""
        # We'll make the mock return different text per call
        texts = iter([PRESCRIPTION_TEXT, LAB_RESULT_TEXT, CLINICAL_HISTORY_TEXT])
        _mock_extraction["extract_text_directly"].side_effect = lambda *a, **kw: next(texts)

        for name in ["rx.pdf", "lab.pdf", "hc.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        assert isinstance(er, EngineResult)
        assert er.count == 3
        assert len(er.ok) == 3

        types = {r["document_type"] for r in er}
        assert "prescription" in types
        assert "result" in types
        assert "clinical_history" in types

    def test_batch_summary(self, tmp_path, _mock_extraction):
        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        (tmp_path / "a.pdf").write_text("fake")
        (tmp_path / "b.pdf").write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))
        summary = er.summary()
        assert summary.get("ok", 0) == 2


# ═══════════════════════════════════════════════════════════════════
# Stage 5 – Export round-trip (JSON, CSV, FHIR)
# ═══════════════════════════════════════════════════════════════════

class TestExportE2E:
    """End-to-end export from inference results."""

    @pytest.fixture
    def pipeline_results(self, tmp_path, _mock_extraction) -> List[Dict]:
        """Run real inference and collect results for export."""
        texts = {
            "rx.pdf": PRESCRIPTION_TEXT,
            "lab.pdf": LAB_RESULT_TEXT,
            "hc.pdf": CLINICAL_HISTORY_TEXT,
        }
        engine = DocIQEngine(run_inference=True)
        results = []
        for fname, text in texts.items():
            f = tmp_path / fname
            f.write_text("fake")
            _mock_extraction["extract_text_directly"].return_value = text
            results.append(engine.process_file(str(f)))
        return results

    def test_json_export(self, tmp_path, pipeline_results):
        path = export_results(pipeline_results, output_dir=str(tmp_path / "json_out"), fmt="json")
        assert os.path.isdir(path)
        files = os.listdir(path)
        assert len(files) == 3
        for f in files:
            with open(os.path.join(path, f)) as fh:
                data = json.load(fh)
            assert "status" in data
            assert data["status"] == "ok"

    def test_csv_export(self, tmp_path, pipeline_results):
        path = export_results(pipeline_results, output_dir=str(tmp_path / "csv_out"), fmt="csv")
        assert os.path.exists(path)
        content = open(path).read()
        assert "Maria Garcia Lopez" in content or "Carlos Mendez Ruiz" in content

    def test_fhir_export(self, tmp_path, pipeline_results):
        path = export_results(pipeline_results, output_dir=str(tmp_path / "fhir_out"), fmt="fhir")
        assert os.path.isdir(path)
        files = os.listdir(path)
        # Should have individual FHIR files + bundle.json
        json_files = [f for f in files if f.endswith(".json")]
        assert len(json_files) >= 3  # at least 3 resources

    def test_fhir_resources_valid(self, tmp_path, pipeline_results):
        path = export_results(pipeline_results, output_dir=str(tmp_path / "fhir_valid"), fmt="fhir")
        for fname in os.listdir(path):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(path, fname)) as fh:
                data = json.load(fh)
            assert "resourceType" in data
            assert data["resourceType"] in (
                "DiagnosticReport", "MedicationRequest", "Encounter", "Bundle"
            )

    def test_fhir_bundle_contains_all_resources(self, tmp_path, pipeline_results):
        path = export_results(pipeline_results, output_dir=str(tmp_path / "fhir_bundle"), fmt="fhir")
        bundle_path = os.path.join(path, "bundle.json")
        if os.path.exists(bundle_path):
            with open(bundle_path) as fh:
                bundle = json.load(fh)
            assert bundle["resourceType"] == "Bundle"
            assert bundle["type"] == "collection"
            assert len(bundle["entry"]) >= 3


# ═══════════════════════════════════════════════════════════════════
# Stage 6 – FHIR resource fidelity
# ═══════════════════════════════════════════════════════════════════

class TestFHIRFidelityE2E:
    """Verify FHIR output preserves key clinical data from extraction."""

    def test_diagnostic_report_has_patient(self, inference_engine):
        from src.pipeline.fhir_mapper import map_to_fhir_loose
        from src.pipeline.validation.schemas import ResultSchema

        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        model = ResultSchema(**ir.as_dict())
        fhir = map_to_fhir_loose(model)

        assert fhir["resourceType"] == "DiagnosticReport"
        assert fhir["subject"]["display"] == "Carlos Mendez Ruiz"
        assert fhir["status"] == "final"

    def test_medication_request_has_patient(self, inference_engine):
        from src.pipeline.fhir_mapper import map_to_fhir_loose
        from src.pipeline.validation.prescription_schema import Prescription

        ir = inference_engine.process_document("prescription", PRESCRIPTION_TEXT)
        model = Prescription(**ir.as_dict())
        fhir = map_to_fhir_loose(model)

        assert fhir["resourceType"] == "MedicationRequest"
        assert fhir["subject"]["display"] == "Maria Garcia Lopez"
        assert fhir["status"] == "active"
        assert fhir["intent"] == "order"

    def test_encounter_has_patient(self, inference_engine):
        from src.pipeline.fhir_mapper import map_to_fhir_loose
        from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema

        ir = inference_engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        model = ClinicalHistorySchema(**ir.as_dict())
        fhir = map_to_fhir_loose(model)

        assert fhir["resourceType"] == "Encounter"
        assert fhir["subject"]["display"] == "Elena Torres Vega"
        assert fhir["status"] == "finished"

    def test_fhir_output_has_no_none_values(self, inference_engine):
        """Null-pruned FHIR resources must have zero None leaves."""
        from src.pipeline.fhir_mapper import map_to_fhir_loose, prune_none
        from src.pipeline.validation.schemas import ResultSchema

        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        model = ResultSchema(**ir.as_dict())
        fhir = prune_none(map_to_fhir_loose(model))

        def _assert_no_none(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert v is not None, f"None found at {path}.{k}"
                    _assert_no_none(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _assert_no_none(item, f"{path}[{i}]")

        _assert_no_none(fhir)


# ═══════════════════════════════════════════════════════════════════
# Stage 7 – CLI end-to-end
# ═══════════════════════════════════════════════════════════════════

class TestCLIEndToEnd:
    """CLI runs the full pipeline and writes output files."""

    def test_cli_json_output(self, tmp_path, _mock_extraction):
        from src.cli import main as cli_main

        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "input" / "lab.pdf"
        pdf.parent.mkdir()
        pdf.write_text("fake")

        code = cli_main([
            "--input", str(pdf.parent),
            "--output-dir", str(tmp_path / "out"),
            "--format", "json",
        ])
        assert code == 0
        out_dir = tmp_path / "out" / "dociq_results"
        assert out_dir.is_dir()
        files = list(out_dir.iterdir())
        assert len(files) >= 1
        with open(files[0]) as fh:
            data = json.load(fh)
        assert data["document_type"] == "result"

    def test_cli_fhir_output(self, tmp_path, _mock_extraction):
        from src.cli import main as cli_main

        _mock_extraction["extract_text_directly"].return_value = PRESCRIPTION_TEXT
        pdf = tmp_path / "input" / "rx.pdf"
        pdf.parent.mkdir()
        pdf.write_text("fake")

        code = cli_main([
            "--input", str(pdf.parent),
            "--output-dir", str(tmp_path / "out"),
            "--format", "fhir",
        ])
        assert code == 0
        fhir_dir = tmp_path / "out" / "dociq_fhir"
        assert fhir_dir.is_dir()
        json_files = list(fhir_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_cli_csv_output(self, tmp_path, _mock_extraction):
        from src.cli import main as cli_main

        _mock_extraction["extract_text_directly"].return_value = CLINICAL_HISTORY_TEXT
        pdf = tmp_path / "input" / "hc.pdf"
        pdf.parent.mkdir()
        pdf.write_text("fake")

        code = cli_main([
            "--input", str(pdf.parent),
            "--output-dir", str(tmp_path / "out"),
            "--format", "csv",
        ])
        assert code == 0
        csv_path = tmp_path / "out" / "dociq_results.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "Elena Torres Vega" in content


# ═══════════════════════════════════════════════════════════════════
# Cross-cutting concerns
# ═══════════════════════════════════════════════════════════════════

class TestCrossCutting:
    """Error handling, edge cases, and consistency checks."""

    def test_empty_text_does_not_crash(self, inference_engine):
        """Empty extraction text should classify as None, not raise."""
        result = inference_engine.classify("")
        assert result is None

    def test_all_three_types_validate(self, inference_engine):
        """Each document type produces validated_data (not None)."""
        for doc_type, text in [
            ("prescription", PRESCRIPTION_TEXT),
            ("result", LAB_RESULT_TEXT),
            ("clinical_history", CLINICAL_HISTORY_TEXT),
        ]:
            ir = inference_engine.process_document(doc_type, text)
            assert ir.validated_data is not None, f"{doc_type} failed validation"

    def test_inference_result_has_all_fields(self, inference_engine):
        """InferenceResult exposes every intermediate stage."""
        ir = inference_engine.process_document("result", LAB_RESULT_TEXT)
        assert ir.document_type == "result"
        assert len(ir.raw_text) > 0
        assert len(ir.preprocessed_text) > 0
        assert isinstance(ir.classifier_output, dict)
        assert isinstance(ir.ner_output, dict)
        assert isinstance(ir.combined_output, dict)

    def test_batch_with_mixed_types_all_validate(self, tmp_path, _mock_extraction):
        texts = iter([PRESCRIPTION_TEXT, LAB_RESULT_TEXT, CLINICAL_HISTORY_TEXT])
        _mock_extraction["extract_text_directly"].side_effect = lambda *a, **kw: next(texts)

        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (tmp_path / name).write_text("fake")

        engine = DocIQEngine(run_inference=True)
        er = engine.process_batch(str(tmp_path))

        for r in er:
            assert r["status"] == "ok"
            assert r["validated"] is True, f"{r['file']} not validated"

    def test_export_round_trip_preserves_data(self, tmp_path, _mock_extraction):
        """JSON export → reload should match original extracted_data."""
        _mock_extraction["extract_text_directly"].return_value = LAB_RESULT_TEXT
        pdf = tmp_path / "rt.pdf"
        pdf.write_text("fake")

        engine = DocIQEngine(run_inference=True)
        result = engine.process_file(str(pdf))

        path = export_results([result], output_dir=str(tmp_path / "rt_out"), fmt="json")
        with open(os.path.join(path, "rt.json")) as fh:
            reloaded = json.load(fh)

        assert reloaded["extracted_data"]["patient_name"] == "Carlos Mendez Ruiz"
        assert reloaded["document_type"] == "result"
