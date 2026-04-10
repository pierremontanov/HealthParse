"""Tests for rule-based entity extractors and the document classifier.

Tests each extractor against representative text samples matching the format
produced by data_generator.py, and verifies the full inference engine
round-trip via create_default_engine().
"""

import pytest

from src.pipeline.extractors.prescription_extractor import PrescriptionExtractor
from src.pipeline.extractors.result_extractor import LabResultExtractor
from src.pipeline.extractors.clinical_history_extractor import ClinicalHistoryExtractor
from src.pipeline.extractors.document_classifier import DocumentClassifier
from src.pipeline.inference import (
    InferenceEngine,
    ModelBundle,
    ModelRegistry,
    create_default_engine,
)


# ── Sample texts matching data_generator.py output ──

PRESCRIPTION_TEXT = """Patient Name: Maria Garcia
Patient ID: 45012
Date of Birth: 1985-03-15
Date of Prescription: 2025-06-10
Doctor: Dr. Carlos Rodriguez
Clinic: Central Medical Center

Prescription:
Amoxicillin 500mg, 3 times daily for 7 days.
Ibuprofen 400mg as needed for pain.
"""

LAB_RESULT_TEXT = """Patient Name: Anthony Harper
Patient ID: 32373
Date of Birth: 1965-05-12
Exam Date: 2025-01-12
Clinic: Foster-Bailey

Test Results:
- Glucose Test: 95.50 (Ref: 70-100)
- Cholesterol Test: 210.30 (Ref: 125-200)
- Hemoglobin: 14.20 (Ref: 12-17)

Summary: Cholesterol slightly elevated. Recommend dietary changes.
"""

CLINICAL_HISTORY_TEXT = """Patient Name: Shelby Brown
Patient ID: 13098
Date of Birth: 1931-09-29
Clinic: Garrett-Wagner

Annotations:
- 2025-02-22: Persistent headache reported. Prescribed analgesics.
- 2025-02-19: Follow-up visit. Blood pressure stable at 130/85.
- 2025-01-06: Initial consultation for recurring migraines and dizziness.
"""


# ═══════════════════════════════════════════════════════════════
# Prescription Extractor
# ═══════════════════════════════════════════════════════════════

class TestPrescriptionExtractor:
    def setup_method(self):
        self.extractor = PrescriptionExtractor()

    def test_extracts_patient_name(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        assert result["patient_name"] == "Maria Garcia"

    def test_extracts_patient_id(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        assert result["patient_id"] == "45012"

    def test_extracts_date(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        assert result["date"] == "2025-06-10"

    def test_extracts_doctor(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        assert result["doctor_name"] == "Dr. Carlos Rodriguez"

    def test_extracts_institution(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        assert result["institution"] == "Central Medical Center"

    def test_extracts_items(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        assert isinstance(result["items"], list)
        assert len(result["items"]) >= 1

    def test_items_have_type_and_name(self):
        result = self.extractor.extract(PRESCRIPTION_TEXT)
        for item in result["items"]:
            assert "type" in item
            assert "name" in item

    def test_handles_empty_text(self):
        result = self.extractor.extract("")
        assert result["patient_name"] is None
        assert isinstance(result["items"], list)


# ═══════════════════════════════════════════════════════════════
# Lab Result Extractor
# ═══════════════════════════════════════════════════════════════

class TestLabResultExtractor:
    def setup_method(self):
        self.extractor = LabResultExtractor()

    def test_extracts_patient_name(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert result["patient_name"] == "Anthony Harper"

    def test_extracts_patient_id(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert result["patient_id"] == "32373"

    def test_extracts_exam_date(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert result["exam_date"] == "2025-01-12"

    def test_extracts_institution(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert result["institution"] == "Foster-Bailey"

    def test_extracts_findings_with_test_results(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert "Glucose Test" in result["findings"]
        assert "Cholesterol Test" in result["findings"]
        assert "Hemoglobin" in result["findings"]

    def test_extracts_impression_from_summary(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert "Cholesterol slightly elevated" in result["impression"]

    def test_infers_exam_type(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert result["exam_type"] is not None
        assert len(result["exam_type"]) > 0

    def test_professional_has_fallback(self):
        result = self.extractor.extract(LAB_RESULT_TEXT)
        assert result["professional"] is not None

    def test_handles_empty_text(self):
        result = self.extractor.extract("")
        assert result["patient_name"] is None
        assert result["findings"] == "No findings recorded"


# ═══════════════════════════════════════════════════════════════
# Clinical History Extractor
# ═══════════════════════════════════════════════════════════════

class TestClinicalHistoryExtractor:
    def setup_method(self):
        self.extractor = ClinicalHistoryExtractor()

    def test_extracts_patient_name(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["patient_name"] == "Shelby Brown"

    def test_extracts_patient_id(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["patient_id"] == "13098"

    def test_extracts_date_of_birth(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["date_of_birth"] == "1931-09-29"

    def test_extracts_institution(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["institution"] == "Garrett-Wagner"

    def test_derives_consultation_date_from_most_recent(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["consultation_date"] == "2025-02-22"

    def test_builds_medical_history(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["medical_history"] is not None
        assert "Persistent headache" in result["medical_history"]

    def test_derives_chief_complaint(self):
        result = self.extractor.extract(CLINICAL_HISTORY_TEXT)
        assert result["chief_complaint"] is not None
        assert "headache" in result["chief_complaint"].lower()

    def test_handles_empty_text(self):
        result = self.extractor.extract("")
        assert result["patient_name"] is None
        assert result["consultation_date"] == "Unknown"


# ═══════════════════════════════════════════════════════════════
# Document Classifier
# ═══════════════════════════════════════════════════════════════

class TestDocumentClassifier:
    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_classifies_prescription(self):
        assert self.classifier.classify(PRESCRIPTION_TEXT) == "prescription"

    def test_classifies_lab_result(self):
        assert self.classifier.classify(LAB_RESULT_TEXT) == "result"

    def test_classifies_clinical_history(self):
        assert self.classifier.classify(CLINICAL_HISTORY_TEXT) == "clinical_history"

    def test_returns_none_for_empty_text(self):
        assert self.classifier.classify("") is None

    def test_predict_returns_dict(self):
        result = self.classifier.predict(LAB_RESULT_TEXT)
        assert result["document_type"] == "result"


# ═══════════════════════════════════════════════════════════════
# Integration: Full Inference Engine Round-Trip
# ═══════════════════════════════════════════════════════════════

class TestInferenceEngineIntegration:
    def setup_method(self):
        self.engine = create_default_engine()

    def test_registered_types(self):
        types = self.engine.registered_types
        assert "prescription" in types
        assert "result" in types
        assert "clinical_history" in types

    def test_process_prescription(self):
        result = self.engine.process_document("prescription", PRESCRIPTION_TEXT)
        assert result.document_type == "prescription"
        assert result.ner_output["patient_name"] == "Maria Garcia"
        assert isinstance(result.combined_output["items"], list)

    def test_process_lab_result(self):
        result = self.engine.process_document("result", LAB_RESULT_TEXT)
        assert result.document_type == "result"
        assert result.ner_output["patient_name"] == "Anthony Harper"
        assert "Glucose" in result.ner_output["findings"]

    def test_process_clinical_history(self):
        result = self.engine.process_document("clinical_history", CLINICAL_HISTORY_TEXT)
        assert result.document_type == "clinical_history"
        assert result.ner_output["patient_name"] == "Shelby Brown"
        assert result.ner_output["consultation_date"] == "2025-02-22"

    def test_unknown_document_type_raises(self):
        with pytest.raises(ValueError, match="No models registered"):
            self.engine.process_document("unknown_type", "some text")
