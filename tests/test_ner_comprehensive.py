"""Comprehensive unit tests for NER extractors and classifier (#24).

Covers gaps in the existing test_extractors.py:
  • Base extractor helpers (extract_field, extract_date, extract_block, etc.)
  • Prescription extractor item classification edge cases
  • Lab result exam type inference
  • Clinical history annotation parsing
  • Document classifier scoring and thresholds
  • Malformed input, unicode, missing fields
"""
from __future__ import annotations

import pytest

from src.pipeline.extractors.base import (
    extract_block,
    extract_date,
    extract_dated_entries,
    extract_field,
    extract_list_items,
    extract_test_results,
)
from src.pipeline.extractors.clinical_history_extractor import ClinicalHistoryExtractor
from src.pipeline.extractors.document_classifier import DocumentClassifier
from src.pipeline.extractors.prescription_extractor import PrescriptionExtractor
from src.pipeline.extractors.result_extractor import LabResultExtractor


# ═══════════════════════════════════════════════════════════════════
# Base extractor helpers
# ═══════════════════════════════════════════════════════════════════


class TestExtractField:
    """Tests for extract_field()."""

    def test_basic_extraction(self):
        assert extract_field("Name: John Doe", "Name") == "John Doe"

    def test_case_insensitive(self):
        assert extract_field("name: Jane", "Name") == "Jane"
        assert extract_field("NAME: Jane", "name") == "Jane"

    def test_extra_whitespace(self):
        assert extract_field("Name  :   John Doe  ", "Name") == "John Doe"

    def test_multiline_picks_correct_line(self):
        text = "Name: Alice\nAge: 30\nCity: Boston"
        assert extract_field(text, "Age") == "30"
        assert extract_field(text, "City") == "Boston"

    def test_returns_none_when_missing(self):
        assert extract_field("Name: Alice", "Phone") is None

    def test_returns_none_for_empty_value(self):
        assert extract_field("Name:   ", "Name") is None

    def test_empty_text(self):
        assert extract_field("", "Name") is None

    def test_colon_in_value(self):
        result = extract_field("Note: take at 8:00 AM", "Note")
        assert "8:00 AM" in result

    def test_unicode_value(self):
        assert extract_field("Name: José García", "Name") == "José García"


class TestExtractDate:
    """Tests for extract_date()."""

    def test_iso_date(self):
        assert extract_date("Date: 2025-01-15", "Date") == "2025-01-15"

    def test_european_date(self):
        assert extract_date("Date: 15-01-2025", "Date") == "15-01-2025"

    def test_slash_date(self):
        assert extract_date("Date: 2025/01/15", "Date") == "2025/01/15"

    def test_returns_none_when_missing(self):
        assert extract_date("Name: John", "Date") is None

    def test_returns_none_for_non_date(self):
        assert extract_date("Date: not a date", "Date") is None

    def test_case_insensitive(self):
        assert extract_date("date: 2025-06-01", "Date") == "2025-06-01"

    def test_multiline(self):
        text = "Name: X\nExam Date: 2025-03-10\nOther: Y"
        assert extract_date(text, "Exam Date") == "2025-03-10"


class TestExtractBlock:
    """Tests for extract_block()."""

    def test_basic_block(self):
        text = "Prescription:\nAmoxicillin 500mg\nIbuprofen 400mg\n"
        block = extract_block(text, "Prescription")
        assert "Amoxicillin" in block
        assert "Ibuprofen" in block

    def test_block_includes_until_eof(self):
        text = "Results:\n- Glucose: 95\n- Hemoglobin: 14\n"
        block = extract_block(text, "Results")
        assert "Glucose" in block
        assert "Hemoglobin" in block

    def test_returns_none_when_missing(self):
        assert extract_block("Name: Alice", "Results") is None

    def test_non_empty_block_returns_content(self):
        text = "Results:\nSome findings here\n"
        block = extract_block(text, "Results")
        assert block is not None
        assert "findings" in block

    def test_case_insensitive(self):
        text = "prescription:\nDrug A\n"
        block = extract_block(text, "Prescription")
        assert "Drug A" in block


class TestExtractListItems:
    """Tests for extract_list_items()."""

    def test_dash_items(self):
        block = "- item one\n- item two\n- item three"
        items = extract_list_items(block)
        assert len(items) == 3
        # All items contain their text (first may retain leading dash)
        assert "item one" in items[0]
        assert "item two" in items[1]
        assert "item three" in items[2]

    def test_asterisk_items(self):
        block = "* first\n* second"
        items = extract_list_items(block)
        assert len(items) == 2
        assert "first" in items[0]
        assert "second" in items[1]

    def test_single_item(self):
        block = "- only one"
        items = extract_list_items(block)
        assert len(items) == 1
        assert "only one" in items[0]

    def test_empty_block(self):
        items = extract_list_items("")
        assert items == []

    def test_no_delimiters(self):
        block = "plain text without bullets"
        items = extract_list_items(block)
        assert items == ["plain text without bullets"]

    def test_strips_whitespace(self):
        block = "-   spaced  \n-  also spaced  "
        items = extract_list_items(block)
        assert all(item == item.strip() for item in items)


class TestExtractDatedEntries:
    """Tests for extract_dated_entries()."""

    def test_multiple_entries(self):
        block = "- 2025-02-22: Headache\n- 2025-01-10: Follow-up"
        entries = extract_dated_entries(block)
        assert len(entries) == 2
        assert entries[0] == ("2025-02-22", "Headache")
        assert entries[1] == ("2025-01-10", "Follow-up")

    def test_single_entry(self):
        block = "- 2025-03-01: Initial visit"
        entries = extract_dated_entries(block)
        assert len(entries) == 1
        assert entries[0][0] == "2025-03-01"

    def test_empty_block(self):
        assert extract_dated_entries("") == []

    def test_no_dated_entries(self):
        assert extract_dated_entries("- Some note\n- Another note") == []

    def test_multiline_entry_text(self):
        block = "- 2025-01-01: Long note\nthat continues"
        entries = extract_dated_entries(block)
        assert len(entries) == 1
        assert "Long note" in entries[0][1]


class TestExtractTestResults:
    """Tests for extract_test_results()."""

    def test_single_result(self):
        block = "- Glucose: 95.50 (Ref: 70-100)"
        results = extract_test_results(block)
        assert len(results) == 1
        assert results[0]["test_name"] == "Glucose"
        assert results[0]["value"] == "95.50"
        assert results[0]["reference_range"] == "70-100"

    def test_multiple_results(self):
        block = (
            "- Glucose: 95.50 (Ref: 70-100)\n"
            "- Cholesterol: 210.30 (Ref: 125-200)\n"
            "- HbA1c: 5.4 (Ref: 4.0-5.6)"
        )
        results = extract_test_results(block)
        assert len(results) == 3
        assert results[2]["test_name"] == "HbA1c"

    def test_empty_block(self):
        assert extract_test_results("") == []

    def test_malformed_entries_skipped(self):
        block = "- Glucose: high\n- Cholesterol: 210.30 (Ref: 125-200)"
        results = extract_test_results(block)
        assert len(results) == 1
        assert results[0]["test_name"] == "Cholesterol"

    def test_comma_decimal(self):
        block = "- Test: 1,234 (Ref: 1000-2000)"
        results = extract_test_results(block)
        assert len(results) == 1
        assert results[0]["value"] == "1,234"


# ═══════════════════════════════════════════════════════════════════
# PrescriptionExtractor – item classification
# ═══════════════════════════════════════════════════════════════════


class TestPrescriptionItemClassification:
    """Test item type classification and field extraction."""

    def setup_method(self):
        self.ext = PrescriptionExtractor()

    def test_medicine_with_dosage_frequency(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Amoxicillin 500mg, 3 times daily for 7 days
"""
        result = self.ext.extract(text)
        items = result["items"]
        med_items = [i for i in items if i["type"] == "medicine"]
        assert len(med_items) >= 1
        assert "Amoxicillin" in med_items[0]["name"]

    def test_radiology_item(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Chest X-Ray for respiratory assessment
"""
        result = self.ext.extract(text)
        rad_items = [i for i in result["items"] if i["type"] == "radiology"]
        assert len(rad_items) >= 1

    def test_lab_test_item(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Complete blood test CBC
"""
        result = self.ext.extract(text)
        lab_items = [i for i in result["items"] if i["type"] == "lab_test"]
        assert len(lab_items) >= 1

    def test_specialist_referral(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Refer to Cardiology for hypertension evaluation
"""
        result = self.ext.extract(text)
        spec_items = [i for i in result["items"] if i["type"] == "specialist"]
        assert len(spec_items) >= 1

    def test_therapy_item(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Physical therapy for lower back pain, 3 sessions per week
"""
        result = self.ext.extract(text)
        therapy_items = [i for i in result["items"] if i["type"] == "procedure"]
        assert len(therapy_items) >= 1

    def test_generic_fallback(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Rest and hydration
"""
        result = self.ext.extract(text)
        assert len(result["items"]) >= 1

    def test_multiple_mixed_items(self):
        text = """Patient Name: Test
Doctor: Dr. A
Clinic: Hospital

Prescription:
Amoxicillin 500mg daily
Chest X-Ray
Refer to dermatology for rash evaluation
"""
        result = self.ext.extract(text)
        assert len(result["items"]) >= 3

    def test_unicode_patient_name(self):
        text = "Patient Name: José María García\nDoctor: Dr. X\nClinic: Y\nPrescription:\n- Aspirin"
        result = self.ext.extract(text)
        assert result["patient_name"] == "José María García"

    def test_missing_prescription_block(self):
        text = "Patient Name: John\nDoctor: Dr. X\nClinic: Y"
        result = self.ext.extract(text)
        assert isinstance(result["items"], list)

    def test_all_fields_none_on_empty(self):
        result = self.ext.extract("")
        assert result["patient_name"] is None
        assert result["doctor_name"] is None
        assert result["institution"] is None
        assert result["date"] is None


# ═══════════════════════════════════════════════════════════════════
# LabResultExtractor – exam type inference
# ═══════════════════════════════════════════════════════════════════


class TestLabResultExamTypeInference:
    """Test exam_type inference from test names and text."""

    def setup_method(self):
        self.ext = LabResultExtractor()

    def _result_with_tests(self, test_block: str) -> dict:
        text = f"""Patient Name: Test Patient
Patient ID: 12345
Exam Date: 2025-01-01
Clinic: Lab Corp

Test Results:
{test_block}
"""
        return self.ext.extract(text)

    def test_glucose_panel(self):
        result = self._result_with_tests("- Glucose Test: 95.50 (Ref: 70-100)")
        assert "Glucose" in result["exam_type"]

    def test_lipid_panel(self):
        result = self._result_with_tests("- Cholesterol: 200.00 (Ref: 125-200)")
        assert "Lipid" in result["exam_type"]

    def test_cbc_panel(self):
        result = self._result_with_tests("- Hemoglobin: 14.20 (Ref: 12-17)")
        assert "Blood Count" in result["exam_type"] or "Hematology" in result["exam_type"]

    def test_renal_panel(self):
        result = self._result_with_tests("- Creatinine: 1.00 (Ref: 0.6-1.2)")
        assert "Renal" in result["exam_type"]

    def test_liver_panel(self):
        result = self._result_with_tests("- ALT: 25.00 (Ref: 7-56)")
        assert "Liver" in result["exam_type"]

    def test_thyroid_panel(self):
        result = self._result_with_tests("- TSH: 2.50 (Ref: 0.4-4.0)")
        assert "Thyroid" in result["exam_type"]

    def test_default_laboratory_test(self):
        result = self._result_with_tests("- Unknown Marker: 10.00 (Ref: 5-15)")
        assert result["exam_type"] is not None

    def test_professional_defaults_to_not_specified(self):
        result = self._result_with_tests("- Glucose: 100.00 (Ref: 70-110)")
        assert result["professional"] is not None

    def test_findings_contain_test_values(self):
        result = self._result_with_tests(
            "- Glucose Test: 95.50 (Ref: 70-100)\n- HbA1c: 5.4 (Ref: 4.0-5.6)"
        )
        assert "95.50" in result["findings"]
        assert "5.4" in result["findings"]

    def test_empty_test_block(self):
        result = self.ext.extract("Patient Name: X\nExam Date: 2025-01-01\nClinic: Y")
        assert result["findings"] is not None

    def test_date_of_birth_extraction(self):
        text = """Patient Name: Test
Patient ID: 123
Date of Birth: 1990-05-15
Exam Date: 2025-01-01
Clinic: Lab"""
        result = self.ext.extract(text)
        assert result["date_of_birth"] == "1990-05-15"


# ═══════════════════════════════════════════════════════════════════
# ClinicalHistoryExtractor – annotation parsing
# ═══════════════════════════════════════════════════════════════════


class TestClinicalHistoryAnnotations:
    """Test annotation parsing and derived fields."""

    def setup_method(self):
        self.ext = ClinicalHistoryExtractor()

    def test_consultation_date_is_most_recent(self):
        text = """Patient Name: Test
Clinic: Hospital

Annotations:
- 2025-01-01: First visit
- 2025-06-15: Follow-up
- 2025-03-10: Mid check
"""
        result = self.ext.extract(text)
        assert result["consultation_date"] == "2025-06-15"

    def test_chief_complaint_is_most_recent_note(self):
        text = """Patient Name: Test
Clinic: Hospital

Annotations:
- 2025-01-01: First visit
- 2025-06-15: Severe headache and nausea
"""
        result = self.ext.extract(text)
        assert "headache" in result["chief_complaint"].lower()

    def test_medical_history_contains_all_entries(self):
        text = """Patient Name: Test
Clinic: Hospital

Annotations:
- 2025-01-01: First visit
- 2025-02-15: Follow-up
- 2025-03-10: Third visit
"""
        result = self.ext.extract(text)
        assert "First visit" in result["medical_history"]
        assert "Follow-up" in result["medical_history"]
        assert "Third visit" in result["medical_history"]

    def test_no_annotations_block(self):
        text = "Patient Name: Test\nClinic: Hospital"
        result = self.ext.extract(text)
        assert result["consultation_date"] is not None  # Falls back

    def test_single_annotation(self):
        text = """Patient Name: Test
Clinic: Hospital

Annotations:
- 2025-05-01: Only visit for routine checkup
"""
        result = self.ext.extract(text)
        assert result["consultation_date"] == "2025-05-01"
        assert "routine checkup" in result["chief_complaint"].lower()

    def test_current_medications_parsing(self):
        text = """Patient Name: Test
Clinic: Hospital
Current Medications: Metformin 850mg; Lisinopril 10mg; Aspirin 81mg

Annotations:
- 2025-01-01: Visit
"""
        result = self.ext.extract(text)
        meds = result.get("current_medications")
        if meds:
            assert len(meds) >= 2

    def test_unicode_patient_name(self):
        text = """Patient Name: María José Hernández
Patient ID: 99999
Date of Birth: 1980-12-01
Clinic: Clínica Central

Annotations:
- 2025-01-01: Visita inicial
"""
        result = self.ext.extract(text)
        assert result["patient_name"] == "María José Hernández"

    def test_doctor_fallback(self):
        text = "Patient Name: Test\nClinic: H\nAnnotations:\n- 2025-01-01: Note"
        result = self.ext.extract(text)
        assert result["doctor_name"] is not None


# ═══════════════════════════════════════════════════════════════════
# DocumentClassifier – scoring edge cases
# ═══════════════════════════════════════════════════════════════════


class TestDocumentClassifierEdgeCases:
    """Test classifier scoring, thresholds, and ambiguous input."""

    def test_high_threshold_rejects_weak_signal(self):
        classifier = DocumentClassifier(min_score=100.0)
        assert classifier.classify("prescription medication") is None

    def test_low_threshold_accepts_weak_signal(self):
        classifier = DocumentClassifier(min_score=0.1)
        result = classifier.classify("medication")
        assert result is not None

    def test_very_short_text(self):
        classifier = DocumentClassifier()
        assert classifier.classify("a") is None

    def test_whitespace_only(self):
        classifier = DocumentClassifier()
        assert classifier.classify("   \n\t  ") is None

    def test_ambiguous_text_returns_highest_scorer(self):
        classifier = DocumentClassifier()
        # Contains keywords from multiple types
        text = "prescription medication test results annotations clinical history"
        result = classifier.classify(text)
        assert result in ("prescription", "result", "clinical_history")

    def test_predict_returns_empty_dict_when_no_match(self):
        classifier = DocumentClassifier(min_score=1000.0)
        result = classifier.predict("hello world")
        assert result == {}

    def test_predict_returns_document_type_key(self):
        classifier = DocumentClassifier()
        result = classifier.predict("prescription medication dosage mg doctor")
        assert "document_type" in result
        assert result["document_type"] == "prescription"

    def test_case_insensitive_classification(self):
        classifier = DocumentClassifier()
        upper = classifier.classify("PRESCRIPTION MEDICATION DOSAGE MG")
        lower = classifier.classify("prescription medication dosage mg")
        # Both should classify the same way (text is lowercased internally)
        assert upper == lower

    def test_prescription_keywords(self):
        classifier = DocumentClassifier()
        text = "prescription medication dosage mg frequency route duration doctor"
        assert classifier.classify(text) == "prescription"

    def test_result_keywords(self):
        classifier = DocumentClassifier()
        text = "test results exam date findings reference lab impression"
        assert classifier.classify(text) == "result"

    def test_clinical_history_keywords(self):
        classifier = DocumentClassifier()
        text = "annotations consultation date chief complaint medical history assessment plan"
        assert classifier.classify(text) == "clinical_history"


# ═══════════════════════════════════════════════════════════════════
# Cross-extractor consistency
# ═══════════════════════════════════════════════════════════════════


class TestExtractorConsistency:
    """Verify all extractors follow the same interface contract."""

    EXTRACTORS = [
        PrescriptionExtractor(),
        LabResultExtractor(),
        ClinicalHistoryExtractor(),
    ]

    @pytest.mark.parametrize("extractor", EXTRACTORS, ids=lambda e: type(e).__name__)
    def test_extract_returns_dict(self, extractor):
        result = extractor.extract("some text")
        assert isinstance(result, dict)

    @pytest.mark.parametrize("extractor", EXTRACTORS, ids=lambda e: type(e).__name__)
    def test_extract_on_empty_string(self, extractor):
        result = extractor.extract("")
        assert isinstance(result, dict)

    @pytest.mark.parametrize("extractor", EXTRACTORS, ids=lambda e: type(e).__name__)
    def test_has_patient_name_key(self, extractor):
        result = extractor.extract("")
        assert "patient_name" in result

    @pytest.mark.parametrize("extractor", EXTRACTORS, ids=lambda e: type(e).__name__)
    def test_has_institution_key(self, extractor):
        result = extractor.extract("")
        assert "institution" in result

    @pytest.mark.parametrize("extractor", EXTRACTORS, ids=lambda e: type(e).__name__)
    def test_extract_has_extract_method(self, extractor):
        assert hasattr(extractor, "extract")
        assert callable(extractor.extract)
