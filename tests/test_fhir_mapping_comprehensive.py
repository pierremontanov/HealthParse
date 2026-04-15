"""Comprehensive pytest tests for FHIR mapping (#32).

Covers edge cases, parametrized field-level mapping, optional-field
handling, multi-item prescriptions, and Bundle builder variants that
the existing test_fhir_mapper.py does not exercise.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

import pytest

from src.pipeline.fhir_mapper import (
    build_fhir_bundle,
    clinical_history_to_fhir,
    map_to_fhir_loose,
    prescription_to_fhir,
    prune_none,
    result_to_fhir_loose,
)
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.schemas import ResultSchema


# ═══════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def minimal_result():
    return ResultSchema(
        patient_name="Min Patient",
        exam_type="X-Ray",
        exam_date="2024-01-01",
        findings="Normal.",
        professional="Dr. A",
        institution="Clinic",
    )


@pytest.fixture
def full_result():
    return ResultSchema(
        patient_name="Full Patient",
        patient_id="ID-999",
        age=65,
        sex="M",
        date_of_birth="10-03-1959",
        exam_type="MRI",
        study_area="Brain",
        exam_date="15-06-2024",
        findings="No abnormalities.",
        impression="Normal brain MRI.",
        professional="Dr. Neuro",
        institution="Metro Hospital",
        notes="Follow-up in 12 months.",
    )


@pytest.fixture
def minimal_prescription():
    return Prescription(
        patient_name="Min Rx",
        items=[{"type": "medicine", "name": "Aspirin"}],
    )


@pytest.fixture
def full_prescription():
    return Prescription(
        patient_name="Full Rx",
        patient_id="RX-001",
        date="20-04-2024",
        doctor_name="Dr. Pharma",
        institution="Pharmacy Center",
        additional_notes="Allergic to penicillin.",
        items=[
            {
                "type": "medicine",
                "name": "Ibuprofen 400mg",
                "dosage": "400mg",
                "frequency": "Every 8 hours",
                "route": "oral",
                "duration": "5 days",
                "notes": "After meals",
            },
            {
                "type": "medicine",
                "name": "Omeprazol 20mg",
                "dosage": "20mg",
                "frequency": "Once daily",
                "route": "oral",
                "duration": "10 days",
            },
            {
                "type": "lab_test",
                "name": "Complete Blood Count",
                "test_type": "blood",
            },
        ],
    )


@pytest.fixture
def minimal_clinical():
    return ClinicalHistorySchema(
        patient_name="Min Clinical",
        consultation_date="2024-01-01",
        doctor_name="Dr. B",
    )


@pytest.fixture
def full_clinical():
    return ClinicalHistorySchema(
        patient_name="Full Clinical",
        patient_id="CH-001",
        age=30,
        sex="F",
        date_of_birth="25-12-1994",
        consultation_date="10-07-2024",
        chief_complaint="Persistent headache.",
        medical_history="Migraine since age 20.",
        current_medications=["Sumatriptan 50mg", "Magnesium"],
        physical_exam="BP 120/80. No papilledema.",
        assessment="Chronic migraine without aura.",
        plan="Prophylactic propranolol. MRI if worsens.",
        doctor_name="Dr. Neuro",
        institution="Neuro Clinic",
    )


# ═══════════════════════════════════════════════════════════════════
# prune_none – edge cases
# ═══════════════════════════════════════════════════════════════════

class TestPruneNoneEdgeCases:
    def test_deeply_nested(self):
        d = {"a": {"b": {"c": {"d": None, "e": 1}}}}
        assert prune_none(d) == {"a": {"b": {"c": {"e": 1}}}}

    def test_list_with_dicts(self):
        d = [{"a": 1, "b": None}, {"c": None}]
        assert prune_none(d) == [{"a": 1}, {}]

    def test_preserves_false_and_zero(self):
        d = {"a": 0, "b": False, "c": "", "d": None}
        result = prune_none(d)
        assert result == {"a": 0, "b": False, "c": ""}

    def test_none_in_list_preserved(self):
        """None inside a list is preserved (only dict keys are pruned)."""
        d = {"items": [1, None, 3]}
        result = prune_none(d)
        assert result == {"items": [1, None, 3]}

    def test_returns_primitives_unchanged(self):
        assert prune_none(42) == 42
        assert prune_none("hello") == "hello"
        assert prune_none(True) is True
        assert prune_none(None) is None

    def test_empty_nested_structures(self):
        d = {"a": {}, "b": [], "c": {"d": {}}}
        assert prune_none(d) == {"a": {}, "b": [], "c": {"d": {}}}


# ═══════════════════════════════════════════════════════════════════
# DiagnosticReport – field-level parametrized tests
# ═══════════════════════════════════════════════════════════════════

class TestResultToFhirFields:
    def test_resource_type(self, minimal_result):
        assert result_to_fhir_loose(minimal_result)["resourceType"] == "DiagnosticReport"

    def test_status_always_final(self, minimal_result, full_result):
        assert result_to_fhir_loose(minimal_result)["status"] == "final"
        assert result_to_fhir_loose(full_result)["status"] == "final"

    def test_id_is_valid_uuid(self, minimal_result):
        fhir = result_to_fhir_loose(minimal_result)
        uuid.UUID(fhir["id"])

    def test_meta_last_updated(self, minimal_result):
        fhir = result_to_fhir_loose(minimal_result)
        assert "T" in fhir["meta"]["lastUpdated"]  # ISO timestamp

    def test_subject_display_matches_patient(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["subject"]["display"] == "Full Patient"

    def test_subject_identifier(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["subject"]["identifier"]["value"] == "ID-999"

    def test_effective_date(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["effectiveDateTime"] == "15-06-2024"

    def test_category_exam_type(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["category"][0]["coding"][0]["display"] == "MRI"

    def test_code_study_area(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["code"]["text"] == "Brain"

    def test_code_defaults_to_general(self, minimal_result):
        fhir = result_to_fhir_loose(minimal_result)
        assert fhir["code"]["text"] == "General"

    def test_conclusion_is_impression(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["conclusion"] == "Normal brain MRI."

    def test_findings_in_presented_form(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["presentedForm"][0]["data"] == "No abnormalities."
        assert fhir["presentedForm"][0]["contentType"] == "text/plain"

    def test_performer(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["performer"][0]["display"] == "Dr. Neuro"

    def test_notes_present_when_set(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["note"][0]["text"] == "Follow-up in 12 months."

    def test_notes_absent_when_none(self, minimal_result):
        fhir = result_to_fhir_loose(minimal_result)
        assert "note" not in fhir

    def test_issuer(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        assert fhir["issuer"]["display"] == "Metro Hospital"

    def test_minimal_has_no_none(self, minimal_result):
        fhir = result_to_fhir_loose(minimal_result)
        _assert_no_none_values(fhir)

    def test_full_has_no_none(self, full_result):
        fhir = result_to_fhir_loose(full_result)
        _assert_no_none_values(fhir)


# ═══════════════════════════════════════════════════════════════════
# MedicationRequest – multi-item & edge cases
# ═══════════════════════════════════════════════════════════════════

class TestPrescriptionToFhirFields:
    def test_resource_type(self, minimal_prescription):
        assert prescription_to_fhir(minimal_prescription)["resourceType"] == "MedicationRequest"

    def test_status_active(self, minimal_prescription):
        assert prescription_to_fhir(minimal_prescription)["status"] == "active"

    def test_intent_order(self, minimal_prescription):
        assert prescription_to_fhir(minimal_prescription)["intent"] == "order"

    def test_subject_display(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        assert fhir["subject"]["display"] == "Full Rx"

    def test_subject_identifier(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        assert fhir["subject"]["identifier"]["value"] == "RX-001"

    def test_authored_on(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        assert fhir["authoredOn"] == "20-04-2024"

    def test_requester(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        assert fhir["requester"]["display"] == "Dr. Pharma"

    def test_additional_notes(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        assert fhir["note"][0]["text"] == "Allergic to penicillin."

    def test_no_notes_when_none(self, minimal_prescription):
        fhir = prescription_to_fhir(minimal_prescription)
        assert "note" not in fhir

    def test_multi_medicine_contained(self, full_prescription):
        """Only medicine items become contained Medication resources."""
        fhir = prescription_to_fhir(full_prescription)
        assert len(fhir["contained"]) == 2  # 2 medicines, 1 lab_test skipped

    def test_contained_medication_names(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        names = [m["code"]["text"] for m in fhir["contained"]]
        assert "Ibuprofen 400mg" in names
        assert "Omeprazol 20mg" in names

    def test_lab_only_no_contained(self):
        """Prescription with only lab items → no contained."""
        p = Prescription(
            patient_name="Lab Only",
            items=[
                {"type": "lab_test", "name": "CBC", "test_type": "blood"},
                {"type": "lab_test", "name": "Urinalysis", "test_type": "urine"},
            ],
        )
        fhir = prescription_to_fhir(p)
        assert "contained" not in fhir

    def test_single_medicine_dosage(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        med = fhir["contained"][0]
        dose_instr = med["dosageInstruction"][0]
        assert dose_instr["doseAndRate"][0]["doseQuantity"]["value"] is not None

    def test_minimal_has_no_none(self, minimal_prescription):
        fhir = prescription_to_fhir(minimal_prescription)
        _assert_no_none_values(fhir)

    def test_full_has_no_none(self, full_prescription):
        fhir = prescription_to_fhir(full_prescription)
        _assert_no_none_values(fhir)


# ═══════════════════════════════════════════════════════════════════
# Encounter – optional fields & edge cases
# ═══════════════════════════════════════════════════════════════════

class TestClinicalHistoryToFhirFields:
    def test_resource_type(self, minimal_clinical):
        assert clinical_history_to_fhir(minimal_clinical)["resourceType"] == "Encounter"

    def test_status_finished(self, minimal_clinical):
        assert clinical_history_to_fhir(minimal_clinical)["status"] == "finished"

    def test_class_ambulatory(self, minimal_clinical):
        fhir = clinical_history_to_fhir(minimal_clinical)
        assert fhir["class"]["code"] == "AMB"
        assert fhir["class"]["display"] == "ambulatory"

    def test_subject_display(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["subject"]["display"] == "Full Clinical"

    def test_subject_identifier(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["subject"]["identifier"]["value"] == "CH-001"

    def test_period_start(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["period"]["start"] == "10-07-2024"

    def test_reason_code(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["reasonCode"][0]["text"] == "Persistent headache."

    def test_no_reason_code_when_none(self, minimal_clinical):
        fhir = clinical_history_to_fhir(minimal_clinical)
        assert "reasonCode" not in fhir

    def test_diagnosis_assessment(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["diagnosis"][0]["condition"]["display"] == "Chronic migraine without aura."

    def test_no_diagnosis_when_none(self, minimal_clinical):
        fhir = clinical_history_to_fhir(minimal_clinical)
        assert "diagnosis" not in fhir

    def test_participant_doctor(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["participant"][0]["individual"]["display"] == "Dr. Neuro"

    def test_location_institution(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert fhir["location"][0]["location"]["display"] == "Neuro Clinic"

    def test_no_location_when_none(self, minimal_clinical):
        fhir = clinical_history_to_fhir(minimal_clinical)
        assert "location" not in fhir

    def test_note_plan(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        assert "Prophylactic propranolol" in fhir["note"][0]["text"]

    def test_no_note_when_none(self, minimal_clinical):
        fhir = clinical_history_to_fhir(minimal_clinical)
        assert "note" not in fhir

    def test_minimal_has_no_none(self, minimal_clinical):
        fhir = clinical_history_to_fhir(minimal_clinical)
        _assert_no_none_values(fhir)

    def test_full_has_no_none(self, full_clinical):
        fhir = clinical_history_to_fhir(full_clinical)
        _assert_no_none_values(fhir)


# ═══════════════════════════════════════════════════════════════════
# Dispatcher – parametrized
# ═══════════════════════════════════════════════════════════════════

class TestMapToFhirLooseParametrized:
    @pytest.mark.parametrize("fixture_name,expected_type", [
        ("minimal_result", "DiagnosticReport"),
        ("full_result", "DiagnosticReport"),
        ("minimal_prescription", "MedicationRequest"),
        ("full_prescription", "MedicationRequest"),
        ("minimal_clinical", "Encounter"),
        ("full_clinical", "Encounter"),
    ])
    def test_dispatches_to_correct_type(self, fixture_name, expected_type, request):
        doc = request.getfixturevalue(fixture_name)
        fhir = map_to_fhir_loose(doc)
        assert fhir["resourceType"] == expected_type

    @pytest.mark.parametrize("fixture_name", [
        "minimal_result", "full_result",
        "minimal_prescription", "full_prescription",
        "minimal_clinical", "full_clinical",
    ])
    def test_all_have_uuid_id(self, fixture_name, request):
        doc = request.getfixturevalue(fixture_name)
        fhir = map_to_fhir_loose(doc)
        uuid.UUID(fhir["id"])

    @pytest.mark.parametrize("fixture_name", [
        "minimal_result", "full_result",
        "minimal_prescription", "full_prescription",
        "minimal_clinical", "full_clinical",
    ])
    def test_all_have_meta(self, fixture_name, request):
        doc = request.getfixturevalue(fixture_name)
        fhir = map_to_fhir_loose(doc)
        assert "lastUpdated" in fhir["meta"]

    @pytest.mark.parametrize("fixture_name", [
        "minimal_result", "full_result",
        "minimal_prescription", "full_prescription",
        "minimal_clinical", "full_clinical",
    ])
    def test_no_none_in_output(self, fixture_name, request):
        doc = request.getfixturevalue(fixture_name)
        fhir = map_to_fhir_loose(doc)
        _assert_no_none_values(fhir)

    def test_unsupported_string_raises(self):
        from src.pipeline.exceptions import FHIRMappingError
        with pytest.raises(FHIRMappingError):
            map_to_fhir_loose("not a model")

    def test_unsupported_dict_raises(self):
        from src.pipeline.exceptions import FHIRMappingError
        with pytest.raises(FHIRMappingError):
            map_to_fhir_loose({"key": "value"})

    def test_unsupported_none_raises(self):
        from src.pipeline.exceptions import FHIRMappingError
        with pytest.raises(FHIRMappingError):
            map_to_fhir_loose(None)


# ═══════════════════════════════════════════════════════════════════
# Bundle builder – advanced
# ═══════════════════════════════════════════════════════════════════

class TestBuildFhirBundleAdvanced:
    def test_mixed_resource_types(
        self, minimal_result, minimal_prescription, minimal_clinical
    ):
        resources = [
            map_to_fhir_loose(minimal_result),
            map_to_fhir_loose(minimal_prescription),
            map_to_fhir_loose(minimal_clinical),
        ]
        bundle = build_fhir_bundle(resources)
        types = {e["resource"]["resourceType"] for e in bundle["entry"]}
        assert types == {"DiagnosticReport", "MedicationRequest", "Encounter"}

    def test_large_bundle(self, minimal_result):
        resources = [result_to_fhir_loose(minimal_result) for _ in range(50)]
        bundle = build_fhir_bundle(resources)
        assert bundle["total"] == 50
        assert len(bundle["entry"]) == 50

    def test_bundle_entries_fullurl_unique(self, full_result):
        resources = [result_to_fhir_loose(full_result) for _ in range(10)]
        bundle = build_fhir_bundle(resources)
        urls = [e["fullUrl"] for e in bundle["entry"]]
        assert len(urls) == len(set(urls)), "fullUrl values should be unique"

    def test_bundle_types(self, minimal_result):
        r = result_to_fhir_loose(minimal_result)
        for btype in ("collection", "document", "batch", "transaction"):
            bundle = build_fhir_bundle([r], bundle_type=btype)
            assert bundle["type"] == btype

    def test_custom_id_preserved(self):
        bundle = build_fhir_bundle([], bundle_id="my-custom-id")
        assert bundle["id"] == "my-custom-id"

    def test_auto_id_is_valid_uuid(self):
        bundle = build_fhir_bundle([])
        uuid.UUID(bundle["id"])

    def test_empty_bundle_valid_structure(self):
        bundle = build_fhir_bundle([])
        assert bundle["resourceType"] == "Bundle"
        assert bundle["total"] == 0
        assert bundle["entry"] == []
        assert "meta" in bundle

    def test_entry_resource_matches_input(self, full_result):
        resource = result_to_fhir_loose(full_result)
        bundle = build_fhir_bundle([resource])
        assert bundle["entry"][0]["resource"] is resource


# ═══════════════════════════════════════════════════════════════════
# Unique ID generation
# ═══════════════════════════════════════════════════════════════════

class TestUniqueIds:
    def test_different_calls_get_different_ids(self, minimal_result):
        fhir1 = result_to_fhir_loose(minimal_result)
        fhir2 = result_to_fhir_loose(minimal_result)
        assert fhir1["id"] != fhir2["id"]

    def test_different_types_get_different_ids(
        self, minimal_result, minimal_prescription, minimal_clinical
    ):
        ids = {
            map_to_fhir_loose(minimal_result)["id"],
            map_to_fhir_loose(minimal_prescription)["id"],
            map_to_fhir_loose(minimal_clinical)["id"],
        }
        assert len(ids) == 3


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def _assert_no_none_values(d: Any, path: str = "") -> None:
    """Recursively verify no dict value is None."""
    if isinstance(d, dict):
        for k, v in d.items():
            assert v is not None, f"None at {path}.{k}"
            _assert_no_none_values(v, f"{path}.{k}")
    elif isinstance(d, list):
        for i, item in enumerate(d):
            _assert_no_none_values(item, f"{path}[{i}]")
