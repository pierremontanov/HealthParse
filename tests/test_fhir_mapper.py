"""Tests for src.pipeline.fhir_mapper – resource mapping & Bundle builder."""
import uuid

import pytest

from src.pipeline.fhir_mapper import (
    build_fhir_bundle,
    clinical_history_to_fhir,
    map_to_fhir_loose,
    prescription_to_fhir,
    prune_none,
    result_to_fhir_loose,
)
from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def result_schema_sample():
    return ResultSchema(
        patient_name="Juan Perez",
        patient_id="12345",
        age=50,
        sex="M",
        date_of_birth="15-02-1973",
        exam_type="CR",
        study_area="Torax",
        exam_date="01-06-2024",
        findings="Infiltrados pulmonares visibles.",
        impression="Posible neumonía.",
        professional="Dra. Clara Gómez",
        institution="Hospital Nacional",
        notes="Requiere seguimiento clínico.",
    )


@pytest.fixture
def prescription_sample():
    return Prescription(
        patient_name="Carlos Ruiz",
        patient_id="88997766",
        date="25-05-2024",
        doctor_name="Dr. Andrés López",
        institution="Clínica Central",
        additional_notes="Paciente con hipertensión controlada.",
        items=[
            {
                "type": "medicine",
                "name": "Losartán 50mg",
                "notes": "Tomar en la mañana",
                "dosage": "50mg",
                "frequency": "1 vez al día",
                "route": "oral",
                "duration": "30 días",
            }
        ],
    )


@pytest.fixture
def clinical_history_sample():
    return ClinicalHistorySchema(
        patient_name="Lucía Méndez",
        patient_id="99887766",
        age=42,
        sex="F",
        date_of_birth="13-09-1982",
        consultation_date="10-05-2024",
        chief_complaint="Dolor abdominal persistente.",
        medical_history="Antecedente de gastritis.",
        current_medications=["Omeprazol 20mg"],
        physical_exam="Sensibilidad en epigastrio.",
        assessment="Probable úlcera gástrica.",
        plan="Endoscopía programada.",
        doctor_name="Dr. Jorge Méndez",
        institution="Centro Médico Colonial",
    )


# ── prune_none ──────────────────────────────────────────────────

class TestPruneNone:
    def test_removes_top_level_none(self):
        assert prune_none({"a": 1, "b": None}) == {"a": 1}

    def test_removes_nested_none(self):
        d = {"a": {"x": 1, "y": None}, "b": [1, None, {"c": None}]}
        result = prune_none(d)
        assert result == {"a": {"x": 1}, "b": [1, None, {}]}

    def test_non_dict_passthrough(self):
        assert prune_none("hello") == "hello"
        assert prune_none(42) == 42

    def test_empty_dict(self):
        assert prune_none({}) == {}


# ── DiagnosticReport mapping ────────────────────────────────────

class TestResultToFhir:
    def test_resource_type(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        assert fhir["resourceType"] == "DiagnosticReport"

    def test_has_uuid_id(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        uuid.UUID(fhir["id"])  # should not raise

    def test_has_meta_last_updated(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        assert "lastUpdated" in fhir["meta"]

    def test_subject_display(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        assert fhir["subject"]["display"] == result_schema_sample.patient_name

    def test_effective_date(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        assert fhir["effectiveDateTime"] == result_schema_sample.exam_date

    def test_conclusion(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        assert fhir["conclusion"] == result_schema_sample.impression

    def test_none_fields_pruned(self):
        """Result with no optional fields → no None in output."""
        minimal = ResultSchema(
            patient_name="Test",
            exam_type="X-Ray",
            exam_date="2024-01-01",
            findings="Normal",
            professional="Dr. A",
            institution="Hospital",
        )
        fhir = result_to_fhir_loose(minimal)
        _assert_no_none_values(fhir)

    def test_status_field(self, result_schema_sample):
        fhir = result_to_fhir_loose(result_schema_sample)
        assert fhir["status"] == "final"


# ── MedicationRequest mapping ───────────────────────────────────

class TestPrescriptionToFhir:
    def test_resource_type(self, prescription_sample):
        fhir = prescription_to_fhir(prescription_sample)
        assert fhir["resourceType"] == "MedicationRequest"

    def test_has_uuid_id(self, prescription_sample):
        fhir = prescription_to_fhir(prescription_sample)
        uuid.UUID(fhir["id"])

    def test_status_and_intent(self, prescription_sample):
        fhir = prescription_to_fhir(prescription_sample)
        assert fhir["status"] == "active"
        assert fhir["intent"] == "order"

    def test_contained_medications(self, prescription_sample):
        fhir = prescription_to_fhir(prescription_sample)
        assert len(fhir["contained"]) == 1
        assert fhir["contained"][0]["resourceType"] == "Medication"

    def test_requester_display(self, prescription_sample):
        fhir = prescription_to_fhir(prescription_sample)
        assert fhir["requester"]["display"] == prescription_sample.doctor_name

    def test_subject_display(self, prescription_sample):
        fhir = prescription_to_fhir(prescription_sample)
        assert fhir["subject"]["display"] == prescription_sample.patient_name

    def test_no_medicines_no_contained(self):
        """Prescription with only lab items → no contained key."""
        p = Prescription(
            patient_name="Test",
            items=[{"type": "lab_test", "name": "CBC", "test_type": "blood"}],
        )
        fhir = prescription_to_fhir(p)
        assert "contained" not in fhir


# ── Encounter mapping ───────────────────────────────────────────

class TestClinicalHistoryToFhir:
    def test_resource_type(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        assert fhir["resourceType"] == "Encounter"

    def test_has_uuid_id(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        uuid.UUID(fhir["id"])

    def test_status_finished(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        assert fhir["status"] == "finished"

    def test_class_ambulatory(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        assert fhir["class"]["code"] == "AMB"

    def test_subject_display(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        assert fhir["subject"]["display"] == clinical_history_sample.patient_name

    def test_period_start(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        assert fhir["period"]["start"] == clinical_history_sample.consultation_date

    def test_note_plan(self, clinical_history_sample):
        fhir = clinical_history_to_fhir(clinical_history_sample)
        assert fhir["note"][0]["text"] == clinical_history_sample.plan

    def test_minimal_no_none(self):
        """Encounter with only required fields → no None in output."""
        c = ClinicalHistorySchema(
            patient_name="Test",
            consultation_date="2024-01-01",
            doctor_name="Dr. B",
        )
        fhir = clinical_history_to_fhir(c)
        _assert_no_none_values(fhir)


# ── Dispatcher ──────────────────────────────────────────────────

class TestMapToFhirLoose:
    def test_dispatches_result(self, result_schema_sample):
        fhir = map_to_fhir_loose(result_schema_sample)
        assert fhir["resourceType"] == "DiagnosticReport"

    def test_dispatches_prescription(self, prescription_sample):
        fhir = map_to_fhir_loose(prescription_sample)
        assert fhir["resourceType"] == "MedicationRequest"

    def test_dispatches_clinical(self, clinical_history_sample):
        fhir = map_to_fhir_loose(clinical_history_sample)
        assert fhir["resourceType"] == "Encounter"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            map_to_fhir_loose({"not": "a model"})


# ── Bundle builder ──────────────────────────────────────────────

class TestBuildFhirBundle:
    def test_bundle_structure(self, result_schema_sample, prescription_sample):
        r1 = map_to_fhir_loose(result_schema_sample)
        r2 = map_to_fhir_loose(prescription_sample)
        bundle = build_fhir_bundle([r1, r2])

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["total"] == 2
        assert len(bundle["entry"]) == 2

    def test_bundle_has_uuid(self):
        bundle = build_fhir_bundle([])
        uuid.UUID(bundle["id"])

    def test_bundle_has_meta(self):
        bundle = build_fhir_bundle([])
        assert "lastUpdated" in bundle["meta"]

    def test_entries_have_full_url(self, result_schema_sample):
        r = map_to_fhir_loose(result_schema_sample)
        bundle = build_fhir_bundle([r])
        assert bundle["entry"][0]["fullUrl"].startswith("urn:uuid:")

    def test_entries_contain_resource(self, result_schema_sample):
        r = map_to_fhir_loose(result_schema_sample)
        bundle = build_fhir_bundle([r])
        assert bundle["entry"][0]["resource"]["resourceType"] == "DiagnosticReport"

    def test_empty_bundle(self):
        bundle = build_fhir_bundle([])
        assert bundle["total"] == 0
        assert bundle["entry"] == []

    def test_custom_bundle_type(self):
        bundle = build_fhir_bundle([], bundle_type="document")
        assert bundle["type"] == "document"

    def test_custom_bundle_id(self):
        bundle = build_fhir_bundle([], bundle_id="custom-123")
        assert bundle["id"] == "custom-123"


# ── Helper ──────────────────────────────────────────────────────

def _assert_no_none_values(d, path=""):
    """Recursively check that no dict value is None."""
    if isinstance(d, dict):
        for k, v in d.items():
            assert v is not None, f"None value at {path}.{k}"
            _assert_no_none_values(v, f"{path}.{k}")
    elif isinstance(d, list):
        for i, item in enumerate(d):
            _assert_no_none_values(item, f"{path}[{i}]")
