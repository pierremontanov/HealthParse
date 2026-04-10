"""End-to-end integration tests for the FHIR pipeline.

Validates the complete path: validated Pydantic models → FHIR mapping →
individual resource files + Bundle → JSON round-trip.
"""
import json
import uuid
from pathlib import Path

import pytest

from src.pipeline.fhir_mapper import build_fhir_bundle, map_to_fhir_loose
from src.pipeline.fhir_output_saver import save_fhir_bundle, save_fhir_output
from src.pipeline.output_formatter import export_fhir, export_results
from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def lab_result():
    return ResultSchema(
        patient_name="María García",
        patient_id="PAT-001",
        age=35,
        sex="F",
        exam_type="Blood Panel",
        exam_date="2024-03-15",
        findings="WBC elevated at 15,000/µL. CRP 42 mg/L.",
        impression="Acute inflammatory process.",
        professional="Dr. Roberto Silva",
        institution="Clínica del Valle",
    )


@pytest.fixture
def prescription():
    return Prescription(
        patient_name="María García",
        patient_id="PAT-001",
        date="2024-03-15",
        doctor_name="Dr. Roberto Silva",
        institution="Clínica del Valle",
        items=[
            {
                "type": "medicine",
                "name": "Amoxicillin 500mg",
                "dosage": "500mg",
                "frequency": "Every 8 hours",
                "route": "oral",
                "duration": "7 days",
                "notes": "Take with food",
            },
            {
                "type": "lab_test",
                "name": "Follow-up CBC",
                "test_type": "blood",
            },
        ],
    )


@pytest.fixture
def clinical_history():
    return ClinicalHistorySchema(
        patient_name="María García",
        patient_id="PAT-001",
        age=35,
        sex="F",
        consultation_date="2024-03-15",
        chief_complaint="Fever and sore throat for 3 days.",
        medical_history="No significant past history.",
        current_medications=["None"],
        physical_exam="Temp 38.5°C, pharyngeal erythema.",
        assessment="Acute pharyngitis with bacterial superinfection.",
        plan="Start antibiotics, rest, follow-up in 7 days.",
        doctor_name="Dr. Roberto Silva",
        institution="Clínica del Valle",
    )


@pytest.fixture
def all_result_dicts(lab_result, prescription, clinical_history):
    """Pipeline-style result dicts for all three document types."""
    return [
        {
            "file": "lab_result.pdf",
            "document_type": "result",
            "extracted_data": lab_result.model_dump(),
        },
        {
            "file": "prescription.pdf",
            "document_type": "prescription",
            "extracted_data": prescription.model_dump(),
        },
        {
            "file": "clinical_note.pdf",
            "document_type": "clinical_history",
            "extracted_data": clinical_history.model_dump(),
        },
    ]


# ── Single-resource round-trip ──────────────────────────────────

class TestSingleResourceRoundTrip:
    """Map a document, save to disk, re-read, and verify structure."""

    def test_result_round_trip(self, tmp_path, lab_result):
        out = tmp_path / "result_fhir.json"
        save_fhir_output(lab_result, str(out))
        data = json.loads(out.read_text("utf-8"))

        assert data["resourceType"] == "DiagnosticReport"
        assert data["subject"]["display"] == "María García"
        uuid.UUID(data["id"])  # valid UUID
        assert "lastUpdated" in data["meta"]

    def test_prescription_round_trip(self, tmp_path, prescription):
        out = tmp_path / "rx_fhir.json"
        save_fhir_output(prescription, str(out))
        data = json.loads(out.read_text("utf-8"))

        assert data["resourceType"] == "MedicationRequest"
        assert data["status"] == "active"
        assert data["intent"] == "order"
        # Only medicine items become contained Medication resources
        assert len(data["contained"]) == 1
        assert data["contained"][0]["code"]["text"] == "Amoxicillin 500mg"

    def test_clinical_history_round_trip(self, tmp_path, clinical_history):
        out = tmp_path / "enc_fhir.json"
        save_fhir_output(clinical_history, str(out))
        data = json.loads(out.read_text("utf-8"))

        assert data["resourceType"] == "Encounter"
        assert data["status"] == "finished"
        assert data["class"]["code"] == "AMB"
        assert data["period"]["start"] == "2024-03-15"


# ── Bundle round-trip ───────────────────────────────────────────

class TestBundleRoundTrip:
    """Build a Bundle, save, re-read, verify."""

    def test_full_bundle_round_trip(
        self, tmp_path, lab_result, prescription, clinical_history
    ):
        resources = [
            map_to_fhir_loose(lab_result),
            map_to_fhir_loose(prescription),
            map_to_fhir_loose(clinical_history),
        ]
        out = tmp_path / "bundle.json"
        save_fhir_bundle(resources, str(out))
        data = json.loads(out.read_text("utf-8"))

        assert data["resourceType"] == "Bundle"
        assert data["type"] == "collection"
        assert data["total"] == 3

        types_in_bundle = {
            e["resource"]["resourceType"] for e in data["entry"]
        }
        assert types_in_bundle == {
            "DiagnosticReport",
            "MedicationRequest",
            "Encounter",
        }

    def test_each_entry_has_full_url(
        self, tmp_path, lab_result, prescription, clinical_history
    ):
        resources = [
            map_to_fhir_loose(lab_result),
            map_to_fhir_loose(prescription),
            map_to_fhir_loose(clinical_history),
        ]
        out = tmp_path / "bundle.json"
        save_fhir_bundle(resources, str(out))
        data = json.loads(out.read_text("utf-8"))

        for entry in data["entry"]:
            assert entry["fullUrl"].startswith("urn:uuid:")
            # fullUrl should contain a valid UUID
            urn_uuid = entry["fullUrl"].replace("urn:uuid:", "")
            uuid.UUID(urn_uuid)


# ── export_fhir batch flow ──────────────────────────────────────

class TestExportFhirBatch:
    """Test the export_fhir function from output_formatter."""

    def test_exports_all_three_types(self, tmp_path, all_result_dicts):
        path = export_fhir(all_result_dicts, str(tmp_path), bundle=False)
        fhir_dir = Path(path)
        files = sorted(fhir_dir.glob("*_fhir.json"))
        assert len(files) == 3

    def test_bundle_includes_all(self, tmp_path, all_result_dicts):
        export_fhir(all_result_dicts, str(tmp_path))
        bundle_path = Path(tmp_path) / "dociq_fhir" / "bundle.json"
        assert bundle_path.exists()
        data = json.loads(bundle_path.read_text("utf-8"))
        assert data["total"] == 3

    def test_individual_files_valid(self, tmp_path, all_result_dicts):
        path = export_fhir(all_result_dicts, str(tmp_path), bundle=False)
        fhir_dir = Path(path)
        for fhir_file in fhir_dir.glob("*_fhir.json"):
            data = json.loads(fhir_file.read_text("utf-8"))
            assert "resourceType" in data
            assert "id" in data
            assert "meta" in data

    def test_skips_unknown_document_type(self, tmp_path):
        items = [
            {
                "file": "mystery.pdf",
                "document_type": "invoice",
                "extracted_data": {"total": 100},
            },
        ]
        export_fhir(items, str(tmp_path), bundle=False)
        fhir_dir = tmp_path / "dociq_fhir"
        assert list(fhir_dir.glob("*_fhir.json")) == []

    def test_skips_invalid_data(self, tmp_path):
        """Document type recognised but data fails schema validation."""
        items = [
            {
                "file": "bad_result.pdf",
                "document_type": "result",
                "extracted_data": {"patient_name": "Missing fields"},
            },
        ]
        export_fhir(items, str(tmp_path), bundle=False)
        fhir_dir = tmp_path / "dociq_fhir"
        assert list(fhir_dir.glob("*_fhir.json")) == []


# ── export_results dispatcher with FHIR ─────────────────────────

class TestExportResultsFhir:
    def test_dispatcher_routes_to_fhir(self, tmp_path, all_result_dicts):
        path = export_results(
            all_result_dicts, output_dir=str(tmp_path), fmt="fhir"
        )
        assert "fhir" in path

    def test_dispatcher_creates_bundle(self, tmp_path, all_result_dicts):
        export_results(
            all_result_dicts, output_dir=str(tmp_path), fmt="fhir"
        )
        bundle_path = tmp_path / "dociq_fhir" / "bundle.json"
        assert bundle_path.exists()


# ── Null-field pruning in persisted output ──────────────────────

class TestNullPruning:
    def test_no_none_in_persisted_result(self, tmp_path, lab_result):
        out = tmp_path / "result.json"
        save_fhir_output(lab_result, str(out))
        data = json.loads(out.read_text("utf-8"))
        _check_no_none(data)

    def test_no_none_in_persisted_bundle(
        self, tmp_path, lab_result, prescription, clinical_history
    ):
        resources = [
            map_to_fhir_loose(lab_result),
            map_to_fhir_loose(prescription),
            map_to_fhir_loose(clinical_history),
        ]
        out = tmp_path / "bundle.json"
        save_fhir_bundle(resources, str(out))
        data = json.loads(out.read_text("utf-8"))
        _check_no_none(data)


def _check_no_none(obj, path="root"):
    """Recursively assert no value in the structure is None."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert v is not None, f"None at {path}.{k}"
            _check_no_none(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _check_no_none(item, f"{path}[{i}]")
