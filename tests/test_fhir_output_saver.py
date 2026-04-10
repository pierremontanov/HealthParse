"""Tests for src.pipeline.fhir_output_saver – single & bundle persistence."""
import json

import pytest

from src.pipeline.fhir_mapper import map_to_fhir_loose
from src.pipeline.fhir_output_saver import save_fhir_bundle, save_fhir_output
from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema


@pytest.fixture
def sample_result():
    return ResultSchema(
        patient_name="Gloria Ines Montaño Villada",
        patient_id="24314628",
        age=71,
        sex="F",
        date_of_birth="27-04-1953",
        exam_type="CR",
        study_area="Columna Dorsal",
        exam_date="08-08-2024",
        findings="Cambios incipientes de tipo degenerativo crónico.",
        impression="Alineamiento satisfactorio. Tejidos blandos normales.",
        professional="Dra. Fátima Mota Arteaga",
        institution="Centro Médico San José",
        notes="No se evidencian fracturas.",
    )


@pytest.fixture
def sample_prescription():
    return Prescription(
        patient_name="Carlos Ruiz",
        patient_id="88997766",
        date="25-05-2024",
        doctor_name="Dr. Andrés López",
        items=[
            {
                "type": "medicine",
                "name": "Losartán 50mg",
                "dosage": "50mg",
                "frequency": "daily",
                "route": "oral",
                "duration": "30 días",
            }
        ],
    )


# ── save_fhir_output ───────────────────────────────────────────

class TestSaveFhirOutput:
    def test_creates_file(self, tmp_path, sample_result):
        out = tmp_path / "fhir_result.json"
        save_fhir_output(sample_result, str(out))
        assert out.exists()

    def test_valid_json(self, tmp_path, sample_result):
        out = tmp_path / "fhir_result.json"
        save_fhir_output(sample_result, str(out))
        data = json.loads(out.read_text("utf-8"))
        assert data["resourceType"] == "DiagnosticReport"

    def test_returns_absolute_path(self, tmp_path, sample_result):
        out = tmp_path / "fhir_result.json"
        result = save_fhir_output(sample_result, str(out))
        assert result.endswith("fhir_result.json")

    def test_creates_parent_dirs(self, tmp_path, sample_result):
        out = tmp_path / "deep" / "nested" / "result.json"
        save_fhir_output(sample_result, str(out))
        assert out.exists()

    def test_has_meta(self, tmp_path, sample_result):
        out = tmp_path / "fhir_result.json"
        save_fhir_output(sample_result, str(out))
        data = json.loads(out.read_text("utf-8"))
        assert "meta" in data
        assert "lastUpdated" in data["meta"]


# ── save_fhir_bundle ───────────────────────────────────────────

class TestSaveFhirBundle:
    def test_creates_bundle_file(self, tmp_path, sample_result, sample_prescription):
        r1 = map_to_fhir_loose(sample_result)
        r2 = map_to_fhir_loose(sample_prescription)
        out = tmp_path / "bundle.json"
        save_fhir_bundle([r1, r2], str(out))
        assert out.exists()

    def test_bundle_structure(self, tmp_path, sample_result, sample_prescription):
        r1 = map_to_fhir_loose(sample_result)
        r2 = map_to_fhir_loose(sample_prescription)
        out = tmp_path / "bundle.json"
        save_fhir_bundle([r1, r2], str(out))

        data = json.loads(out.read_text("utf-8"))
        assert data["resourceType"] == "Bundle"
        assert data["type"] == "collection"
        assert data["total"] == 2
        assert len(data["entry"]) == 2

    def test_bundle_entries_have_resources(self, tmp_path, sample_result):
        r = map_to_fhir_loose(sample_result)
        out = tmp_path / "bundle.json"
        save_fhir_bundle([r], str(out))

        data = json.loads(out.read_text("utf-8"))
        assert data["entry"][0]["resource"]["resourceType"] == "DiagnosticReport"
        assert data["entry"][0]["fullUrl"].startswith("urn:uuid:")

    def test_returns_absolute_path(self, tmp_path, sample_result):
        r = map_to_fhir_loose(sample_result)
        out = tmp_path / "bundle.json"
        result = save_fhir_bundle([r], str(out))
        assert result.endswith("bundle.json")

    def test_custom_bundle_type(self, tmp_path, sample_result):
        r = map_to_fhir_loose(sample_result)
        out = tmp_path / "bundle.json"
        save_fhir_bundle([r], str(out), bundle_type="document")
        data = json.loads(out.read_text("utf-8"))
        assert data["type"] == "document"

    def test_empty_bundle(self, tmp_path):
        out = tmp_path / "empty.json"
        save_fhir_bundle([], str(out))
        data = json.loads(out.read_text("utf-8"))
        assert data["total"] == 0
