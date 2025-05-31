import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.fhir_mapper import map_to_fhir_loose
from pipeline.validation.schemas import ResultSchema
from pipeline.validation.prescription_schema import Prescription
from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema


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
        notes="Requiere seguimiento clínico."
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
                "duration": "30 días"
            }
        ]
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
        institution="Centro Médico Colonial"
    )


def test_result_to_fhir_fields(result_schema_sample):
    fhir = map_to_fhir_loose(result_schema_sample)
    assert fhir["resourceType"] == "DiagnosticReport"
    assert fhir["subject"]["name"] == result_schema_sample.patient_name
    assert fhir["effectiveDateTime"] == "2024-06-01"
    assert fhir["conclusion"] == result_schema_sample.impression


def test_prescription_to_fhir_fields(prescription_sample):
    fhir = map_to_fhir_loose(prescription_sample)
    assert fhir["resourceType"] == "MedicationRequest"
    assert fhir["subject"]["identifier"]["value"] == prescription_sample.patient_id
    assert fhir["contained"][0]["resourceType"] == "Medication"


def test_clinical_history_to_fhir_fields(clinical_history_sample):
    fhir = map_to_fhir_loose(clinical_history_sample)
    assert fhir["resourceType"] == "Encounter"
    assert fhir["subject"]["display"] == clinical_history_sample.patient_name
    assert fhir["period"]["start"] == "2024-05-10"
    assert fhir["note"][0]["text"] == clinical_history_sample.plan
