"""Tests for ClinicalHistorySchema validation."""

import pytest
from pydantic import ValidationError

from src.pipeline.validation.validator import validate_clinical_history
from src.pipeline.validation.clinical_history_schema import ClinicalHistorySchema


@pytest.fixture
def valid_clinical_data():
    return {
        "patient_name": "Laura Gómez",
        "patient_id": "123456789",
        "age": 52,
        "sex": "F",
        "date_of_birth": "1972-01-15",
        "consultation_date": "2024-05-30",
        "chief_complaint": "Dolor abdominal",
        "medical_history": "Hipertensión, gastritis",
        "current_medications": ["Omeprazol", "Losartán"],
        "physical_exam": "Dolor en hipogastrio, sin fiebre",
        "assessment": "Gastritis crónica",
        "plan": "Dieta blanda, seguimiento en 1 semana",
        "doctor_name": "Dra. Mariana Ríos",
        "institution": "Hospital General",
    }


class TestClinicalHistoryValid:
    def test_returns_schema_instance(self, valid_clinical_data):
        validated = validate_clinical_history(valid_clinical_data)
        assert isinstance(validated, ClinicalHistorySchema)

    def test_patient_name_preserved(self, valid_clinical_data):
        validated = validate_clinical_history(valid_clinical_data)
        assert validated.patient_name == "Laura Gómez"

    def test_doctor_name_preserved(self, valid_clinical_data):
        validated = validate_clinical_history(valid_clinical_data)
        assert validated.doctor_name.startswith("Dra.")

    def test_medications_list_preserved(self, valid_clinical_data):
        validated = validate_clinical_history(valid_clinical_data)
        assert validated.current_medications == ["Omeprazol", "Losartán"]

    def test_date_normalised_from_ddmmyyyy(self):
        data = {
            "patient_name": "Test",
            "date_of_birth": "15-01-1972",
            "consultation_date": "30-05-2024",
            "doctor_name": "Dr. Test",
        }
        validated = validate_clinical_history(data)
        assert validated.date_of_birth == "1972-01-15"
        assert validated.consultation_date == "2024-05-30"

    def test_iso_dates_passed_through(self, valid_clinical_data):
        validated = validate_clinical_history(valid_clinical_data)
        assert validated.date_of_birth == "1972-01-15"
        assert validated.consultation_date == "2024-05-30"

    def test_minimal_required_fields(self):
        minimal = {
            "patient_name": "Test",
            "consultation_date": "2024-01-01",
            "doctor_name": "Dr. Test",
        }
        validated = validate_clinical_history(minimal)
        assert validated.chief_complaint is None
        assert validated.medical_history is None
        assert validated.institution is None


class TestClinicalHistoryInvalid:
    def test_missing_patient_name_raises(self, valid_clinical_data):
        del valid_clinical_data["patient_name"]
        with pytest.raises(ValidationError):
            validate_clinical_history(valid_clinical_data)

    def test_missing_consultation_date_raises(self, valid_clinical_data):
        del valid_clinical_data["consultation_date"]
        with pytest.raises(ValidationError):
            validate_clinical_history(valid_clinical_data)

    def test_missing_doctor_name_raises(self, valid_clinical_data):
        del valid_clinical_data["doctor_name"]
        with pytest.raises(ValidationError):
            validate_clinical_history(valid_clinical_data)

    def test_extra_field_raises(self, valid_clinical_data):
        valid_clinical_data["random_field"] = "oops"
        with pytest.raises(ValidationError):
            validate_clinical_history(valid_clinical_data)

    def test_empty_patient_name_raises(self, valid_clinical_data):
        valid_clinical_data["patient_name"] = ""
        with pytest.raises(ValidationError):
            validate_clinical_history(valid_clinical_data)
