"""Tests for Prescription schema validation."""

import pytest
from pydantic import ValidationError

from src.pipeline.validation.validator import validate_prescription
from src.pipeline.validation.prescription_schema import Prescription


@pytest.fixture
def valid_prescription_data():
    return {
        "patient_name": "Carlos Ruiz",
        "patient_id": "88997766",
        "date": "2024-05-25",
        "doctor_name": "Dr. Andrés López",
        "institution": "Clínica Central",
        "additional_notes": "Paciente con hipertensión controlada.",
        "items": [
            {
                "type": "medicine",
                "name": "Losartán 50mg",
                "dosage": "50mg",
                "frequency": "1 vez al día",
                "route": "oral",
                "duration": "30 días",
                "notes": "Tomar en la mañana",
            }
        ],
    }


class TestPrescriptionValid:
    def test_returns_prescription_instance(self, valid_prescription_data):
        result = validate_prescription(valid_prescription_data)
        assert isinstance(result, Prescription)

    def test_patient_name_preserved(self, valid_prescription_data):
        result = validate_prescription(valid_prescription_data)
        assert result.patient_name == "Carlos Ruiz"

    def test_medicine_item_parsed(self, valid_prescription_data):
        result = validate_prescription(valid_prescription_data)
        item = result.items[0]
        assert item.type == "medicine"
        assert item.name == "Losartán 50mg"
        assert item.dosage == "50mg"
        assert item.route == "oral"

    def test_multiple_item_types(self, valid_prescription_data):
        valid_prescription_data["items"].append(
            {"type": "lab_test", "name": "CBC", "test_type": "blood panel", "parameters": ["WBC", "RBC"]}
        )
        valid_prescription_data["items"].append(
            {"type": "specialist", "name": "Cardiology Referral", "specialty": "Cardiology", "reason": "Hypertension"}
        )
        result = validate_prescription(valid_prescription_data)
        assert len(result.items) == 3
        assert result.items[1].type == "lab_test"
        assert result.items[2].type == "specialist"

    def test_generic_item(self, valid_prescription_data):
        valid_prescription_data["items"] = [
            {"type": "other", "name": "Bed rest for 3 days"}
        ]
        result = validate_prescription(valid_prescription_data)
        assert result.items[0].type == "other"

    def test_date_normalised_from_ddmmyyyy(self):
        data = {
            "patient_name": "Test",
            "date": "25-05-2024",
            "items": [{"type": "other", "name": "Rest"}],
        }
        result = validate_prescription(data)
        assert result.date == "2024-05-25"

    def test_iso_date_passed_through(self, valid_prescription_data):
        result = validate_prescription(valid_prescription_data)
        assert result.date == "2024-05-25"

    def test_all_fields_optional_except_items(self):
        minimal = {"items": [{"type": "other", "name": "Vitamins"}]}
        result = validate_prescription(minimal)
        assert result.patient_name is None
        assert result.doctor_name is None
        assert len(result.items) == 1


class TestPrescriptionInvalid:
    def test_empty_items_raises(self, valid_prescription_data):
        valid_prescription_data["items"] = []
        with pytest.raises(ValidationError):
            validate_prescription(valid_prescription_data)

    def test_missing_items_raises(self, valid_prescription_data):
        del valid_prescription_data["items"]
        with pytest.raises(ValidationError):
            validate_prescription(valid_prescription_data)

    def test_invalid_item_type_raises(self, valid_prescription_data):
        valid_prescription_data["items"] = [
            {"type": "INVALID", "name": "Something"}
        ]
        with pytest.raises(ValidationError):
            validate_prescription(valid_prescription_data)

    def test_item_missing_name_raises(self, valid_prescription_data):
        valid_prescription_data["items"] = [{"type": "other"}]
        with pytest.raises(ValidationError):
            validate_prescription(valid_prescription_data)
