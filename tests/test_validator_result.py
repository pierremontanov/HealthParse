"""Tests for ResultSchema validation."""

import pytest
from pydantic import ValidationError

from src.pipeline.validation.validator import validate_result_schema
from src.pipeline.validation.schemas import ResultSchema


@pytest.fixture
def valid_result_data():
    return {
        "patient_name": "Gloria Ines Montaño Villada",
        "patient_id": "24314628",
        "age": 71,
        "sex": "F",
        "date_of_birth": "27-04-1953",
        "exam_type": "CR",
        "study_area": "Columna Dorsal",
        "exam_date": "08-08-2024",
        "findings": "Cambios incipientes de tipo degenerativo crónico.",
        "impression": "Alineamiento satisfactorio. Tejidos blandos normales.",
        "professional": "Dra. Fátima Mota Arteaga",
        "institution": "Centro Médico San José",
        "notes": "No se evidencian fracturas.",
    }


class TestResultSchemaValid:
    def test_returns_result_schema_instance(self, valid_result_data):
        validated = validate_result_schema(valid_result_data)
        assert isinstance(validated, ResultSchema)

    def test_patient_name_preserved(self, valid_result_data):
        validated = validate_result_schema(valid_result_data)
        assert validated.patient_name == "Gloria Ines Montaño Villada"

    def test_date_normalised_to_iso(self, valid_result_data):
        validated = validate_result_schema(valid_result_data)
        assert validated.date_of_birth == "1953-04-27"
        assert validated.exam_date == "2024-08-08"

    def test_iso_dates_passed_through(self, valid_result_data):
        valid_result_data["date_of_birth"] = "1953-04-27"
        valid_result_data["exam_date"] = "2024-08-08"
        validated = validate_result_schema(valid_result_data)
        assert validated.date_of_birth == "1953-04-27"

    def test_optional_fields_can_be_none(self):
        minimal = {
            "patient_name": "Test Patient",
            "exam_type": "CBC",
            "exam_date": "2024-01-01",
            "findings": "Normal",
            "professional": "Dr. Test",
            "institution": "Test Lab",
        }
        validated = validate_result_schema(minimal)
        assert validated.patient_id is None
        assert validated.impression is None
        assert validated.notes is None

    def test_sex_accepts_valid_values(self, valid_result_data):
        for sex in ["M", "F", "Other"]:
            valid_result_data["sex"] = sex
            validated = validate_result_schema(valid_result_data)
            assert validated.sex == sex


class TestResultSchemaInvalid:
    def test_missing_patient_name_raises(self, valid_result_data):
        del valid_result_data["patient_name"]
        with pytest.raises(ValidationError):
            validate_result_schema(valid_result_data)

    def test_missing_findings_raises(self, valid_result_data):
        del valid_result_data["findings"]
        with pytest.raises(ValidationError):
            validate_result_schema(valid_result_data)

    def test_invalid_sex_raises(self, valid_result_data):
        valid_result_data["sex"] = "X"
        with pytest.raises(ValidationError, match="Invalid sex value"):
            validate_result_schema(valid_result_data)

    def test_extra_field_raises(self, valid_result_data):
        valid_result_data["unknown_field"] = "surprise"
        with pytest.raises(ValidationError):
            validate_result_schema(valid_result_data)

    def test_age_negative_raises(self, valid_result_data):
        valid_result_data["age"] = -5
        with pytest.raises(ValidationError):
            validate_result_schema(valid_result_data)

    def test_empty_patient_name_raises(self, valid_result_data):
        valid_result_data["patient_name"] = ""
        with pytest.raises(ValidationError):
            validate_result_schema(valid_result_data)
