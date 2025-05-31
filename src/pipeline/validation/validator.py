from pydantic import ValidationError
from pipeline.validation.schemas import ResultSchema
from pipeline.validation.prescription_schema import Prescription
from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema
from pipeline.utils.date_utils import normalize_dates

@normalize_dates("date_of_birth", "exam_date")
def validate_result_schema(data: dict) -> ResultSchema:
    try:
        return ResultSchema(**data)
    except ValidationError as e:
        print("❌ Validation error in ResultSchema:")
        print(e.json(indent=2))
        raise

@normalize_dates("date")
def validate_prescription(data: dict) -> Prescription:
    try:
        return Prescription(**data)
    except ValidationError as e:
        print("❌ Prescription validation failed:\n", e.json(indent=2))
        raise

@normalize_dates("date_of_birth", "consultation_date")
def validate_clinical_history(data: dict) -> ClinicalHistorySchema:
    try:
        return ClinicalHistorySchema(**data)
    except ValidationError as e:
        print("❌ Clinical history validation failed:\n", e.json(indent=2))
        raise
