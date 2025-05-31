
from pydantic import ValidationError
from pipeline.validation.schemas import ResultSchema
from pipeline.validation.prescription_schema import Prescription


def validate_result_schema(data: dict) -> ResultSchema:
    """
    Validates a dictionary against the ResultSchema.

    Args:
        data (dict): Raw extracted fields.

    Returns:
        ResultSchema: A validated ResultSchema object.

    Raises:
        ValidationError: If any field is missing or has an invalid format.
    """
    try:
        validated = ResultSchema(**data)
        return validated
    except ValidationError as e:
        print("❌ Validation error in ResultSchema:")
        print(e.json(indent=2))
        raise

def validate_prescription(data: dict) -> Prescription:
    """
    Validate input data against the Prescription schema.
    Raises ValidationError if data is invalid.
    """
    try:
        return Prescription(**data)
    except ValidationError as e:
        print("❌ Prescription validation failed:\n", e.json(indent=2))
        raise


from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema

def validate_clinical_history(data: dict) -> ClinicalHistorySchema:
    """
    Validate input data against the ClinicalHistory schema.
    Raises ValidationError if data is invalid.
    """
    try:
        return ClinicalHistorySchema(**data)
    except ValidationError as e:
        print("❌ Clinical history validation failed:\n", e.json(indent=2))
        raise
