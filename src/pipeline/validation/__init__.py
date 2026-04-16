"""Pydantic schemas and validation functions for extracted document data.

Convenience imports::

    from src.pipeline.validation import (
        ResultSchema, Prescription, ClinicalHistorySchema,
        validate_result_schema, validate_prescription, validate_clinical_history,
        validate_batch,
    )
"""

from src.pipeline.validation.clinical_history_schema import ClinicalHistorySchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.validator import (
    SCHEMA_REGISTRY,
    OutputValidationResult,
    validate_batch,
    validate_clinical_history,
    validate_output,
    validate_prescription,
    validate_result_schema,
)

__all__ = [
    # Schemas
    "ResultSchema",
    "Prescription",
    "ClinicalHistorySchema",
    # Validators
    "validate_result_schema",
    "validate_prescription",
    "validate_clinical_history",
    "validate_output",
    "validate_batch",
    "OutputValidationResult",
    "SCHEMA_REGISTRY",
]
