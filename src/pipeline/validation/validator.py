"""Validation functions for each document type.

Each function accepts a raw ``dict``, normalises date fields via the
:func:`~src.pipeline.utils.date_utils.normalize_dates` decorator, and
returns a validated Pydantic model instance.  On failure the original
:class:`pydantic.ValidationError` is re-raised after being logged with
the document type for easier debugging by API consumers.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema
from src.pipeline.utils.date_utils import normalize_dates

logger = logging.getLogger(__name__)


@normalize_dates("date_of_birth", "exam_date")
def validate_result_schema(data: dict) -> ResultSchema:
    """Validate a lab/imaging result payload.

    Raises
    ------
    pydantic.ValidationError
        When required fields are missing or constraints are violated.
    """
    try:
        return ResultSchema(**data)
    except ValidationError as exc:
        logger.error(
            "ResultSchema validation failed (%d error(s)): %s",
            exc.error_count(),
            exc,
        )
        raise


@normalize_dates("date")
def validate_prescription(data: dict) -> Prescription:
    """Validate a prescription payload.

    Raises
    ------
    pydantic.ValidationError
        When required fields are missing or constraints are violated.
    """
    try:
        return Prescription(**data)
    except ValidationError as exc:
        logger.error(
            "Prescription validation failed (%d error(s)): %s",
            exc.error_count(),
            exc,
        )
        raise


@normalize_dates("date_of_birth", "consultation_date")
def validate_clinical_history(data: dict) -> ClinicalHistorySchema:
    """Validate a clinical history payload.

    Raises
    ------
    pydantic.ValidationError
        When required fields are missing or constraints are violated.
    """
    try:
        return ClinicalHistorySchema(**data)
    except ValidationError as exc:
        logger.error(
            "ClinicalHistorySchema validation failed (%d error(s)): %s",
            exc.error_count(),
            exc,
        )
        raise
