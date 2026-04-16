"""Validation functions for each document type.

Each function accepts a raw ``dict``, normalises date fields via the
:func:`~src.pipeline.utils.date_utils.normalize_dates` decorator, and
returns a validated Pydantic model instance.  On failure the original
:class:`pydantic.ValidationError` is re-raised after being logged with
the document type for easier debugging by API consumers.

The :func:`validate_output` function provides a pre-export safety gate
that re-validates ``extracted_data`` against the correct schema before
data leaves the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.clinical_history_schema import ClinicalHistorySchema
from src.pipeline.utils.date_utils import normalize_dates

logger = logging.getLogger(__name__)

# Schema registry: maps document_type strings to their Pydantic model.
SCHEMA_REGISTRY: Dict[str, type] = {
    "result": ResultSchema,
    "prescription": Prescription,
    "clinical_history": ClinicalHistorySchema,
}


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


# ── Pre-export validation gate ───────────────────────────────────

@dataclass
class OutputValidationResult:
    """Result of a single-document output validation check."""

    file: str
    valid: bool
    document_type: Optional[str] = None
    error: Optional[str] = None


def validate_output(item: Dict[str, Any]) -> OutputValidationResult:
    """Re-validate a result dict's ``extracted_data`` against its schema.

    This is intended as a safety gate before export.  It confirms that the
    data stored in ``extracted_data`` still conforms to the Pydantic schema
    for its ``document_type``.

    Parameters
    ----------
    item : dict
        A result dict as produced by :meth:`DocumentResult.as_dict`.

    Returns
    -------
    OutputValidationResult
        ``valid=True`` if the data passes (or if there is nothing to
        validate), ``valid=False`` with an error message otherwise.
    """
    filename = item.get("file", "unknown")
    doc_type = item.get("document_type")
    data = item.get("extracted_data")

    # Nothing to validate — extraction-only or failed earlier
    if not doc_type or not data:
        return OutputValidationResult(file=filename, valid=True, document_type=doc_type)

    schema_cls = SCHEMA_REGISTRY.get(doc_type)
    if schema_cls is None:
        # Unknown document type — can't validate, let it through
        return OutputValidationResult(
            file=filename,
            valid=True,
            document_type=doc_type,
            error=f"No schema registered for '{doc_type}'; skipping validation.",
        )

    try:
        schema_cls(**data)
        return OutputValidationResult(file=filename, valid=True, document_type=doc_type)
    except ValidationError as exc:
        msg = f"{exc.error_count()} validation error(s): {exc}"
        logger.warning("Pre-export validation failed for %s (%s): %s", filename, doc_type, msg)
        return OutputValidationResult(
            file=filename, valid=False, document_type=doc_type, error=msg,
        )


def validate_batch(
    items: List[Dict[str, Any]],
    *,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Validate a batch of result dicts and optionally strip invalid ones.

    Parameters
    ----------
    items : list[dict]
        Result dicts to check.
    strict : bool
        When ``True``, items that fail validation are **excluded** from
        the returned list and a warning is logged.  When ``False``
        (default), all items are returned but each gets an
        ``_validation_error`` key when it fails.

    Returns
    -------
    list[dict]
        The (possibly filtered) list of result dicts.
    """
    checked: List[Dict[str, Any]] = []
    for item in items:
        result = validate_output(item)
        if result.valid:
            checked.append(item)
        elif strict:
            logger.warning(
                "Dropping '%s' from export: %s", result.file, result.error,
            )
        else:
            item = dict(item)  # shallow copy to avoid mutating original
            item["_validation_error"] = result.error
            checked.append(item)
    return checked
