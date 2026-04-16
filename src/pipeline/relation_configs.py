"""Domain-specific relation configurations for medical document entity wiring.

Each configuration maps **anchor** entity labels to their **dependent** labels.
The anchor is the primary concept (e.g. a medication name) and the dependents
are attributes that belong to it (e.g. dosage, frequency, route).

These configs are consumed by :class:`~src.pipeline.relation_mapper.RelationMapper`
to transform flat NER entity lists into structured, schema-compatible relations.

The three document-type configs mirror the validation schemas:
  • ``PRESCRIPTION_RELATIONS`` → :class:`~src.pipeline.validation.prescription_schema.Prescription`
  • ``RESULT_RELATIONS``       → :class:`~src.pipeline.validation.schemas.ResultSchema`
  • ``CLINICAL_HISTORY_RELATIONS`` → :class:`~src.pipeline.validation.ClinicalHistorySchema.ClinicalHistorySchema`
"""
from __future__ import annotations

from typing import Dict, Sequence

# ═══════════════════════════════════════════════════════════════════
# Prescription
# ═══════════════════════════════════════════════════════════════════

PRESCRIPTION_RELATIONS: Dict[str, Sequence[str]] = {
    # Medications: core prescription item
    "MEDICATION": ["DOSAGE", "FREQUENCY", "ROUTE", "DURATION", "NOTES"],
    # Radiology orders
    "RADIOLOGY": ["MODALITY", "BODY_PART", "NOTES"],
    # Lab test orders
    "LAB_TEST": ["TEST_TYPE", "PARAMETERS", "NOTES"],
    # Specialist referrals
    "SPECIALIST": ["SPECIALTY", "REASON", "NOTES"],
    # Therapy / procedure orders
    "THERAPY": ["THERAPY_TYPE", "BODY_PART", "FREQUENCY", "DURATION", "NOTES"],
}

# ═══════════════════════════════════════════════════════════════════
# Lab / Imaging Results
# ═══════════════════════════════════════════════════════════════════

RESULT_RELATIONS: Dict[str, Sequence[str]] = {
    # Individual test results (e.g. "Glucose" → value + range + unit)
    "TEST_NAME": ["TEST_VALUE", "REFERENCE_RANGE", "UNIT", "FLAG"],
    # Exam-level entities (e.g. "Blood Chemistry" → findings + impression)
    "EXAM_TYPE": ["FINDINGS", "IMPRESSION", "STUDY_AREA", "NOTES"],
}

# ═══════════════════════════════════════════════════════════════════
# Clinical History
# ═══════════════════════════════════════════════════════════════════

CLINICAL_HISTORY_RELATIONS: Dict[str, Sequence[str]] = {
    # Diagnosis / condition anchored to its context
    "DIAGNOSIS": ["DATE", "TREATMENT", "STATUS", "NOTES"],
    # Current medication list entries
    "MEDICATION": ["DOSAGE", "FREQUENCY", "INDICATION"],
    # Chief complaint with linked assessment
    "COMPLAINT": ["ONSET", "SEVERITY", "ASSESSMENT", "PLAN"],
}

# ═══════════════════════════════════════════════════════════════════
# Registry: document_type → relation config
# ═══════════════════════════════════════════════════════════════════

RELATION_CONFIG_REGISTRY: Dict[str, Dict[str, Sequence[str]]] = {
    "prescription": PRESCRIPTION_RELATIONS,
    "result": RESULT_RELATIONS,
    "clinical_history": CLINICAL_HISTORY_RELATIONS,
}


def get_relation_config(document_type: str) -> Dict[str, Sequence[str]]:
    """Return the relation config for a document type.

    Parameters
    ----------
    document_type:
        One of ``"prescription"``, ``"result"``, or ``"clinical_history"``.

    Returns
    -------
    dict
        Anchor → dependent label mapping.

    Raises
    ------
    KeyError
        If *document_type* has no registered config.
    """
    try:
        return RELATION_CONFIG_REGISTRY[document_type]
    except KeyError:
        raise KeyError(
            f"No relation config for document type '{document_type}'. "
            f"Available: {', '.join(RELATION_CONFIG_REGISTRY)}"
        ) from None


__all__ = [
    "PRESCRIPTION_RELATIONS",
    "RESULT_RELATIONS",
    "CLINICAL_HISTORY_RELATIONS",
    "RELATION_CONFIG_REGISTRY",
    "get_relation_config",
]
