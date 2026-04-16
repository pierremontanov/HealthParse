"""Rule-based entity extractors for clinical document types.

Each extractor implements an ``extract(text) -> dict`` interface compatible
with the :class:`~src.pipeline.inference.InferenceEngine`.
"""

from src.pipeline.extractors.clinical_history_extractor import ClinicalHistoryExtractor
from src.pipeline.extractors.field_aliases import (
    resolve_assessment,
    resolve_chief_complaint,
    resolve_doctor,
    resolve_exam_date,
    resolve_institution,
    resolve_medications,
    resolve_physical_exam,
    resolve_plan,
    resolve_prescription_date,
)
from src.pipeline.extractors.prescription_extractor import PrescriptionExtractor
from src.pipeline.extractors.result_extractor import LabResultExtractor

__all__ = [
    "PrescriptionExtractor",
    "LabResultExtractor",
    "ClinicalHistoryExtractor",
    # Field alias resolvers
    "resolve_assessment",
    "resolve_chief_complaint",
    "resolve_doctor",
    "resolve_exam_date",
    "resolve_institution",
    "resolve_medications",
    "resolve_physical_exam",
    "resolve_plan",
    "resolve_prescription_date",
]
