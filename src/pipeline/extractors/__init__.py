"""Rule-based entity extractors for clinical document types.

Each extractor implements an ``extract(text) -> dict`` interface compatible
with the :class:`~src.pipeline.inference.InferenceEngine`.
"""

from src.pipeline.extractors.prescription_extractor import PrescriptionExtractor
from src.pipeline.extractors.result_extractor import LabResultExtractor
from src.pipeline.extractors.clinical_history_extractor import ClinicalHistoryExtractor

__all__ = [
    "PrescriptionExtractor",
    "LabResultExtractor",
    "ClinicalHistoryExtractor",
]
