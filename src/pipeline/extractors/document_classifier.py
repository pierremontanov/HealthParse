"""Rule-based document type classifier.

Analyses extracted text to determine the clinical document type so that the
correct NER extractor can be selected by the inference engine.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ── Keyword scoring tables ──
# Each keyword contributes a score towards a document type.  The type with the
# highest aggregate score wins.

_PRESCRIPTION_KEYWORDS: Dict[str, float] = {
    "prescription": 3.0,
    "prescribed": 2.0,
    "doctor": 1.0,
    "dosage": 2.0,
    "frequency": 1.5,
    "medication": 2.5,
    "medicine": 2.0,
    "mg": 1.5,
    "ml": 1.0,
    "route": 1.0,
    "oral": 1.0,
    "topical": 1.0,
    "tablet": 2.0,
    "capsule": 2.0,
    "date of prescription": 3.0,
}

_RESULT_KEYWORDS: Dict[str, float] = {
    "test results": 3.0,
    "exam date": 3.0,
    "exam type": 2.0,
    "findings": 2.5,
    "impression": 2.0,
    "reference": 1.5,
    "ref:": 2.0,
    "summary": 1.0,
    "lab": 1.5,
    "result": 1.5,
    "blood": 1.0,
    "urine": 1.0,
    "specimen": 1.5,
    "hemoglobin": 2.0,
    "glucose": 2.0,
    "cholesterol": 2.0,
}

_CLINICAL_HISTORY_KEYWORDS: Dict[str, float] = {
    "annotations": 3.0,
    "clinic history": 3.0,
    "clinical history": 3.0,
    "consultation date": 3.0,
    "visit date": 2.0,
    "chief complaint": 2.5,
    "medical history": 2.5,
    "assessment": 2.0,
    "plan": 1.0,
    "physical exam": 2.0,
    "current medications": 2.0,
    "reason for visit": 2.0,
}

DOCUMENT_TYPES = {
    "prescription": _PRESCRIPTION_KEYWORDS,
    "result": _RESULT_KEYWORDS,
    "clinical_history": _CLINICAL_HISTORY_KEYWORDS,
}


class DocumentClassifier:
    """Classify a clinical document based on keyword scoring.

    The classifier exposes a ``predict(text)`` method so it can be registered
    as the classifier model inside a :class:`~src.pipeline.inference.ModelBundle`.
    """

    def __init__(self, min_score: float = 1.0) -> None:
        self._min_score = min_score

    def predict(self, text: str) -> Dict[str, str]:
        """Return ``{"document_type": "<type>"}`` or ``{}`` if no match."""
        doc_type = self.classify(text)
        if doc_type:
            return {"document_type": doc_type}
        return {}

    def classify(self, text: str) -> Optional[str]:
        """Return the most likely document type for *text*, or ``None``."""
        lower = text.lower()
        scores: Dict[str, float] = {}

        for doc_type, keywords in DOCUMENT_TYPES.items():
            total = 0.0
            for keyword, weight in keywords.items():
                count = len(re.findall(re.escape(keyword), lower))
                total += count * weight
            scores[doc_type] = total

        if not scores:
            return None

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best_type] >= self._min_score:
            return best_type

        return None
