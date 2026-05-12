"""Rule-based extractor for laboratory result / imaging report documents.

Parses the structured text layout produced by the document generator and
returns a dictionary compatible with the ``ResultSchema`` Pydantic schema.

Uses the unified bilingual helpers from ``base.py`` so that English and
Spanish (Chile, Colombia, Mexico, etc.) documents are handled identically.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.pipeline.extractors.base import (
    extract_block,
    extract_test_results,
    # Unified bilingual helpers
    extract_patient_name,
    extract_patient_id,
    extract_exam_date,
    extract_birth_date,
    extract_doctor,
    extract_institution,
    extract_age,
    extract_sex,
    extract_findings_block,
    extract_impression_block,
    extract_study_area,
    infer_exam_type,
)


class LabResultExtractor:
    """Extract structured fields from a lab result or imaging report document.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.
    """

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary that matches the ``ResultSchema`` schema."""
        patient_name = extract_patient_name(text)
        patient_id = extract_patient_id(text)
        date_of_birth = extract_birth_date(text)
        exam_date = extract_exam_date(text)
        institution = extract_institution(text)
        professional = extract_doctor(text)
        age = extract_age(text)
        sex = extract_sex(text)

        # ── Extract test results block (English and Spanish headers) ──
        results_block = extract_block(text, "Test Results")
        if not results_block:
            results_block = extract_block(text, "Resultados")
        test_results = extract_test_results(results_block) if results_block else []

        # ── Findings ──
        findings = self._format_findings(test_results, results_block)
        if findings == "No findings recorded":
            findings = extract_findings_block(text) or findings

        # ── Exam type & study area ──
        exam_type = infer_exam_type(text, test_results or None)
        study_area = extract_study_area(text)

        # ── Summary / Impression ──
        summary = extract_impression_block(text)

        return {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "date_of_birth": date_of_birth,
            "exam_date": exam_date,
            "exam_type": exam_type,
            "study_area": study_area,
            "findings": findings,
            "impression": summary,
            "professional": professional,
            "institution": institution,
            "notes": None,
            "raw_text": text.strip(),
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _format_findings(
        test_results: list,
        raw_block: Optional[str],
    ) -> str:
        """Build a human-readable findings string from parsed test results."""
        if test_results:
            lines = []
            for tr in test_results:
                lines.append(
                    f"{tr['test_name']}: {tr['value']} (Reference: {tr['reference_range']})"
                )
            return "; ".join(lines)
        if raw_block:
            return raw_block.strip()
        return "No findings recorded"
