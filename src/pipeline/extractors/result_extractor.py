"""Rule-based extractor for laboratory result / imaging report documents.

Parses the structured text layout produced by the document generator and
returns a dictionary compatible with the ``ResultSchema`` Pydantic schema.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.pipeline.extractors.base import (
    extract_block,
    extract_date,
    extract_field,
    extract_test_results,
)


class LabResultExtractor:
    """Extract structured fields from a lab result or imaging report document.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.

    Expected text layout::

        Patient Name: <name>
        Patient ID: <id>
        Date of Birth: <date>
        Exam Date: <date>
        Clinic: <name>

        Test Results:
        - TestName: value (Ref: range)
        - TestName: value (Ref: range)

        Summary: <text>
    """

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary that matches the ``ResultSchema`` schema."""
        patient_name = extract_field(text, "Patient Name")
        patient_id = extract_field(text, "Patient ID")
        date_of_birth = extract_date(text, "Date of Birth")
        exam_date = (
            extract_date(text, "Exam Date")
            or extract_date(text, "Date of Exam")
            or extract_date(text, "Date")
        )
        institution = extract_field(text, "Clinic") or extract_field(text, "Institution")
        professional = (
            extract_field(text, "Doctor")
            or extract_field(text, "Professional")
            or extract_field(text, "Physician")
        )

        # ── Extract test results block ──
        results_block = extract_block(text, "Test Results")
        test_results = extract_test_results(results_block) if results_block else []

        findings = self._format_findings(test_results, results_block)
        exam_type = self._infer_exam_type(test_results, text)
        summary = extract_field(text, "Summary")

        return {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "date_of_birth": date_of_birth,
            "exam_date": exam_date,
            "exam_type": exam_type,
            "study_area": None,
            "findings": findings,
            "impression": summary,
            "professional": professional or "Not specified",
            "institution": institution or "Not specified",
            "notes": None,
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

    @staticmethod
    def _infer_exam_type(test_results: list, text: str) -> str:
        """Attempt to infer the exam type from the test names or text content."""
        if test_results:
            names = [tr["test_name"].lower() for tr in test_results]
            all_names = " ".join(names)

            if any(kw in all_names for kw in ["glucose", "hba1c", "insulin"]):
                return "Blood Chemistry – Glucose Panel"
            if any(kw in all_names for kw in ["cholesterol", "ldl", "hdl", "triglyceride", "lipid"]):
                return "Blood Chemistry – Lipid Panel"
            if any(kw in all_names for kw in ["hemoglobin", "hematocrit", "platelet", "wbc", "rbc", "cbc"]):
                return "Hematology – Complete Blood Count"
            if any(kw in all_names for kw in ["creatinine", "bun", "urea", "kidney"]):
                return "Blood Chemistry – Renal Panel"
            if any(kw in all_names for kw in ["alt", "ast", "bilirubin", "liver"]):
                return "Blood Chemistry – Liver Panel"
            if any(kw in all_names for kw in ["tsh", "t3", "t4", "thyroid"]):
                return "Blood Chemistry – Thyroid Panel"

            return "Laboratory Test"

        lower = text.lower()
        if any(kw in lower for kw in ["x-ray", "xray", "radiograph"]):
            return "Radiology – X-Ray"
        if "mri" in lower or "magnetic resonance" in lower:
            return "Radiology – MRI"
        if "ct" in lower or "computed tomography" in lower:
            return "Radiology – CT Scan"
        if "ultrasound" in lower or "sonograph" in lower:
            return "Radiology – Ultrasound"

        return "Laboratory Test"
