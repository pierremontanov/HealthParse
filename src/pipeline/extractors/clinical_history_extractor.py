"""Rule-based extractor for clinical history / visit record documents.

Parses the structured text layout produced by the document generator and
returns a dictionary compatible with the ``ClinicalHistorySchema`` Pydantic
schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.extractors.base import (
    extract_block,
    extract_date,
    extract_dated_entries,
    extract_field,
)


class ClinicalHistoryExtractor:
    """Extract structured fields from a clinical history document.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.

    Expected text layout::

        Patient Name: <name>
        Patient ID: <id>
        Date of Birth: <date>
        Clinic: <name>

        Annotations:
        - YYYY-MM-DD: <note text>
        - YYYY-MM-DD: <note text>
    """

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary that matches the ``ClinicalHistorySchema``."""
        patient_name = extract_field(text, "Patient Name")
        patient_id = extract_field(text, "Patient ID")
        date_of_birth = extract_date(text, "Date of Birth")
        institution = extract_field(text, "Clinic") or extract_field(text, "Institution")
        doctor_name = (
            extract_field(text, "Doctor")
            or extract_field(text, "Physician")
            or extract_field(text, "Professional")
        )

        # ── Parse annotations ──
        annotations_block = extract_block(text, "Annotations")
        entries = extract_dated_entries(annotations_block) if annotations_block else []

        consultation_date = self._derive_consultation_date(entries, text)
        medical_history = self._build_medical_history(entries)
        chief_complaint = self._derive_chief_complaint(entries)

        # ── Try alternative fields for richer documents ──
        assessment = extract_field(text, "Assessment") or extract_field(text, "Diagnosis")
        plan = extract_field(text, "Plan") or extract_field(text, "Treatment Plan")
        physical_exam = extract_field(text, "Physical Exam") or extract_field(text, "Examination")
        chief_complaint_explicit = extract_field(text, "Chief Complaint") or extract_field(text, "Reason for Visit")
        medications_field = extract_field(text, "Current Medications") or extract_field(text, "Medications")

        current_medications: Optional[List[str]] = None
        if medications_field:
            current_medications = [
                m.strip() for m in medications_field.replace(";", ",").split(",") if m.strip()
            ]

        return {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "date_of_birth": date_of_birth,
            "consultation_date": consultation_date,
            "chief_complaint": chief_complaint_explicit or chief_complaint,
            "medical_history": medical_history,
            "current_medications": current_medications,
            "physical_exam": physical_exam,
            "assessment": assessment,
            "plan": plan,
            "doctor_name": doctor_name or "Not specified",
            "institution": institution,
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _derive_consultation_date(
        entries: list,
        text: str,
    ) -> str:
        """Derive the consultation date from annotations or fallback fields.

        Uses the most recent annotation date.  Falls back to explicit date
        fields or today's date as a last resort.
        """
        if entries:
            dates = [e[0] for e in entries]
            dates.sort(reverse=True)
            return dates[0]

        for key in ("Consultation Date", "Visit Date", "Date"):
            val = extract_date(text, key)
            if val:
                return val

        return "Unknown"

    @staticmethod
    def _build_medical_history(entries: list) -> Optional[str]:
        """Concatenate all annotation notes into a medical history narrative."""
        if not entries:
            return None
        lines = [f"{date}: {note}" for date, note in entries]
        return "\n".join(lines)

    @staticmethod
    def _derive_chief_complaint(entries: list) -> Optional[str]:
        """Use the most recent annotation note as the chief complaint."""
        if not entries:
            return None
        sorted_entries = sorted(entries, key=lambda e: e[0], reverse=True)
        return sorted_entries[0][1]
