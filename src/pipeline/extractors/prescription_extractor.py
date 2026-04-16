"""Rule-based extractor for prescription documents.

Parses the structured text layout produced by the document generator and
returns a dictionary compatible with the ``Prescription`` Pydantic schema.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.pipeline.extractors.base import extract_block, extract_field
from src.pipeline.extractors.field_aliases import (
    resolve_institution,
    resolve_prescription_date,
)


class PrescriptionExtractor:
    """Extract structured fields from a prescription document.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.

    Expected text layout::

        Patient Name: <name>
        Patient ID: <id>
        Date of Birth: <date>
        Date of Prescription: <date>
        Doctor: <name>
        Clinic: <name>

        Prescription:
        <free-text prescription body>
    """

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary that matches the ``Prescription`` schema."""
        patient_name = extract_field(text, "Patient Name")
        patient_id = extract_field(text, "Patient ID")
        date = resolve_prescription_date(text)
        doctor_name = extract_field(text, "Doctor")
        institution = resolve_institution(text)

        prescription_body = extract_block(text, "Prescription")
        items = self._parse_items(prescription_body)

        return {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "date": date,
            "doctor_name": doctor_name,
            "institution": institution,
            "additional_notes": None,
            "items": items,
        }

    # ── Item parsing ────────────────────────────────────────────────

    def _parse_items(self, body: Optional[str]) -> List[Dict[str, Any]]:
        """Parse the free-text prescription body into structured items.

        The method applies a series of heuristics:
        1. If the body contains bullet/dash-delimited lines, treat each as a
           separate item.
        2. For each line, attempt to classify and extract structured fields
           using keyword matching and simple regex patterns.
        3. Fall back to a ``GenericItem`` when no specific type is matched.
        """
        if not body:
            return [{"type": "other", "name": "Prescription", "notes": None}]

        lines = self._split_into_lines(body)
        items: List[Dict[str, Any]] = []
        for line in lines:
            item = self._classify_and_extract(line)
            items.append(item)

        return items if items else [{"type": "other", "name": "Prescription", "notes": body}]

    @staticmethod
    def _split_into_lines(body: str) -> List[str]:
        """Split the prescription body into individual item lines."""
        parts = re.split(r"\n\s*[-*•]\s*", body)
        if len(parts) <= 1:
            parts = [p.strip() for p in body.split("\n") if p.strip()]
        return [p.strip() for p in parts if p.strip()]

    def _classify_and_extract(self, line: str) -> Dict[str, Any]:
        """Attempt to classify a single line as a prescription item type."""

        lower = line.lower()

        # ── Medicine patterns ──
        med_patterns = [
            r"(?i)([\w\s]+?)\s+(\d+\s*(?:mg|ml|g|mcg|iu|units?))\s*[,;]?\s*"
            r"(\d+\s*(?:times?|x)\s*(?:daily|a day|per day|weekly)?)?",
            r"(?i)([\w\s]+?)\s+(\d+\s*(?:mg|ml|g))\s+(?:every|each)\s+(\d+\s*(?:hours?|hrs?))",
        ]
        for pat in med_patterns:
            m = re.match(pat, line)
            if m:
                return {
                    "type": "medicine",
                    "name": m.group(1).strip(),
                    "dosage": m.group(2).strip() if m.lastindex >= 2 else None,
                    "frequency": m.group(3).strip() if m.lastindex >= 3 else None,
                    "route": None,
                    "duration": None,
                    "notes": line,
                }

        # ── Keyword-based classification ──
        if any(kw in lower for kw in ["x-ray", "xray", "mri", "ct scan", "ultrasound", "imaging", "radiograph"]):
            return {"type": "radiology", "name": line, "modality": None, "body_part": None, "notes": None}

        if any(kw in lower for kw in ["blood test", "urinalysis", "cbc", "lipid panel", "lab test", "glucose test"]):
            return {"type": "lab_test", "name": line, "test_type": None, "parameters": None, "notes": None}

        if any(kw in lower for kw in ["refer to", "referral", "consult with", "specialist"]):
            return {"type": "specialist", "name": line, "specialty": None, "reason": None, "notes": None}

        if any(kw in lower for kw in ["therapy", "physiotherapy", "rehabilitation", "exercise"]):
            return {
                "type": "procedure",
                "name": line,
                "therapy_type": None,
                "body_part": None,
                "frequency": None,
                "duration": None,
                "notes": None,
            }

        # ── Fallback: GenericItem ──
        return {"type": "other", "name": line, "notes": None}
