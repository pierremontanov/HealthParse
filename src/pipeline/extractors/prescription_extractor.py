"""Rule-based extractor for prescription documents.

Parses the structured text layout produced by the document generator and
returns a dictionary compatible with the ``Prescription`` Pydantic schema.

Uses the unified bilingual helpers from ``base.py`` so that English and
Spanish (Chile, Colombia, Mexico, etc.) documents are handled identically.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.pipeline.extractors.base import (
    extract_block,
    # Unified bilingual helpers
    extract_patient_name,
    extract_patient_id,
    extract_prescription_date,
    extract_doctor,
    extract_institution,
)


class PrescriptionExtractor:
    """Extract structured fields from a prescription document.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.
    """

    # Spanish block headers for the prescription body.
    # Accent-flex matching in extract_block() handles accented variants.
    _PRESCRIPTION_HEADERS = [
        "Prescription",
        "Prescripcion",
        "Formulacion",
        "Receta",
        "Medicamentos",
        "Tratamiento",
        "Indicaciones",
        "Ordenes Medicas",
        "Formula Medica",
        "Medicacion",
        "Farmacoterapia",
    ]

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary that matches the ``Prescription`` schema."""
        patient_name = extract_patient_name(text)
        patient_id = extract_patient_id(text)
        date = extract_prescription_date(text)
        doctor_name = extract_doctor(text)
        institution = extract_institution(text)

        # Try all known prescription block headers
        prescription_body = None
        for header in self._PRESCRIPTION_HEADERS:
            prescription_body = extract_block(text, header)
            if prescription_body:
                break

        items = self._parse_items(prescription_body)

        return {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "date": date,
            "doctor_name": doctor_name,
            "institution": institution,
            "additional_notes": None,
            "items": items,
            "raw_text": text.strip(),
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

        # ── Medicine patterns (English + Spanish) ──
        med_patterns = [
            # English: "Ibuprofen 400mg, 3 times daily"
            r"(?i)([\w\s]+?)\s+(\d+\s*(?:mg|ml|g|mcg|iu|units?))\s*[,;]?\s*"
            r"(\d+\s*(?:times?|x)\s*(?:daily|a day|per day|weekly)?)?",
            r"(?i)([\w\s]+?)\s+(\d+\s*(?:mg|ml|g))\s+(?:every|each)\s+(\d+\s*(?:hours?|hrs?))",
            # Spanish: "Ibuprofeno 400mg cada 8 horas"
            r"(?i)([\w\s]+?)\s+(\d+\s*(?:mg|ml|g|mcg|ui|unidades?))\s*[,;]?\s*"
            r"(?:cada\s+(\d+\s*(?:horas?|hrs?|dias?|semanas?)))?",
            # Spanish: "Tomar 1 tableta cada 12 horas"
            r"(?i)([\w\s]+?)\s+(\d+\s*(?:tabletas?|comprimidos?|capsulas?|gotas?|ampollas?|sobres?))"
            r"\s*[,;]?\s*(?:cada\s+(\d+\s*(?:horas?|hrs?|dias?)))?",
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

        # ── Keyword-based classification (English + Spanish) ──
        if any(kw in lower for kw in [
            "x-ray", "xray", "mri", "ct scan", "ultrasound", "imaging", "radiograph",
            "radiografia", "ecografia", "tomografia", "resonancia", "mamografia",
        ]):
            return {"type": "radiology", "name": line, "modality": None, "body_part": None, "notes": None}

        if any(kw in lower for kw in [
            "blood test", "urinalysis", "cbc", "lipid panel", "lab test", "glucose test",
            "examen de sangre", "hemograma", "perfil lipidico", "glicemia", "urocultivo",
            "examen de orina", "laboratorio",
        ]):
            return {"type": "lab_test", "name": line, "test_type": None, "parameters": None, "notes": None}

        if any(kw in lower for kw in [
            "refer to", "referral", "consult with", "specialist",
            "remitir", "interconsulta", "derivar", "especialista", "valoracion por",
        ]):
            return {"type": "specialist", "name": line, "specialty": None, "reason": None, "notes": None}

        if any(kw in lower for kw in [
            "therapy", "physiotherapy", "rehabilitation", "exercise",
            "terapia", "fisioterapia", "rehabilitacion", "ejercicio", "kinesioterapia",
        ]):
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
