"""Rule-based extractor for clinical history / visit record documents.

Parses the structured text layout produced by the document generator and
returns a dictionary compatible with the ``ClinicalHistorySchema`` Pydantic
schema.

Uses the unified bilingual helpers from ``base.py`` so that English and
Spanish (Chile, Colombia, Mexico, etc.) documents are handled identically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.extractors.base import (
    extract_block,
    extract_dated_entries,
    extract_field,
    resolve_field_flexible,
    # Unified bilingual helpers
    extract_patient_name,
    extract_patient_id,
    extract_birth_date,
    extract_consultation_date,
    extract_doctor,
    extract_institution,
    extract_age,
    extract_sex,
)


class ClinicalHistoryExtractor:
    """Extract structured fields from a clinical history document.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.
    """

    # ── Spanish-aware field alias lists ──
    # Only unaccented forms needed: accent-flex matching in base.py
    # handles both "Diagnóstico" and "Diagnostico" automatically.
    _ASSESSMENT_ALIASES = [
        "Assessment", "Diagnosis",
        "Impresion Diagnostica",
        "Diagnostico Principal",
        "Diagnostico",
        "Valoracion",
        "Evaluacion",
        "Dx",
        "DX",
    ]
    _PLAN_ALIASES = [
        "Plan", "Treatment Plan",
        "Plan de Tratamiento", "Plan Terapeutico",
        "Conducta", "Manejo",
        "Ordenes Medicas",
        "Recomendaciones",
    ]
    _PHYSICAL_EXAM_ALIASES = [
        "Physical Exam", "Examination",
        "Examen Fisico", "Exploracion Fisica", "Examen Clinico",
        "Revision por Sistemas",
        "Signos Vitales",
    ]
    _CHIEF_COMPLAINT_ALIASES = [
        "Chief Complaint", "Reason for Visit",
        "Motivo de Consulta", "Motivo Consulta", "Motivo de Ingreso",
        "Queja Principal", "Sintoma Principal",
        "Enfermedad Actual",
        "Enfermedad actual",
        "Causa Externa",
    ]
    _MEDICATIONS_ALIASES = [
        "Current Medications", "Medications",
        "Medicamentos", "Medicamentos Actuales", "Tratamiento Actual",
        "Medicacion", "Farmacoterapia",
        "Farmacologicos", "Farmacologicas",
        "Tratamiento Farmacologico",
    ]
    _MEDICAL_HISTORY_ALIASES = [
        "Medical History", "Past Medical History",
        "Antecedentes Patologicos", "Patologicos",
        "Antecedentes Personales", "Antecedentes",
        "Antecedentes Familiares",
        "Quirurgicos",
        "Antecedentes Quirurgicos",
        "Alergicos",
        "Antecedentes Alergicos",
        "Toxicologicos",
    ]
    _ANNOTATIONS_HEADERS = [
        "Annotations",
        "Anotaciones", "Notas Clinicas", "Evoluciones", "Evolucion",
        "Historia Clinica", "Antecedentes",
        "Subjetivo",                    # SOAP format
        "Objetivo",
    ]

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary that matches the ``ClinicalHistorySchema``."""
        patient_name = extract_patient_name(text)
        patient_id = extract_patient_id(text)
        date_of_birth = extract_birth_date(text)
        institution = extract_institution(text)
        doctor_name = extract_doctor(text)
        age = extract_age(text)
        sex = extract_sex(text)

        # ── Parse annotations (English + Spanish headers) ──
        annotations_block = None
        for header in self._ANNOTATIONS_HEADERS:
            annotations_block = extract_block(text, header)
            if annotations_block:
                break

        entries = extract_dated_entries(annotations_block) if annotations_block else []

        consultation_date = self._derive_consultation_date(entries, text)
        medical_history = self._build_medical_history(entries)
        chief_complaint = self._derive_chief_complaint(entries)

        # ── Try alternative fields for richer documents ──
        assessment = resolve_field_flexible(text, self._ASSESSMENT_ALIASES)
        plan = resolve_field_flexible(text, self._PLAN_ALIASES)
        physical_exam = resolve_field_flexible(text, self._PHYSICAL_EXAM_ALIASES)
        chief_complaint_explicit = resolve_field_flexible(text, self._CHIEF_COMPLAINT_ALIASES)
        medications_field = resolve_field_flexible(text, self._MEDICATIONS_ALIASES)

        current_medications: Optional[List[str]] = None
        if medications_field:
            current_medications = [
                m.strip() for m in medications_field.replace(";", ",").split(",") if m.strip()
            ]

        # ── Fallback: medical history from "Patologicos" / "Antecedentes" ──
        if not medical_history:
            medical_history = resolve_field_flexible(text, self._MEDICAL_HISTORY_ALIASES)

        return {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "date_of_birth": date_of_birth,
            "consultation_date": consultation_date,
            "chief_complaint": chief_complaint_explicit or chief_complaint,
            "medical_history": medical_history,
            "current_medications": current_medications,
            "physical_exam": physical_exam,
            "assessment": assessment,
            "plan": plan,
            "doctor_name": doctor_name,
            "institution": institution,
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _derive_consultation_date(
        entries: list,
        text: str,
    ) -> str:
        """Derive the consultation date from annotations or the unified helper."""
        if entries:
            dates = [e[0] for e in entries]
            dates.sort(reverse=True)
            return dates[0]

        val = extract_consultation_date(text)
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
