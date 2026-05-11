"""Shared field-alias helpers for rule-based entity extractors.

This module now delegates to the comprehensive bilingual helpers in
``base.py``.  The public functions are kept for backward compatibility
so existing imports continue to work.

Usage
-----
    from src.pipeline.extractors.field_aliases import resolve_institution

    institution = resolve_institution(text)
"""
from __future__ import annotations

from typing import Optional

# Re-export unified helpers from base for backward compatibility
from src.pipeline.extractors.base import (
    extract_institution as resolve_institution,
    extract_doctor as resolve_doctor,
    extract_exam_date as resolve_exam_date,
    extract_prescription_date as resolve_prescription_date,
    resolve_field_flexible,
    extract_field,
)


def resolve_assessment(text: str) -> Optional[str]:
    """Resolve assessment from ``Assessment`` / ``Diagnosis`` / Spanish equivalents."""
    return resolve_field_flexible(text, [
        "Assessment", "Diagnosis",
        "Impresion Diagnostica", "Diagnostico Principal",
        "Diagnostico", "Valoracion", "Evaluacion",
        "Dx", "DX",
    ])


def resolve_plan(text: str) -> Optional[str]:
    """Resolve plan from ``Plan`` / ``Treatment Plan`` / Spanish equivalents."""
    return resolve_field_flexible(text, [
        "Plan", "Treatment Plan",
        "Plan de Tratamiento", "Plan Terapeutico",
        "Conducta", "Manejo", "Ordenes Medicas", "Recomendaciones",
    ])


def resolve_physical_exam(text: str) -> Optional[str]:
    """Resolve physical exam from ``Physical Exam`` / ``Examination`` / Spanish equivalents."""
    return resolve_field_flexible(text, [
        "Physical Exam", "Examination",
        "Examen Fisico", "Exploracion Fisica", "Examen Clinico",
        "Revision por Sistemas", "Signos Vitales",
    ])


def resolve_chief_complaint(text: str) -> Optional[str]:
    """Resolve chief complaint from ``Chief Complaint`` / ``Reason for Visit`` / Spanish equivalents."""
    return resolve_field_flexible(text, [
        "Chief Complaint", "Reason for Visit",
        "Motivo de Consulta", "Motivo Consulta", "Motivo de Ingreso",
        "Enfermedad Actual", "Causa Externa",
    ])


def resolve_medications(text: str) -> Optional[str]:
    """Resolve medications from ``Current Medications`` / ``Medications`` / Spanish equivalents."""
    return resolve_field_flexible(text, [
        "Current Medications", "Medications",
        "Medicamentos", "Medicamentos Actuales", "Tratamiento Actual",
        "Farmacologicos", "Tratamiento Farmacologico",
    ])
