"""Shared field-alias helpers for rule-based entity extractors.

Each medical document type uses slightly different headers for the same
semantic field (e.g. ``Clinic`` vs ``Institution``, ``Doctor`` vs
``Physician``).  This module centralises the ordered fallback chains so
that every extractor resolves them identically.

Usage
-----
    from src.pipeline.extractors.field_aliases import resolve_institution

    institution = resolve_institution(text)
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from src.pipeline.extractors.base import extract_date, extract_field

# ── Generic resolver ──────────────────────────────────────────────


def _resolve(
    text: str,
    aliases: Sequence[str],
    extractor: Callable[[str, str], Optional[str]],
) -> Optional[str]:
    """Try *aliases* in order using *extractor*, return the first hit."""
    for alias in aliases:
        value = extractor(text, alias)
        if value:
            return value
    return None


# ── Shared field resolvers ────────────────────────────────────────


def resolve_institution(text: str) -> Optional[str]:
    """Resolve institution name from ``Clinic`` / ``Institution``."""
    return _resolve(text, ("Clinic", "Institution"), extract_field)


def resolve_doctor(text: str) -> Optional[str]:
    """Resolve doctor name from ``Doctor`` / ``Physician`` / ``Professional``."""
    return _resolve(text, ("Doctor", "Physician", "Professional"), extract_field)


def resolve_exam_date(text: str) -> Optional[str]:
    """Resolve exam date from ``Exam Date`` / ``Date of Exam`` / ``Date``."""
    return _resolve(text, ("Exam Date", "Date of Exam", "Date"), extract_date)


def resolve_prescription_date(text: str) -> Optional[str]:
    """Resolve prescription date from ``Date of Prescription`` / ``Date``."""
    return _resolve(text, ("Date of Prescription", "Date"), extract_date)


def resolve_assessment(text: str) -> Optional[str]:
    """Resolve assessment from ``Assessment`` / ``Diagnosis``."""
    return _resolve(text, ("Assessment", "Diagnosis"), extract_field)


def resolve_plan(text: str) -> Optional[str]:
    """Resolve plan from ``Plan`` / ``Treatment Plan``."""
    return _resolve(text, ("Plan", "Treatment Plan"), extract_field)


def resolve_physical_exam(text: str) -> Optional[str]:
    """Resolve physical exam from ``Physical Exam`` / ``Examination``."""
    return _resolve(text, ("Physical Exam", "Examination"), extract_field)


def resolve_chief_complaint(text: str) -> Optional[str]:
    """Resolve chief complaint from ``Chief Complaint`` / ``Reason for Visit``."""
    return _resolve(text, ("Chief Complaint", "Reason for Visit"), extract_field)


def resolve_medications(text: str) -> Optional[str]:
    """Resolve medications from ``Current Medications`` / ``Medications``."""
    return _resolve(text, ("Current Medications", "Medications"), extract_field)
