"""Pydantic schema for clinical history documents.

This schema validates structured data extracted from clinical history,
consultation notes, and patient encounter records.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClinicalHistorySchema(BaseModel):
    """Validated payload for a clinical history / encounter document."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    patient_name: str = Field(
        ...,
        min_length=1,
        description="Full name of the patient.",
        json_schema_extra={"example": "Maria Elena Rodriguez"},
    )
    patient_id: Optional[str] = Field(
        None,
        description="Patient ID or document number.",
        json_schema_extra={"example": "45012"},
    )
    age: Optional[int] = Field(
        None,
        ge=0,
        le=150,
        description="Patient age in years at the time of the consultation.",
    )
    sex: Optional[str] = Field(
        None,
        description="Biological sex: 'M', 'F', or 'Other'.",
    )
    date_of_birth: Optional[str] = Field(
        None,
        description="Date of birth in ISO 8601 format (yyyy-mm-dd).",
        json_schema_extra={"example": "1985-03-15"},
    )

    consultation_date: str = Field(
        ...,
        description="Date of the consultation in ISO 8601 format (yyyy-mm-dd).",
        json_schema_extra={"example": "2025-02-22"},
    )
    chief_complaint: Optional[str] = Field(
        None,
        description="Primary reason for the visit or chief complaint.",
    )
    medical_history: Optional[str] = Field(
        None,
        description="Relevant past medical, surgical, or family history.",
    )
    current_medications: Optional[List[str]] = Field(
        None,
        description="List of medications the patient is currently taking.",
    )
    physical_exam: Optional[str] = Field(
        None,
        description="Summary of physical examination findings.",
    )
    assessment: Optional[str] = Field(
        None,
        description="Doctor's clinical assessment or working diagnosis.",
    )
    plan: Optional[str] = Field(
        None,
        description="Treatment plan, follow-up actions, or referrals.",
    )

    doctor_name: str = Field(
        ...,
        min_length=1,
        description="Name of the attending physician.",
        json_schema_extra={"example": "Dr. Carlos Rodriguez"},
    )
    institution: Optional[str] = Field(
        None,
        description="Clinic or hospital where the consultation took place.",
        json_schema_extra={"example": "Central Medical Center"},
    )
