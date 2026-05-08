"""Pydantic schema for lab/imaging result documents.

This schema validates structured data extracted from lab results,
imaging reports, and diagnostic documents.  All fields include
descriptions that double as API documentation for consumers.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResultSchema(BaseModel):
    """Validated payload for a lab or imaging result document."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    patient_name: Optional[str] = Field(
        None,
        description="Full name of the patient.",
        json_schema_extra={"example": "Gloria Ines Montaño Villada"},
    )
    patient_id: Optional[str] = Field(
        None,
        description="Patient ID or document number assigned by the institution.",
        json_schema_extra={"example": "24314628"},
    )
    age: Optional[int] = Field(
        None,
        ge=0,
        le=150,
        description="Patient age in years at the time of the exam.",
    )
    sex: Optional[str] = Field(
        None,
        description="Biological sex: 'M', 'F', or 'Other'.",
    )
    date_of_birth: Optional[str] = Field(
        None,
        description="Date of birth in ISO 8601 format (yyyy-mm-dd).",
        json_schema_extra={"example": "1953-04-27"},
    )
    exam_type: Optional[str] = Field(
        "Laboratory Test",
        description="Type of exam performed (e.g. 'CBC', 'Blood Chemistry', 'X-Ray CR').",
        json_schema_extra={"example": "Blood Chemistry – Glucose Panel"},
    )
    study_area: Optional[str] = Field(
        None,
        description="Body part or system studied (e.g. 'Thorax', 'Abdomen').",
    )
    exam_date: Optional[str] = Field(
        None,
        description="Date the exam was performed in ISO 8601 format (yyyy-mm-dd).",
        json_schema_extra={"example": "2024-08-08"},
    )
    findings: Optional[str] = Field(
        None,
        description="Clinical findings described in the document.",
    )
    impression: Optional[str] = Field(
        None,
        description="Final interpretation, diagnosis, or summary by the professional.",
    )
    professional: Optional[str] = Field(
        None,
        description="Name of the doctor or technician who validated the report.",
        json_schema_extra={"example": "Dra. Fátima Mota Arteaga"},
    )
    institution: Optional[str] = Field(
        None,
        description="Institution or laboratory where the exam was performed.",
        json_schema_extra={"example": "Centro Médico San José"},
    )
    notes: Optional[str] = Field(
        None,
        description="Additional observations or free-text notes.",
    )

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.upper() not in {"M", "F", "OTHER"}:
            raise ValueError(
                f"Invalid sex value '{v}'. Must be 'M', 'F', or 'Other'."
            )
        return v
