"""Pydantic schemas for prescription documents.

Prescriptions contain a polymorphic ``items`` list where each item is
discriminated by its ``type`` field (medicine, radiology, lab_test,
specialist, procedure, or other).
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ── Item subtypes ─────────────────────────────────────────────────

class BasePrescriptionItem(BaseModel):
    """Shared fields for every prescription line item."""

    type: str = Field(..., description="Item category discriminator.")
    name: str = Field(
        ...,
        min_length=1,
        description="Name of the prescribed item (drug, test, referral, etc.).",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text instructions or observations for this item.",
    )


class MedicineItem(BasePrescriptionItem):
    """A prescribed medication."""

    type: Literal["medicine"]
    dosage: Optional[str] = Field(None, description="Dose amount (e.g. '500mg').")
    frequency: Optional[str] = Field(None, description="How often to take (e.g. '3 times daily').")
    route: Optional[str] = Field(None, description="Administration route (e.g. 'oral', 'topical').")
    duration: Optional[str] = Field(None, description="Duration of treatment (e.g. '7 days').")


class TherapyItem(BasePrescriptionItem):
    """A prescribed therapy or procedure."""

    type: Literal["procedure"]
    therapy_type: Optional[str] = Field(None, description="Kind of therapy (e.g. 'physical therapy').")
    body_part: Optional[str] = Field(None, description="Target body part.")
    frequency: Optional[str] = Field(None, description="Session frequency.")
    duration: Optional[str] = Field(None, description="Treatment duration.")


class RadiologyItem(BasePrescriptionItem):
    """A prescribed imaging study."""

    type: Literal["radiology"]
    modality: Optional[str] = Field(None, description="Imaging modality (e.g. 'X-Ray', 'MRI').")
    body_part: Optional[str] = Field(None, description="Body region to image.")


class SpecialistItem(BasePrescriptionItem):
    """A referral to a specialist."""

    type: Literal["specialist"]
    specialty: Optional[str] = Field(None, description="Medical specialty (e.g. 'Cardiology').")
    reason: Optional[str] = Field(None, description="Reason for the referral.")


class LabTestItem(BasePrescriptionItem):
    """A prescribed laboratory test."""

    type: Literal["lab_test"]
    test_type: Optional[str] = Field(None, description="Category of lab test (e.g. 'blood panel').")
    parameters: Optional[List[str]] = Field(None, description="Specific parameters to measure.")


class GenericItem(BasePrescriptionItem):
    """Catch-all for items that don't fit other categories."""

    type: Literal["other"]


# Discriminated union of all item types
PrescriptionItem = Union[
    MedicineItem,
    TherapyItem,
    RadiologyItem,
    SpecialistItem,
    LabTestItem,
    GenericItem,
]


# ── Top-level prescription schema ─────────────────────────────────

class Prescription(BaseModel):
    """Validated payload for a prescription document."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    patient_name: Optional[str] = Field(
        None,
        description="Full name of the patient.",
        json_schema_extra={"example": "Carlos Ruiz"},
    )
    patient_id: Optional[str] = Field(
        None,
        description="Patient ID or document number.",
        json_schema_extra={"example": "88997766"},
    )
    date: Optional[str] = Field(
        None,
        description="Prescription date in ISO 8601 format (yyyy-mm-dd).",
        json_schema_extra={"example": "2024-05-25"},
    )
    doctor_name: Optional[str] = Field(
        None,
        description="Name of the prescribing physician.",
        json_schema_extra={"example": "Dr. Andrés López"},
    )
    institution: Optional[str] = Field(
        None,
        description="Clinic or hospital that issued the prescription.",
        json_schema_extra={"example": "Clínica Central"},
    )
    additional_notes: Optional[str] = Field(
        None,
        description="General notes or warnings for the prescription.",
    )
    items: List[PrescriptionItem] = Field(
        ...,
        min_length=1,
        description="List of prescribed items (medications, tests, referrals, etc.).",
    )
