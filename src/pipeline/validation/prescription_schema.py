from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal


# Enum-like types
class PrescriptionType(str):
    MEDICINE = "medicine"
    RADIOLOGY = "radiology"
    LAB_TEST = "lab_test"
    SPECIALIST = "specialist"
    PROCEDURE = "procedure"
    OTHER = "other"


# Base item schema
class BasePrescriptionItem(BaseModel):
    type: str
    name: str = Field(..., description="Name of the prescribed item")
    notes: Optional[str] = Field(None, description="Optional notes or instructions")


class MedicineItem(BasePrescriptionItem):
    type: Literal["medicine"]
    dosage: Optional[str] = Field(None, description="Dosage amount or strength")
    frequency: Optional[str] = Field(None, description="Frequency of intake")
    route: Optional[str] = Field(None, description="Route of administration")
    duration: Optional[str] = Field(None, description="Duration of treatment")


class TherapyItem(BasePrescriptionItem):
    type: Literal["procedure"]
    therapy_type: Optional[str] = Field(None, description="Type of therapy")
    body_part: Optional[str] = Field(None, description="Targeted body part")
    frequency: Optional[str] = Field(None, description="Therapy frequency")
    duration: Optional[str] = Field(None, description="Duration of therapy")


class RadiologyItem(BasePrescriptionItem):
    type: Literal["radiology"]
    modality: Optional[str] = Field(None, description="Imaging modality")
    body_part: Optional[str] = Field(None, description="Imaged body part")


class SpecialistItem(BasePrescriptionItem):
    type: Literal["specialist"]
    specialty: Optional[str] = Field(None, description="Medical specialty")
    reason: Optional[str] = Field(None, description="Reason for referral")


class LabTestItem(BasePrescriptionItem):
    type: Literal["lab_test"]
    test_type: Optional[str] = Field(None, description="Type of lab test")
    parameters: Optional[List[str]] = Field(None, description="Test parameters")


class GenericItem(BasePrescriptionItem):
    type: Literal["other"]


# Union type
PrescriptionItem = Union[
    MedicineItem,
    TherapyItem,
    RadiologyItem,
    SpecialistItem,
    LabTestItem,
    GenericItem,
]


# Final prescription schema
class Prescription(BaseModel):
    patient_name: Optional[str] = Field(None, description="Full name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or document number")
    date: Optional[str] = Field(None, description="Date of the prescription")
    doctor_name: Optional[str] = Field(None, description="Doctor issuing the prescription")
    institution: Optional[str] = Field(None, description="Name of the issuing clinic or hospital")
    additional_notes: Optional[str] = Field(None, description="General comments")
    items: List[PrescriptionItem] = Field(..., description="List of prescribed items")

    class Config:
        anystr_strip_whitespace = True
        extra = "forbid"
