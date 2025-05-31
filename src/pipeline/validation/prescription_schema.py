from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal

# Enum for types
class PrescriptionType(str):
    MEDICINE = "medicine"
    RADIOLOGY = "radiology"
    LAB_TEST = "lab_test"
    SPECIALIST = "specialist"
    PROCEDURE = "procedure"
    OTHER = "other"

# Base model for shared fields
class BasePrescriptionItem(BaseModel):
    type: str
    name: str = Field(..., description="Name of the prescribed item")
    notes: Optional[str] = Field(None, description="Optional notes or instructions")

# Specialized subtypes
class MedicineItem(BasePrescriptionItem):
    type: Literal["medicine"]
    dosage: Optional[str]
    frequency: Optional[str]
    route: Optional[str]
    duration: Optional[str]

class TherapyItem(BasePrescriptionItem):
    type: Literal["procedure"]
    therapy_type: Optional[str]
    body_part: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]

class RadiologyItem(BasePrescriptionItem):
    type: Literal["radiology"]
    modality: Optional[str]
    body_part: Optional[str]

class SpecialistItem(BasePrescriptionItem):
    type: Literal["specialist"]
    specialty: Optional[str]
    reason: Optional[str]

class LabTestItem(BasePrescriptionItem):
    type: Literal["lab_test"]
    test_type: Optional[str]
    parameters: Optional[List[str]]

class GenericItem(BasePrescriptionItem):
    type: Literal["other"]

# Union of all items
PrescriptionItem = Union[
    MedicineItem,
    TherapyItem,
    RadiologyItem,
    SpecialistItem,
    LabTestItem,
    GenericItem,
]

# Complete prescription schema
class Prescription(BaseModel):
    patient_name: Optional[str]
    patient_id: Optional[str]
    date: Optional[str]  # Expected to be normalized to ISO 8601 before validation
    doctor_name: Optional[str]
    institution: Optional[str]
    additional_notes: Optional[str]
    items: List[PrescriptionItem]
