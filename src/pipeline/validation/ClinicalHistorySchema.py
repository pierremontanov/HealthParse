from pydantic import BaseModel, Field
from typing import Optional, List

class ClinicalHistorySchema(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or document number")
    age: Optional[int] = Field(None, description="Patient's age")
    sex: Optional[str] = Field(None, description="Gender: M/F/Other")
    date_of_birth: Optional[str] = Field(None, description="ISO 8601 date format (yyyy-mm-dd)")

    consultation_date: str = Field(..., description="ISO 8601 consultation date")
    chief_complaint: Optional[str] = Field(None, description="Primary reason for the visit")
    medical_history: Optional[str] = Field(None, description="Relevant past medical/surgical history")
    current_medications: Optional[List[str]] = Field(None, description="List of current medications")
    physical_exam: Optional[str] = Field(None, description="Summary of physical exam findings")
    assessment: Optional[str] = Field(None, description="Doctor's interpretation or impressions")
    plan: Optional[str] = Field(None, description="Treatment plan or follow-up actions")

    doctor_name: str = Field(..., description="Name of the attending physician")
    institution: Optional[str] = Field(None, description="Clinic or hospital name")

    class Config:
        str_strip_whitespace = True
        extra = "forbid"
