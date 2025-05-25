from pydantic import BaseModel, Field
from typing import Optional

class ResultSchema(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or document number")
    age: Optional[int] = Field(None, description="Age in years")
    sex: Optional[str] = Field(None, description="Gender: M/F/Other")
    date_of_birth: Optional[str] = Field(None, description="Date in dd-mm-yyyy format")
    
    exam_type: str = Field(..., description="Type of medical imaging or lab test")
    study_area: Optional[str] = Field(None, description="Body part or system studied")
    exam_date: str = Field(..., description="Date of the exam or test")
    findings: str = Field(..., description="Clinical findings described in the document")
    impression: Optional[str] = Field(None, description="Final interpretation or diagnosis")

    professional: str = Field(..., description="Name of the doctor who validated the report")
    institution: Optional[str] = Field(None, description="Name of the clinic or institution where the study was performed")
    notes: Optional[str] = Field(None, description="Additional observations")

    class Config:
        str_strip_whitespace = True
        extra = "forbid"
