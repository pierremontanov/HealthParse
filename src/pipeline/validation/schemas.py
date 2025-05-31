from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

class ResultSchema(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or document number")
    age: Optional[int] = Field(None, description="Age in years")
    sex: Optional[str] = Field(None, description="Gender: M/F/Other")
    date_of_birth: Optional[str] = Field(None, description="Date in yyyy-mm-dd (ISO 8601)")
    exam_type: str = Field(..., description="Type of medical imaging or lab test")
    study_area: Optional[str] = Field(None, description="Body part or system studied")
    exam_date: str = Field(..., description="Date of the exam or test in yyyy-mm-dd (ISO 8601)")
    findings: str = Field(..., description="Clinical findings described in the document")
    impression: Optional[str] = Field(None, description="Final interpretation or diagnosis")
    professional: str = Field(..., description="Name of the doctor who validated the report")
    institution: str = Field(..., description="Institution where the test was performed")
    notes: Optional[str] = Field(None, description="Additional observations")

    # Validate 'sex' only
    @field_validator("sex")
    def validate_sex(cls, v):
        if v and v.upper() not in {"M", "F", "OTHER"}:
            raise ValueError("Sex must be 'M', 'F', or 'Other'")
        return v

    class Config:
        str_strip_whitespace = True
        extra = "forbid"
