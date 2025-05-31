import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.validator import validate_prescription
from pipeline.validation.prescription_schema import Prescription

def test_validate_prescription():
    test_data = {
        "patient_name": "Carlos Ruiz",
        "patient_id": "88997766",
        "date": "2024-05-25",
        "doctor_name": "Dr. Andrés López",
        "institution": "Clínica Central",
        "additional_notes": "Paciente con hipertensión controlada.",
        "items": [
            {
                "type": "medicine",
                "name": "Losartán 50mg",
                "dosage": "50mg",
                "frequency": "1 vez al día",
                "route": "oral",
                "duration": "30 días",
                "notes": "Tomar en la mañana"
            }
        ]
    }

    result = validate_prescription(test_data)
    assert isinstance(result, Prescription)
    assert result.patient_name == "Carlos Ruiz"
    assert result.items[0].type == "medicine"
    assert result.items[0].name == "Losartán 50mg"
