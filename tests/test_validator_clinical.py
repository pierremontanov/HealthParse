import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.validator import validate_clinical_history

def test_validate_clinical_history():
    data = {
        "patient_name": "Laura Gómez",
        "patient_id": "123456789",
        "age": 52,
        "sex": "F",
        "date_of_birth": "1972-01-15",
        "consultation_date": "2024-05-30",
        "chief_complaint": "Dolor abdominal",
        "medical_history": "Hipertensión, gastritis",
        "current_medications": ["Omeprazol", "Losartán"],
        "physical_exam": "Dolor en hipogastrio, sin fiebre",
        "assessment": "Gastritis crónica",
        "plan": "Dieta blanda, seguimiento en 1 semana",
        "doctor_name": "Dra. Mariana Ríos",
        "institution": "Hospital General"
    }

    validated = validate_clinical_history(data)
    print("✅ Clinical history schema validated:", validated)
