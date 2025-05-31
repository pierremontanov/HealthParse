import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.fhir_mapper import map_to_fhir_loose
from pipeline.validation.schemas import ResultSchema
from pipeline.validation.prescription_schema import Prescription
from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema

def test_result_mapping():
    result = ResultSchema(
        patient_name="Juan Perez",
        patient_id="12345",
        age=50,
        sex="M",
        date_of_birth="1973-02-15",
        exam_type="CR",
        study_area="Torax",
        exam_date="2024-06-01",
        findings="Infiltrados pulmonares visibles.",
        impression="Posible neumonía.",
        professional="Dra. Clara Gómez",
        institution="Hospital Nacional",
        notes="Requiere seguimiento clínico."
    )
    print("\n🧪 FHIR Output for Result:")
    print(map_to_fhir_loose(result))

def test_prescription_mapping():
    prescription = Prescription(
        patient_name="Carlos Ruiz",
        patient_id="88997766",
        date="2024-05-25",
        doctor_name="Dr. Andrés López",
        institution="Clínica Central",
        additional_notes="Paciente con hipertensión controlada.",
        items=[
            {
                "type": "medicine",
                "name": "Losartán 50mg",
                "notes": "Tomar en la mañana",
                "dosage": "50mg",
                "frequency": "1 vez al día",
                "route": "oral",
                "duration": "30 días"
            }
        ]
    )
    print("\n🧪 FHIR Output for Prescription:")
    print(map_to_fhir_loose(prescription))

def test_clinical_history_mapping():
    clinical = ClinicalHistorySchema(
        patient_name="Lucía Méndez",
        patient_id="99887766",
        age=42,
        sex="F",
        date_of_birth="1982-09-13",
        consultation_date="2024-05-10",
        chief_complaint="Dolor abdominal persistente.",
        medical_history="Antecedente de gastritis.",
        current_medications=["Omeprazol 20mg"],
        physical_exam="Sensibilidad en epigastrio.",
        assessment="Probable úlcera gástrica.",
        plan="Endoscopía programada.",
        doctor_name="Dr. Jorge Méndez",
        institution="Centro Médico Colonial"
    )
    print("\n🧪 FHIR Output for Clinical History:")
    print(map_to_fhir_loose(clinical))

if __name__ == "__main__":
    test_result_mapping()
    test_prescription_mapping()
    test_clinical_history_mapping()
