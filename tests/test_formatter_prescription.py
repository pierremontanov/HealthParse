import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.prescription_schema import (
    Prescription,
    MedicineItem,
    RadiologyItem,
    LabTestItem,
    SpecialistItem,
    TherapyItem,
    GenericItem
)
from pipeline.output_formatter import save_json_output

def run_prescription_test_all_types():
    prescription = Prescription(
        patient_name="Ana Torres",
        patient_id="11223344",
        date="2024-05-25",
        doctor_name="Dr. Natalia Pineda",
        institution="Hospital San Marcos",
        additional_notes="Múltiples derivaciones según sintomatología.",
        items=[
            MedicineItem(
                type="medicine",
                name="Paracetamol 500mg",
                dosage="500mg",
                frequency="cada 8 horas",
                route="oral",
                duration="5 días",
                notes="Tomar después de las comidas"
            ),
            RadiologyItem(
                type="radiology",
                name="Radiografía de tórax",
                modality="RX",
                body_part="tórax",
                notes="Descartar neumonía"
            ),
            LabTestItem(
                type="lab_test",
                name="Examen de sangre",
                test_type="Hemograma completo",
                parameters=["glóbulos rojos", "glóbulos blancos", "plaquetas"],
                notes="Ayuno requerido"
            ),
            SpecialistItem(
                type="specialist",
                name="Consulta con neumólogo",
                specialty="neumología",
                reason="tos persistente",
                notes="Solicitar cita prioritaria"
            ),
            TherapyItem(
                type="procedure",
                name="Fisioterapia respiratoria",
                therapy_type="respiratoria",
                body_part="pulmones",
                frequency="3 veces por semana",
                duration="2 semanas",
                notes="Realizar en sede principal"
            ),
            GenericItem(
                type="other",
                name="Dieta blanda por 7 días",
                notes="Evitar alimentos grasos"
            )
        ]
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(project_root, "output", "formatted_prescription.json")

    save_json_output(prescription, output_path)
    print(f"✅ Formatted prescription with all item types saved to: {output_path}")

if __name__ == "__main__":
    run_prescription_test_all_types()