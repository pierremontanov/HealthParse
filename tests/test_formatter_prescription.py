import sys
import os
import pytest
import json

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

@pytest.fixture
def sample_prescription():
    return Prescription(
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

def test_save_formatted_prescription(tmp_path, sample_prescription):
    output_path = tmp_path / "formatted_prescription.json"
    save_json_output(sample_prescription, str(output_path))
    
    assert output_path.exists(), "Output JSON file was not created"
    
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["patient_name"] == "Ana Torres"
    assert len(data["items"]) == 6
    assert data["items"][0]["type"] == "medicine"
