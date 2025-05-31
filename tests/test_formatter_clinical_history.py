import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema
from pipeline.output_formatter import save_json_output

@pytest.fixture
def clinical_history_example():
    return ClinicalHistorySchema(
        patient_name="Lucía Hernández",
        patient_id="55667788",
        age=63,
        sex="F",
        date_of_birth="1960-01-15",
        consultation_date="2024-05-24",
        chief_complaint="Dolor torácico intermitente",
        medical_history="Antecedentes de hipertensión y dislipidemia.",
        current_medications=["Atorvastatina", "Losartán"],
        physical_exam="PA 135/85, FC 82, sin hallazgos anormales.",
        assessment="Dolor torácico no relacionado a esfuerzo. Posible origen musculoesquelético.",
        plan="Indicar AINE por 5 días, control en 1 semana.",
        doctor_name="Dra. Marcela Gómez",
        institution="Centro Médico Vida"
    )

def test_save_clinical_history_output(clinical_history_example, tmp_path):
    output_path = tmp_path / "formatted_clinical_history_output.json"
    save_json_output(clinical_history_example, str(output_path))

    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["patient_name"] == "Lucía Hernández"
    assert data["doctor_name"] == "Dra. Marcela Gómez"
    assert data["institution"] == "Centro Médico Vida"
