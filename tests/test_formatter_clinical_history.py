import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema
from pipeline.output_formatter import save_json_output

def run_clinical_history_test():
    example_history = ClinicalHistorySchema(
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

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(project_root, "output", "formatted_clinical_history_output.json")

    save_json_output(example_history, output_path)
    print(f"✅ Formatted clinical history output saved to: {output_path}")

if __name__ == "__main__":
    run_clinical_history_test()