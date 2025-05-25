import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.schemas import ResultSchema
from pipeline.output_formatter import format_document, save_json_output

def run_formatter_test():
    example_result = ResultSchema(
        patient_name="Gloria Ines Montaño Villada",
        patient_id="24314628",
        age=71,
        sex="F",
        date_of_birth="27-04-1953",
        exam_type="CR",
        study_area="Columna Dorsal",
        exam_date="08-08-2024",
        findings="Cambios incipientes de tipo degenerativo crónico.",
        impression="Alineamiento satisfactorio. Tejidos blandos normales.",
        professional="Dra. Fátima Mota Arteaga",
        institution="Centro Médico San José",
        notes="No se evidencian fracturas."
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(project_root, "output", "formatted_result_output.json")

    save_json_output(example_result, output_path)
    print(f"✅ Formatted output saved to: {output_path}")

if __name__ == "__main__":
    run_formatter_test()
