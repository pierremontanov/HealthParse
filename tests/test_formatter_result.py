# test_formatter_result.py

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.schemas import ResultSchema
from pipeline.output_formatter import format_document, save_json_output

@pytest.fixture
def example_result():
    return ResultSchema(
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

def test_result_formatter_output(tmp_path, example_result):
    output_path = tmp_path / "formatted_result_output.json"
    save_json_output(example_result, str(output_path))

    assert output_path.exists(), "Output file was not created"

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["patient_name"] == "Gloria Ines Montaño Villada"
    assert data["exam_type"] == "CR"
    assert data["institution"] == "Centro Médico San José"
