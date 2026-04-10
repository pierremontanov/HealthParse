import json
import pytest

from src.pipeline.fhir_output_saver import save_fhir_output
from src.pipeline.validation.schemas import ResultSchema

@pytest.fixture
def sample_result():
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

def test_save_fhir_output(tmp_path, sample_result):
    output_file = tmp_path / "fhir_result.json"
    save_fhir_output(sample_result, str(output_file))

    # ✅ Assert file exists
    assert output_file.exists(), "❌ Output file was not created"

    # ✅ Assert file contains valid JSON
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["resourceType"] == "DiagnosticReport"
    assert data["subject"]["name"] == sample_result.patient_name
    assert data["issuer"]["display"] == sample_result.institution
