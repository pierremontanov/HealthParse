import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline.validation.validator import validate_result_schema
from pipeline.validation.schemas import ResultSchema


def test_validate_result_schema():
    data = {
        "patient_name": "Gloria Ines Montaño Villada",
        "patient_id": "24314628",
        "age": 71,
        "sex": "F",
        "date_of_birth": "27-04-1953",
        "exam_type": "CR",
        "study_area": "Columna Dorsal",
        "exam_date": "08-08-2024",
        "findings": "Cambios incipientes de tipo degenerativo crónico.",
        "impression": "Alineamiento satisfactorio. Tejidos blandos normales.",
        "professional": "Dra. Fátima Mota Arteaga",
        "institution": "Centro Médico San José",
        "notes": "No se evidencian fracturas."
    }

    validated = validate_result_schema(data)
    assert isinstance(validated, ResultSchema)
    assert validated.patient_name == "Gloria Ines Montaño Villada"
