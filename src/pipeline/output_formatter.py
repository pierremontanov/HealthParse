import os
import json
from typing import Union
from pydantic import BaseModel

from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema

# 1. Union of valid schemas
FormattedDoc = Union[ResultSchema, Prescription, ClinicalHistorySchema]

# 2. Convert Pydantic model to dict
def format_document(doc: FormattedDoc) -> dict:
    """Convert the validated document into a standard dict."""
    return doc.model_dump(exclude_none=True)

# 3. Save to JSON file
def save_json_output(doc: FormattedDoc, output_path: str):
    """Save the formatted document as a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    formatted = format_document(doc)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
