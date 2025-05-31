import os
import json
import logging
from pipeline.fhir_mapper import map_to_fhir_loose

logger = logging.getLogger(__name__)

def save_fhir_output(document, output_path):
    """
    Converts a validated document to FHIR loose format and saves it as a JSON file.

    Args:
        document (ResultSchema | Prescription | ClinicalHistorySchema): A validated document instance.
        output_path (str): Path to save the JSON file.
    """
    fhir_data = map_to_fhir_loose(document)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fhir_data, f, indent=2, ensure_ascii=False)

    logger.info(f"FHIR JSON saved to: {output_path}")
