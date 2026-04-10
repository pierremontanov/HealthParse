"""FHIR JSON persistence for DocIQ documents.

Provides helpers to persist individual FHIR resources and full
FHIR Bundles to disk.

Usage
-----
    from src.pipeline.fhir_output_saver import save_fhir_output, save_fhir_bundle

    save_fhir_output(validated_doc, "output/patient_fhir.json")
    save_fhir_bundle([resource1, resource2], "output/bundle.json")
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.pipeline.fhir_mapper import build_fhir_bundle, map_to_fhir_loose

logger = logging.getLogger(__name__)

__all__ = ["save_fhir_output", "save_fhir_bundle"]


def save_fhir_output(document: object, output_path: str) -> str:
    """Convert a validated document to a FHIR resource and write to disk.

    Parameters
    ----------
    document : ResultSchema | Prescription | ClinicalHistorySchema
        A validated Pydantic document instance.
    output_path : str
        Destination file path (parent directories created automatically).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    fhir_data = map_to_fhir_loose(document)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fhir_data, f, indent=2, ensure_ascii=False)

    logger.info("FHIR JSON saved to: %s", output_path)
    return os.path.abspath(output_path)


def save_fhir_bundle(
    resources: Sequence[dict],
    output_path: str,
    *,
    bundle_type: str = "collection",
) -> str:
    """Wrap FHIR resources in a Bundle and write to disk.

    Parameters
    ----------
    resources : sequence of dict
        Individual FHIR resource dicts.
    output_path : str
        Destination file path.
    bundle_type : str
        FHIR Bundle type.  Defaults to ``collection``.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    bundle = build_fhir_bundle(resources, bundle_type=bundle_type)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    logger.info(
        "FHIR Bundle (%d entries) saved to: %s", len(resources), output_path
    )
    return os.path.abspath(output_path)
