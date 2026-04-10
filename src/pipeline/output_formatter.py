"""Unified output formatting and export for DocIQ results.

Provides functions for every supported export format (JSON, CSV, FHIR)
as well as single-document helpers.  :class:`~src.pipeline.core_engine.DocIQEngine`
delegates all disk I/O through this module.

Usage
-----
    from src.pipeline.output_formatter import export_results

    path = export_results(results, output_dir="output", fmt="json")
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel

from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema

logger = logging.getLogger(__name__)

# ── Type alias ───────────────────────────────────────────────────

FormattedDoc = Union[ResultSchema, Prescription, ClinicalHistorySchema]

_SCHEMA_MAP: Dict[str, type] = {
    "result": ResultSchema,
    "prescription": Prescription,
    "clinical_history": ClinicalHistorySchema,
}


# ── Single-document helpers ──────────────────────────────────────

def format_document(doc: FormattedDoc) -> dict:
    """Convert a validated Pydantic document into a plain dict.

    Parameters
    ----------
    doc : FormattedDoc
        A validated schema instance.

    Returns
    -------
    dict
        The document with ``None`` values excluded.
    """
    return doc.model_dump(exclude_none=True)


def save_json_output(doc: FormattedDoc, output_path: str) -> str:
    """Serialise a single validated document to a JSON file.

    Parameters
    ----------
    doc : FormattedDoc
        A validated schema instance.
    output_path : str
        Destination file path (parent directories created automatically).

    Returns
    -------
    str
        The absolute path of the written file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    formatted = format_document(doc)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
    logger.debug("Wrote single JSON to %s", output_path)
    return os.path.abspath(output_path)


# ── Batch export – JSON ──────────────────────────────────────────

def export_json(
    items: List[Dict[str, Any]],
    output_dir: str,
    *,
    dirname: str | None = None,
) -> str:
    """Export each result dict as a separate JSON file.

    Parameters
    ----------
    items : list[dict]
        Document result dicts (from :class:`DocumentResult.as_dict`).
    output_dir : str
        Parent directory.
    dirname : str, optional
        Name of the sub-directory for the JSON files.  Defaults to
        ``dociq_results``.

    Returns
    -------
    str
        Path to the output sub-directory.
    """
    json_dir = Path(output_dir) / (dirname or "dociq_results")
    json_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        safe_name = Path(item["file"]).stem
        doc_path = json_dir / f"{safe_name}.json"
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d JSON files to %s", len(items), json_dir)
    return str(json_dir)


# ── Batch export – CSV ───────────────────────────────────────────

def export_csv(
    items: List[Dict[str, Any]],
    output_dir: str,
    *,
    filename: str | None = None,
) -> str:
    """Export results as a single CSV table.

    Nested ``extracted_data`` dicts are JSON-serialised into the cell so
    that the CSV remains flat.

    Parameters
    ----------
    items : list[dict]
        Document result dicts.
    output_dir : str
        Parent directory.
    filename : str, optional
        CSV filename.  Defaults to ``dociq_results.csv``.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / (filename or "dociq_results.csv")

    if not items:
        csv_path.write_text("")
        return str(csv_path)

    fieldnames = list(items[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in items:
            flat = dict(row)
            if isinstance(flat.get("extracted_data"), dict):
                flat["extracted_data"] = json.dumps(
                    flat["extracted_data"], ensure_ascii=False
                )
            writer.writerow(flat)

    logger.info("Exported %d results to %s", len(items), csv_path)
    return str(csv_path)


# ── Batch export – FHIR ─────────────────────────────────────────

def export_fhir(
    items: List[Dict[str, Any]],
    output_dir: str,
    *,
    dirname: str | None = None,
) -> str:
    """Export results as FHIR-mapped JSON resources.

    Each document whose type is recognised and whose ``extracted_data``
    can be validated is converted to a loose FHIR resource and saved as
    a ``<stem>_fhir.json`` file.

    Parameters
    ----------
    items : list[dict]
        Document result dicts.
    output_dir : str
        Parent directory.
    dirname : str, optional
        Sub-directory name.  Defaults to ``dociq_fhir``.

    Returns
    -------
    str
        Path to the output sub-directory.
    """
    from src.pipeline.fhir_mapper import map_to_fhir_loose

    fhir_dir = Path(output_dir) / (dirname or "dociq_fhir")
    fhir_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for item in items:
        doc_type = item.get("document_type")
        data = item.get("extracted_data")
        if not doc_type or not data or doc_type not in _SCHEMA_MAP:
            continue
        try:
            model_cls = _SCHEMA_MAP[doc_type]
            model = model_cls(**data)
            fhir_resource = map_to_fhir_loose(model)
            safe_name = Path(item["file"]).stem
            fhir_path = fhir_dir / f"{safe_name}_fhir.json"
            with open(fhir_path, "w", encoding="utf-8") as f:
                json.dump(fhir_resource, f, indent=2, ensure_ascii=False)
            exported += 1
        except Exception as exc:
            logger.warning("FHIR export skipped for %s: %s", item.get("file"), exc)

    logger.info("Exported %d FHIR resource(s) to %s", exported, fhir_dir)
    return str(fhir_dir)


# ── Unified dispatcher ───────────────────────────────────────────

def export_results(
    results: Sequence[Dict[str, Any]],
    *,
    output_dir: str,
    fmt: str = "json",
    filename: str | None = None,
) -> str:
    """Persist results to disk in the requested format.

    Parameters
    ----------
    results : sequence of dict
        The results to export.
    output_dir : str
        Directory to write output into (created if missing).
    fmt : str
        ``"json"`` | ``"csv"`` | ``"fhir"``.
    filename : str, optional
        Override the output filename or sub-directory name.

    Returns
    -------
    str
        Path to the output file or directory.

    Raises
    ------
    ValueError
        If *fmt* is not one of the supported formats.
    """
    items = list(results)

    if fmt == "json":
        return export_json(items, output_dir, dirname=filename)
    elif fmt == "csv":
        return export_csv(items, output_dir, filename=filename)
    elif fmt == "fhir":
        return export_fhir(items, output_dir, dirname=filename)
    else:
        raise ValueError(
            f"Unsupported export format: {fmt!r}. Use 'json', 'csv', or 'fhir'."
        )


__all__ = [
    "FormattedDoc",
    "export_csv",
    "export_fhir",
    "export_json",
    "export_results",
    "format_document",
    "save_json_output",
]
