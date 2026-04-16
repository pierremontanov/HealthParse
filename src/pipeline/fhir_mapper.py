"""FHIR resource mapping for validated DocIQ documents.

Maps validated Pydantic schema instances to loose FHIR R4 resources:

- :class:`ResultSchema`          → ``DiagnosticReport``
- :class:`Prescription`          → ``MedicationRequest``
- :class:`ClinicalHistorySchema` → ``Encounter``

The ``map_to_fhir_loose`` dispatcher selects the correct mapper at
runtime via ``isinstance`` checks.  :func:`build_fhir_bundle` wraps
one or more resources into a FHIR ``Bundle`` of type ``collection``.

All mappers strip ``None`` values from the output so that only
populated fields appear in the JSON artefact.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.clinical_history_schema import ClinicalHistorySchema

__all__ = [
    "map_to_fhir_loose",
    "result_to_fhir_loose",
    "prescription_to_fhir",
    "clinical_history_to_fhir",
    "build_fhir_bundle",
    "prune_none",
]


# ── Utility helpers ─────────────────────────────────────────────

def _new_uuid() -> str:
    """Return a new UUID-4 string."""
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def prune_none(d: Any) -> Any:
    """Recursively remove keys whose value is ``None`` from *d*.

    Works on dicts and lists (including nested structures).
    Non-dict/list values pass through unchanged.
    """
    if isinstance(d, dict):
        return {k: prune_none(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [prune_none(item) for item in d]
    return d


# ── Dispatcher ──────────────────────────────────────────────────

def map_to_fhir_loose(document: object) -> dict:
    """Convert a validated Pydantic document to a loose FHIR resource.

    Parameters
    ----------
    document : ResultSchema | Prescription | ClinicalHistorySchema
        A validated schema instance.

    Returns
    -------
    dict
        A FHIR-like resource dict with a unique ``id`` and ``meta.lastUpdated``.

    Raises
    ------
    FHIRMappingError
        If *document* is not a recognised schema type.
    """
    if isinstance(document, ResultSchema):
        return result_to_fhir_loose(document)
    elif isinstance(document, Prescription):
        return prescription_to_fhir(document)
    elif isinstance(document, ClinicalHistorySchema):
        return clinical_history_to_fhir(document)
    else:
        from src.pipeline.exceptions import FHIRMappingError
        raise FHIRMappingError(
            "unknown", f"Unsupported document type: {type(document).__name__}"
        )


# ── Result → DiagnosticReport ───────────────────────────────────

def result_to_fhir_loose(result: ResultSchema) -> dict:
    """Map a lab / imaging result to a FHIR ``DiagnosticReport``."""
    resource: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": _new_uuid(),
        "meta": {"lastUpdated": _utc_now_iso()},
        "status": "final",
        "subject": {
            "display": result.patient_name,
            "identifier": {"value": result.patient_id},
            "birthDate": result.date_of_birth,
            "gender": result.sex,
        },
        "effectiveDateTime": result.exam_date,
        "category": [{"coding": [{"display": result.exam_type}]}],
        "code": {"text": result.study_area or "General"},
        "conclusion": result.impression,
        "presentedForm": [
            {"contentType": "text/plain", "data": result.findings}
        ],
        "performer": [{"display": result.professional}],
        "note": [{"text": result.notes}] if result.notes else None,
        "issuer": {"display": result.institution},
    }
    return prune_none(resource)


# ── Prescription → MedicationRequest ────────────────────────────

def prescription_to_fhir(prescription: Prescription) -> dict:
    """Map a prescription to a FHIR ``MedicationRequest``."""
    contained = []
    for item in prescription.items:
        if item.type == "medicine":
            med: Dict[str, Any] = {
                "resourceType": "Medication",
                "code": {"text": item.name},
                "form": {"text": getattr(item, "route", None)},
                "dosageInstruction": [
                    {
                        "text": item.notes,
                        "timing": {
                            "repeat": {
                                "frequency": 1,
                                "period": getattr(item, "duration", None),
                            }
                        },
                        "doseAndRate": [
                            {
                                "doseQuantity": {
                                    "value": getattr(item, "dosage", None),
                                }
                            }
                        ],
                    }
                ],
            }
            contained.append(prune_none(med))

    resource: Dict[str, Any] = {
        "resourceType": "MedicationRequest",
        "id": _new_uuid(),
        "meta": {"lastUpdated": _utc_now_iso()},
        "status": "active",
        "intent": "order",
        "subject": {
            "display": prescription.patient_name,
            "identifier": {"value": prescription.patient_id},
        },
        "authoredOn": prescription.date,
        "requester": {"display": prescription.doctor_name},
        "note": (
            [{"text": prescription.additional_notes}]
            if prescription.additional_notes
            else None
        ),
        "contained": contained or None,
    }
    return prune_none(resource)


# ── Clinical History → Encounter ────────────────────────────────

def clinical_history_to_fhir(clinical: ClinicalHistorySchema) -> dict:
    """Map a clinical history record to a FHIR ``Encounter``."""
    resource: Dict[str, Any] = {
        "resourceType": "Encounter",
        "id": _new_uuid(),
        "meta": {"lastUpdated": _utc_now_iso()},
        "status": "finished",
        "class": {"code": "AMB", "display": "ambulatory"},
        "subject": {
            "display": clinical.patient_name,
            "identifier": {"value": clinical.patient_id},
        },
        "period": {"start": clinical.consultation_date},
        "reasonCode": (
            [{"text": clinical.chief_complaint}]
            if clinical.chief_complaint
            else None
        ),
        "diagnosis": (
            [{"condition": {"display": clinical.assessment}}]
            if clinical.assessment
            else None
        ),
        "participant": [{"individual": {"display": clinical.doctor_name}}],
        "location": (
            [{"location": {"display": clinical.institution}}]
            if clinical.institution
            else None
        ),
        "note": [{"text": clinical.plan}] if clinical.plan else None,
    }
    return prune_none(resource)


# ── Bundle builder ──────────────────────────────────────────────

def build_fhir_bundle(
    resources: Sequence[dict],
    *,
    bundle_type: str = "collection",
    bundle_id: Optional[str] = None,
) -> dict:
    """Wrap one or more FHIR resources into a FHIR ``Bundle``.

    Parameters
    ----------
    resources : sequence of dict
        Individual FHIR resource dicts (each must have ``resourceType``).
    bundle_type : str
        Bundle type (``collection``, ``document``, ``batch``, etc.).
        Defaults to ``collection``.
    bundle_id : str, optional
        Override the Bundle ``id``.  A UUID-4 is generated when omitted.

    Returns
    -------
    dict
        A complete FHIR Bundle resource.
    """
    entries: List[Dict[str, Any]] = []
    for res in resources:
        entry: Dict[str, Any] = {"resource": res}
        res_type = res.get("resourceType", "Unknown")
        res_id = res.get("id", _new_uuid())
        entry["fullUrl"] = f"urn:uuid:{res_id}"
        entries.append(entry)

    return {
        "resourceType": "Bundle",
        "id": bundle_id or _new_uuid(),
        "meta": {"lastUpdated": _utc_now_iso()},
        "type": bundle_type,
        "total": len(entries),
        "entry": entries,
    }
