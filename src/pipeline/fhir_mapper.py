# fhir_mapper.py

from pipeline.validation.schemas import ResultSchema
from pipeline.validation.prescription_schema import Prescription
from pipeline.validation.ClinicalHistorySchema import ClinicalHistorySchema


# ----------------------------
# Generic FHIR Mapper Dispatcher
# ----------------------------
def map_to_fhir_loose(document: object) -> dict:
    if isinstance(document, ResultSchema):
        return result_to_fhir_loose(document)
    elif isinstance(document, Prescription):
        return prescription_to_fhir(document)
    elif isinstance(document, ClinicalHistorySchema):
        return clinical_history_to_fhir(document)
    else:
        raise TypeError(f"Unsupported document type: {type(document).__name__}")

# ----------------------------
# Result to Loose FHIR Mapper
# ----------------------------
def result_to_fhir_loose(result: ResultSchema) -> dict:
    return {
        "resourceType": "DiagnosticReport",
        "id": result.patient_id or "unknown-id",
        "subject": {
            "name": result.patient_name,
            "identifier": result.patient_id,
            "birthDate": result.date_of_birth,
            "gender": result.sex
        },
        "effectiveDateTime": result.exam_date,
        "category": {"text": result.exam_type},
        "code": {"text": result.study_area or "General"},
        "conclusion": result.impression,
        "presentedForm": [{
            "contentType": "text/plain",
            "data": result.findings
        }],
        "performer": [{"display": result.professional}],
        "note": [{"text": result.notes}],
        "issuer": {"display": result.institution}
    }

# ----------------------------
# Prescription to Loose FHIR
# ----------------------------
def prescription_to_fhir(prescription: Prescription) -> dict:
    return {
        "resourceType": "MedicationRequest",
        "subject": {"identifier": {"value": prescription.patient_id}},
        "authoredOn": prescription.date,
        "requester": {"display": prescription.doctor_name},
        "note": [{"text": prescription.additional_notes}],
        "contained": [
            {
                "resourceType": "Medication",
                "code": {"text": item.name},
                "form": {"text": item.route},
                "dosageInstruction": [{
                    "text": item.notes,
                    "timing": {"repeat": {"frequency": 1, "period": item.duration}},
                    "doseAndRate": [{"doseQuantity": {"value": item.dosage}}]
                }]
            } for item in prescription.items if item.type == "medicine"
        ]
    }

# ----------------------------
# Clinical History to Loose FHIR
# ----------------------------
def clinical_history_to_fhir(clinical: ClinicalHistorySchema) -> dict:
    return {
        "resourceType": "Encounter",
        "status": "finished",
        "subject": {
            "identifier": {"value": clinical.patient_id},
            "display": clinical.patient_name
        },
        "period": {
            "start": clinical.consultation_date
        },
        "reasonCode": [{"text": clinical.chief_complaint}],
        "diagnosis": [{"condition": {"text": clinical.assessment}}],
        "participant": [{"individual": {"display": clinical.doctor_name}}],
        "location": [{"location": {"display": clinical.institution}}],
        "note": [{"text": clinical.plan}]
    }
