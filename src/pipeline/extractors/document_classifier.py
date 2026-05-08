"""Rule-based document type classifier.

Analyses extracted text to determine the clinical document type so that the
correct NER extractor can be selected by the inference engine.

Supports both English and Spanish (Latin American) medical documents.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ── Keyword scoring tables ──
# Each keyword contributes a score towards a document type.  The type with the
# highest aggregate score wins.

_PRESCRIPTION_KEYWORDS: Dict[str, float] = {
    # ── English ──
    "prescription": 3.0,
    "prescribed": 2.0,
    "dosage": 2.0,
    "frequency": 1.5,
    "medication": 2.5,
    "medicine": 2.0,
    "tablet": 2.0,
    "capsule": 2.0,
    "date of prescription": 3.0,
    "route": 1.0,
    "oral": 0.5,       # reduced — appears in clinical histories too
    "topical": 1.0,
    "mg": 0.5,          # reduced — medications are listed in clinical histories
    "ml": 0.5,          # reduced
    # ── Spanish ──
    "prescripcion": 3.0,
    "receta medica": 3.0,
    "receta": 2.5,
    "formulacion": 3.0,
    "formula medica": 3.0,
    "ordenes medicas": 2.5,
    "indicaciones medicas": 2.5,
    "posologia": 2.5,
    "via oral": 1.0,
    "via topica": 1.0,
    "cada 8 horas": 1.5,
    "cada 12 horas": 1.5,
    "cada 24 horas": 1.5,
    "tomar": 0.5,
    "aplicar": 0.5,
    "comprimido": 2.0,
    "tableta": 2.0,
    "capsula": 2.0,
}

_RESULT_KEYWORDS: Dict[str, float] = {
    # ── English ──
    "test results": 3.0,
    "exam date": 3.0,
    "exam type": 2.0,
    "findings": 2.5,
    "impression": 2.0,
    "reference": 1.0,
    "ref:": 2.0,
    "summary": 1.0,
    "lab": 1.5,
    "result": 1.5,
    "blood": 1.0,
    "urine": 1.0,
    "specimen": 1.5,
    "hemoglobin": 2.0,
    "glucose": 1.5,     # reduced — glucose/diabetes appears in clinical histories
    "cholesterol": 2.0,
    # ── Spanish ──
    "resultados": 2.5,
    "resultado de laboratorio": 3.0,
    "resultado de examen": 3.0,
    "informe de laboratorio": 3.0,
    "laboratorio clinico": 3.0,
    "examen de laboratorio": 3.0,
    "fecha de examen": 2.5,
    "tipo de examen": 2.0,
    "hallazgos": 2.5,
    "conclusion": 2.0,
    "impresion diagnostica": 1.5,   # also used in clinical history
    "valor de referencia": 3.0,
    "valores de referencia": 3.0,
    "rango de referencia": 3.0,
    "muestra": 1.5,
    "hemograma": 3.0,
    "perfil lipidico": 3.0,
    "glicemia": 2.0,
    "creatinina": 2.0,
    "hemoglobina": 2.0,
    "hematocrito": 2.0,
    "leucocitos": 2.0,
    "plaquetas": 2.0,
    "orina": 1.5,
    "sangre": 1.0,
    "radiografia": 2.5,
    "ecografia": 2.5,
    "tomografia": 2.5,
    "resonancia": 2.5,
    "informe radiologico": 3.0,
    "tecnica": 1.5,
}

_CLINICAL_HISTORY_KEYWORDS: Dict[str, float] = {
    # ── English ──
    "annotations": 3.0,
    "clinic history": 3.0,
    "clinical history": 3.0,
    "consultation date": 3.0,
    "visit date": 2.0,
    "chief complaint": 2.5,
    "medical history": 2.5,
    "assessment": 2.0,
    "plan": 0.5,          # very generic word — reduced
    "physical exam": 2.0,
    "current medications": 2.0,
    "reason for visit": 2.0,
    # ── Spanish — general clinical history ──
    "historia clinica": 4.0,
    "historial clinico": 3.5,
    "anamnesis": 4.0,
    "motivo de consulta": 4.0,
    "motivo consulta": 3.5,
    "motivo de ingreso": 3.5,
    "enfermedad actual": 3.5,
    "antecedentes": 3.0,
    "antecedentes personales": 3.5,
    "antecedentes familiares": 3.5,
    "antecedentes patologicos": 3.5,
    "antecedentes quirurgicos": 3.5,
    "antecedentes farmacologicos": 3.5,
    "antecedentes alergicos": 3.0,
    "revision por sistemas": 3.5,
    "examen fisico": 3.5,
    "exploracion fisica": 3.0,
    "signos vitales": 3.0,
    "diagnostico": 2.0,
    "impresion diagnostica": 2.0,
    "plan de tratamiento": 2.5,
    "conducta": 2.0,
    "evolucion": 2.5,
    "evoluciones": 2.5,
    "notas clinicas": 3.0,
    "consulta": 1.5,
    "fecha de consulta": 3.0,
    "fecha de ingreso": 3.0,
    "fecha de atencion": 3.0,
    # ── Colombian FOMAG / EPS form terms ──
    "medicina general": 2.5,
    "medicina interna": 2.5,
    "servicio": 1.0,
    "eps": 1.5,
    "ips": 1.5,
    "fomag": 2.0,
    "ficha de atencion": 3.0,
    "consulta medica": 3.0,
    "epicrisis": 3.5,
    "resumen de atencion": 3.0,
    "pa sistolica": 2.5,
    "pa diastolica": 2.5,
    "tension arterial": 2.5,
    "frecuencia cardiaca": 2.5,
    "peso": 1.0,
    "talla": 1.0,
    "imc": 1.5,
    # ── Body systems review (Spanish) ──
    "cabeza y cuello": 2.0,
    "cardiovascular": 1.5,
    "respiratorio": 1.5,
    "gastrointestinal": 1.5,
    "genitourinario": 1.5,
    "musculoesqueletico": 1.5,
    "neurologico": 1.5,
    "piel y faneras": 2.0,
    "extremidades": 1.5,
    # ── Referral / authorization (maps to clinical_history for now) ──
    "remision": 2.5,
    "autorizacion": 1.5,
    "solicitud de autorizacion": 2.5,
    "orden de servicio": 2.0,
    # ── Medical abbreviations (common in short consultation notes) ──
    "cedula": 2.0,
    "procedencia": 2.5,
    "ocupacion": 2.0,
    "patologicos": 3.0,
    "farmacologicos": 3.0,
    "alergicos": 2.5,
    "quirurgicos": 2.5,
    "toxicos": 2.0,
    "histerectomia": 2.0,
    "ooforectomia": 2.0,
    "disartri": 2.0,
    "neurologo": 3.0,
    "neurology": 2.0,
    "cardiology": 2.0,
    "cardiologo": 3.0,
    "internista": 3.0,
    "dermatologo": 3.0,
    "oftalmologo": 3.0,
    "ortopedista": 3.0,
    "urologo": 3.0,
    "ginecologo": 3.0,
    "pensionada": 1.5,
    "pensionado": 1.5,
}

_RECEIPT_KEYWORDS: Dict[str, float] = {
    # ── Spanish pharmacy / store receipts ──
    "factura electronica": 4.0,
    "factura de venta": 4.0,
    "factura": 3.0,
    "drogueria": 4.0,
    "farmacia": 3.0,
    "distribuidora": 3.0,
    "nit": 2.5,
    "sub total": 2.0,
    "total factura": 3.0,
    "valor recibido": 2.5,
    "cambio": 1.0,
    "cajero": 2.0,
    "vendedor": 1.5,
    "tipo de pago": 2.5,
    "efectivo": 1.5,
    "forma de pago": 2.5,
    "contado": 1.5,
    "regimen": 1.5,
    "resolucion": 1.5,
    "cufe": 2.0,
    "gracias por su compra": 3.0,
    "consumidor final": 2.5,
    "sistema pos": 2.0,
    "proveedor tecnologico": 2.0,
    "caja": 1.0,
    "descuento": 1.5,
    # ── English receipt/invoice ──
    "invoice": 3.0,
    "receipt": 3.0,
    "total": 1.0,
    "subtotal": 1.5,
    "payment": 2.0,
    "cashier": 2.0,
    "change": 0.5,
}

DOCUMENT_TYPES = {
    "prescription": _PRESCRIPTION_KEYWORDS,
    "result": _RESULT_KEYWORDS,
    "clinical_history": _CLINICAL_HISTORY_KEYWORDS,
    "receipt": _RECEIPT_KEYWORDS,
}


class DocumentClassifier:
    """Classify a clinical document based on keyword scoring.

    The classifier exposes a ``predict(text)`` method so it can be registered
    as the classifier model inside a :class:`~src.pipeline.inference.ModelBundle`.

    Supports bilingual (English + Spanish) documents.
    """

    def __init__(self, min_score: float = 1.0) -> None:
        self._min_score = min_score

    def predict(self, text: str) -> Dict[str, str]:
        """Return ``{\"document_type\": \"<type>\"}`` or ``{}`` if no match."""
        doc_type = self.classify(text)
        if doc_type:
            return {"document_type": doc_type}
        return {}

    def classify(self, text: str) -> Optional[str]:
        """Return the most likely document type for *text*, or ``None``."""
        lower = text.lower()
        scores: Dict[str, float] = {}

        for doc_type, keywords in DOCUMENT_TYPES.items():
            total = 0.0
            for keyword, weight in keywords.items():
                count = len(re.findall(re.escape(keyword), lower))
                total += count * weight
            scores[doc_type] = total

        if not scores:
            return None

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best_type] >= self._min_score:
            return best_type

        return None
