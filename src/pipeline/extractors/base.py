"""Shared helpers for rule-based entity extraction.

Supports both English and Spanish medical document formats from
across Latin America (Chile, Colombia, Mexico, etc.).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
#  Core field extraction (language-agnostic)
# =====================================================================

def extract_field(text: str, key: str) -> Optional[str]:
    """Extract a single ``Key: Value`` field from *text*.

    Matches lines of the form ``Key: <value>`` and returns the stripped
    value.  Also handles OCR'd form layouts where the colon may be
    missing (``Key  Value``).  Returns ``None`` when the key is not found.
    """
    # Try with colon first (most reliable)
    pattern = re.compile(
        rf"^{re.escape(key)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        return value if value else None

    # Fallback: no colon — common in OCR'd tabular forms
    # Require at least 2 spaces or a tab between key and value to avoid
    # false positives from normal prose.
    pattern_nocolon = re.compile(
        rf"(?:^|\n)\s*{re.escape(key)}\s{{2,}}(.+?)(?:\s{{2,}}|\n|$)",
        re.IGNORECASE,
    )
    match = pattern_nocolon.search(text)
    if match:
        value = match.group(1).strip()
        return value if value else None

    return None


def extract_date(text: str, key: str) -> Optional[str]:
    """Extract a date value from a ``Key: <date>`` or ``Key  <date>`` line.

    Supports ISO-8601 (``yyyy-mm-dd``), European (``dd-mm-yyyy``),
    slash-separated variants, and Spanish month names.
    Handles OCR'd forms where the colon may be missing.
    """
    # ── With colon ──
    # Try numeric date first: yyyy-mm-dd, dd-mm-yyyy, dd/mm/yyyy
    # Also handles dates with time suffix: "17/12/2024 8:10.00 a m"
    pattern = re.compile(
        rf"^{re.escape(key)}\s*:\s*(\d{{2,4}}[\-/]\d{{2}}[\-/]\d{{2,4}})"
        rf"(?:\s+\d{{1,2}}[:.]\d{{2}})?",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return parse_spanish_date(match.group(1).strip()) or match.group(1).strip()

    # Try Spanish month name: "13-ene-2025", "13 de enero de 2025"
    pattern2 = re.compile(
        rf"^{re.escape(key)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE,
    )
    match2 = pattern2.search(text)
    if match2:
        raw = match2.group(1).strip()
        iso = parse_spanish_date(raw)
        if iso:
            return iso

    # ── Without colon (OCR'd tabular forms) ──
    # In OCR'd tables, fields appear mid-line (e.g. "...Mur Fecha Ingreso 17/12/2024...")
    # so we drop the line-start anchor.  Dates are unambiguous enough to avoid
    # false positives even without it.
    pattern_nocolon = re.compile(
        rf"{re.escape(key)}\s+"
        rf"(\d{{2,4}}[\-/]\d{{2}}[\-/]\d{{2,4}})"
        rf"(?:\s+\d{{1,2}}[:.]\d{{2}})?",  # optional time suffix
        re.IGNORECASE,
    )
    match3 = pattern_nocolon.search(text)
    if match3:
        return parse_spanish_date(match3.group(1).strip()) or match3.group(1).strip()

    # ── Bare digits (OCR'd date without separators): "Fecha nac. 27041953" ──
    pattern_bare = re.compile(
        rf"{re.escape(key)}\s+(\d{{8}})\b",
        re.IGNORECASE,
    )
    match4 = pattern_bare.search(text)
    if match4:
        iso = parse_spanish_date(match4.group(1).strip())
        if iso:
            return iso

    return None


def extract_block(text: str, header: str) -> Optional[str]:
    """Extract a multi-line block that starts with *header*.

    Returns all lines after the header until either the next header-style
    line (``SomeTitle:``) or the end of the text.
    """
    pattern = re.compile(
        rf"^{re.escape(header)}\s*:\s*\n((?:.+(?:\n|$))*)",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        block = match.group(1).strip()
        return block if block else None
    return None


def extract_list_items(block: str) -> List[str]:
    """Split a block into items that start with ``- `` or ``* ``."""
    items = re.split(r"\n\s*[-*]\s+", block)
    items = [item.strip() for item in items if item.strip()]
    if block.lstrip().startswith(("-", "*")):
        return items
    if items:
        items[0] = items[0].lstrip("- *").strip()
    return items


def extract_dated_entries(block: str) -> List[Tuple[str, str]]:
    """Parse ``- YYYY-MM-DD: text`` entries from a block."""
    pattern = re.compile(
        r"-\s*(\d{4}-\d{2}-\d{2})\s*:\s*(.+?)(?=\n\s*-\s*\d{4}-\d{2}-\d{2}|$)",
        re.DOTALL,
    )
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(block)]


def extract_test_results(block: str) -> List[Dict[str, Any]]:
    """Parse ``- TestName: value (Ref: range)`` entries from a lab result block."""
    pattern = re.compile(
        r"-\s*(.+?):\s*([\d.,]+)\s*\(Ref:\s*([^)]+)\)",
    )
    results = []
    for m in pattern.finditer(block):
        results.append({
            "test_name": m.group(1).strip(),
            "value": m.group(2).strip(),
            "reference_range": m.group(3).strip(),
        })
    return results


# =====================================================================
#  Flexible multi-alias field resolver
# =====================================================================

def resolve_field_flexible(text: str, aliases: List[str]) -> Optional[str]:
    """Try each alias as a ``Key: Value`` field; return first match."""
    for alias in aliases:
        val = extract_field(text, alias)
        if val:
            return val
    return None


def resolve_date_flexible(text: str, aliases: List[str]) -> Optional[str]:
    """Try each alias as a date field; return first match."""
    for alias in aliases:
        val = extract_date(text, alias)
        if val:
            return val
    return None


# =====================================================================
#  Spanish date handling
# =====================================================================

_ES_MONTHS = {
    # Abbreviations
    "ene": "01", "feb": "02", "mar": "03", "abr": "04",
    "may": "05", "jun": "06", "jul": "07", "ago": "08",
    "sep": "09", "oct": "10", "nov": "11", "dic": "12",
    # Full names
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11",
    "diciembre": "12",
}


def parse_spanish_date(raw: str) -> Optional[str]:
    """Convert a Spanish date string to ISO 8601 (``yyyy-mm-dd``).

    Handles: ``13-ene-2025``, ``13 de enero de 2025``, ``13/ene/2025``,
    ``dd-mm-yyyy``, ``dd/mm/yyyy``, ``yyyy/mm/dd``.
    """
    if not raw:
        return None
    raw = raw.strip()

    # Already ISO? "2021-02-12"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if m:
        return raw

    # "13-ene-2025" or "13/ene/2025"
    m = re.match(r"(\d{1,2})[-/]([a-z]+)[-/](\d{4})", raw, re.IGNORECASE)
    if m:
        day, month_str, year = m.group(1), m.group(2).lower(), m.group(3)
        mm = _ES_MONTHS.get(month_str)
        if mm:
            return f"{year}-{mm}-{int(day):02d}"

    # "13 de enero de 2025"
    m = re.match(
        r"(\d{1,2})\s+de\s+([a-z]+)\s+de[l]?\s+(\d{4})", raw, re.IGNORECASE,
    )
    if m:
        day, month_str, year = m.group(1), m.group(2).lower(), m.group(3)
        mm = _ES_MONTHS.get(month_str)
        if mm:
            return f"{year}-{mm}-{int(day):02d}"

    # dd-mm-yyyy or dd/mm/yyyy
    m = re.match(r"(\d{2})[-/](\d{2})[-/](\d{4})", raw)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month}-{day}"

    # yyyy/mm/dd
    m = re.match(r"(\d{4})[-/](\d{2})[-/](\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 8-digit no separators: ddmmyyyy (common in OCR'd forms)
    m = re.match(r"^(\d{2})(\d{2})(\d{4})$", raw)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{year}-{month}-{day}"

    return None


def extract_spanish_date(text: str) -> Optional[str]:
    """Find a Spanish-format date anywhere in *text* and return ISO 8601."""
    # "dd-mon-yyyy" or "dd/mon/yyyy"
    m = re.search(r"(\d{1,2}[-/][a-z]+[-/]\d{4})", text, re.IGNORECASE)
    if m:
        return parse_spanish_date(m.group(1))

    # "dd de monthname de yyyy"
    m = re.search(
        r"(\d{1,2}\s+de\s+[a-z]+\s+de[l]?\s+\d{4})", text, re.IGNORECASE,
    )
    if m:
        return parse_spanish_date(m.group(1))

    return None


# =====================================================================
#  Flexible patient name extraction
# =====================================================================

# Comprehensive aliases for patient name across Latin American documents
_PATIENT_NAME_ALIASES = [
    "Patient Name",
    "NOMBRE",
    "Nombre",
    "Nombre del Paciente",
    "Nombre Completo",
    "Nombre del paciente",
    "Paciente",
    "Nombre y Apellido",
    "Nombre y Apellidos",
    "Nombres y Apellidos",
    "Nombres",
]


def extract_patient_name(text: str) -> Optional[str]:
    """Extract patient name using all known field aliases and patterns.

    Tries structured ``Key: Value`` fields first, then falls back to
    narrative sentence patterns common in Chilean and Colombian documents.
    """
    # 1. Structured field aliases
    for alias in _PATIENT_NAME_ALIASES:
        val = extract_field(text, alias)
        if val:
            # Strip trailing age like ", 40 a" or ", 43 ANOS"
            val = re.sub(r"\s*,\s*\d+\s*(?:a|anos|ANOS|ANOS)?\s*$", "", val)
            return val.strip() if val.strip() else None

    # 2. Narrative pattern: "paciente , NAME ha dado"
    val = extract_patient_name_from_sentence(text)
    if val:
        return val

    return None


def extract_patient_name_from_sentence(text: str) -> Optional[str]:
    """Extract a patient name from Spanish narrative sentence patterns."""
    patterns = [
        # "paciente , NAME ha dado" / "paciente, NAME ha"
        re.compile(
            r"paciente\s*[,:]\s*([A-ZÀ-Ü]"
            r"[A-ZÀ-Ü\s]+?)\s+ha\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        # "paciente NAME," (without ha)
        re.compile(
            r"paciente\s*[,:]\s*([A-ZÀ-Ü]"
            r"[A-ZÀ-Ü\s]{3,}?)\s*,",
            re.IGNORECASE | re.MULTILINE,
        ),
        # "se atiende a NAME" / "se examina a NAME"
        re.compile(
            r"se\s+(?:atiende|examina|presenta|recibe)\s+a\s+"
            r"([A-ZÀ-Ü][A-ZÀ-Ü\s]{3,}?)(?:\s*[,.]|\s+de\s+\d)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            name = m.group(1).strip().rstrip(",.")
            if len(name) > 2:
                return name
    return None


# =====================================================================
#  Flexible patient ID extraction
# =====================================================================

_PATIENT_ID_ALIASES = [
    "Patient ID",
    "ID Paciente",
    "No. Identificacion",
    "Identificacion",
    "Tipo y No. de Identificacion",
    "No Historia o Afiliacion",
    "No Historia",
    "No. Historia",
    "Historia Clinica",
    "Historia",
    "HC",
    "Num. Historia",
    "Numero de Historia",
    "Patologia No",
    "Patologia No.",
    "No. Identificacion",
    "Identificacion",
    "Cedula",
    "RUT",
    "RUN",
    "DNI",
    "Documento",
    "No. Documento",
    "Afiliacion",
    "No. Afiliacion",
    "Num. Afiliacion",
]


def extract_patient_id(text: str) -> Optional[str]:
    """Extract patient ID using all known aliases and Colombian patterns."""
    val = resolve_field_flexible(text, _PATIENT_ID_ALIASES)
    if val:
        return val

    # Colombian CC / TI / CE patterns: "CC 24314628", "C.C. 24314628"
    m = re.search(
        r"(?:C\.?C\.?|T\.?I\.?|C\.?E\.?|Cedula)\s*[:\s]\s*(\d{5,12})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    return None


# =====================================================================
#  Flexible date extraction
# =====================================================================

_EXAM_DATE_ALIASES = [
    "Exam Date",
    "Date of Exam",
    "Date",
    "Fecha de Examen",
    "Fecha del Examen",
    "Fecha Examen",
    "Fecha Solicitud",
    "Fecha de Solicitud",
    "Fecha Informe",
    "Fecha de Informe",
    "Fecha del Informe",
    "Fecha Resultado",
    "Fecha de Resultado",
    "Fecha Estudio",
    "Fecha del Estudio",
    "Fecha de Toma",
    "Fecha de Muestra",
    "Fecha Reporte",
    "Fecha",
]

_CONSULTATION_DATE_ALIASES = [
    "Consultation Date",
    "Visit Date",
    "Date",
    "Fecha de Consulta",
    "Fecha Consulta",
    "Fecha de Atencion",
    "Fecha Atencion",
    "Fecha de la Consulta",
    "Fecha Ingreso",
    "Fecha de Ingreso",
    "Fecha de Admision",
    "Fecha Admision",
    "Fecha",
]

_PRESCRIPTION_DATE_ALIASES = [
    "Date of Prescription",
    "Prescription Date",
    "Date",
    "Fecha de Prescripcion",
    "Fecha Prescripcion",
    "Fecha de Formulacion",
    "Fecha Formulacion",
    "Fecha de Receta",
    "Fecha Receta",
    "Fecha",
]

_BIRTH_DATE_ALIASES = [
    "Date of Birth",
    "DOB",
    "Fecha de Nacimiento",
    "Fecha Nacimiento",
    "F. Nacimiento",
    "F. Nac",
    "F. Nac.",
    "Fecha nac",
    "Fecha nac.",
    "Nacimiento",
]


def extract_exam_date(text: str) -> Optional[str]:
    """Extract exam/report date using all known aliases, with Spanish fallback."""
    val = resolve_date_flexible(text, _EXAM_DATE_ALIASES)
    if val:
        return val
    return extract_spanish_date(text)


def extract_consultation_date(text: str) -> Optional[str]:
    """Extract consultation date using all known aliases, with Spanish fallback."""
    val = resolve_date_flexible(text, _CONSULTATION_DATE_ALIASES)
    if val:
        return val
    return extract_spanish_date(text)


def extract_prescription_date(text: str) -> Optional[str]:
    """Extract prescription date using all known aliases, with Spanish fallback."""
    val = resolve_date_flexible(text, _PRESCRIPTION_DATE_ALIASES)
    if val:
        return val
    return extract_spanish_date(text)


def extract_birth_date(text: str) -> Optional[str]:
    """Extract date of birth using all known aliases."""
    return resolve_date_flexible(text, _BIRTH_DATE_ALIASES)


# =====================================================================
#  Flexible doctor / professional extraction
# =====================================================================

_DOCTOR_ALIASES = [
    "Doctor",
    "Physician",
    "Professional",
    "Medico",
    "Profesional",
    "Medico Tratante",
    "Medico Responsable",
    "Medico Solicitante",
    "Dr",
    "Dra",
    "Validado por",
    "Informado por",
    "Firmado por",
    "Responsable",
    "Patologo",
    "Radiologo",
]


def extract_doctor(text: str) -> Optional[str]:
    """Extract doctor name from structured fields or Spanish signatures."""
    # Try structured fields
    val = resolve_field_flexible(text, _DOCTOR_ALIASES)
    if val:
        return val

    # Narrative patterns
    return extract_professional_spanish(text)


def extract_professional_spanish(text: str) -> Optional[str]:
    """Extract the professional name from Spanish-format signatures."""
    patterns = [
        # "Informe Validado por: Dra. Fatima Mota"
        re.compile(
            r"(?:Informe\s+)?[Vv]alidado\s+por\s*:\s*(.+?)$",
            re.MULTILINE,
        ),
        # "Firmado por: ..."
        re.compile(r"[Ff]irmado\s+por\s*:\s*(.+?)$", re.MULTILINE),
        # "Informado por: ..."
        re.compile(r"[Ii]nformado\s+por\s*:\s*(.+?)$", re.MULTILINE),
        # Standalone "Dr./Dra. Name" on its own line (not inside a sentence)
        re.compile(r"^(Dra?\.\s*[A-ZÀ-Ü][a-zA-ZÀ-ÿ\s]+?)$", re.MULTILINE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            name = m.group(1).strip()
            if len(name) > 3:
                return name
    return None


# =====================================================================
#  Flexible institution extraction
# =====================================================================

_INSTITUTION_ALIASES = [
    "Clinic",
    "Institution",
    "Institucion",
    "Centro",
    "Hospital",
    "Clinica",
    "Laboratorio",
    "Entidad",
    "EPS",
    "IPS",
    "Establecimiento",
    "Sede",
    "Unidad",
]


def extract_institution(text: str) -> Optional[str]:
    """Extract institution from structured fields or Spanish patterns."""
    val = resolve_field_flexible(text, _INSTITUTION_ALIASES)
    if val:
        return val
    return extract_institution_spanish(text)


def extract_institution_spanish(text: str) -> Optional[str]:
    """Detect institution names from narrative text."""
    patterns = [
        # Standard institution names
        re.compile(
            r"((?:Centro\s+(?:M[eé]dico|de\s+Salud)|Hospital|Cl[ií]nica|"
            r"Laboratorio|Instituto|Fundaci[oó]n|Sanatorio|Consultorio)"
            r"\s+[A-ZÀ-Ü][a-zA-ZÀ-ÿ\s]+?)(?:\n|$)",
            re.IGNORECASE,
        ),
        # Colombian corporate names: "MEIDE SAS", "COLSANITAS S.A."
        re.compile(
            r"([A-ZÀ-Ü]{3,}(?:\s+[A-ZÀ-Ü]+)*\s+(?:S\.?A\.?S\.?|S\.?A\.?|LTDA|E\.?S\.?E\.?|I\.?P\.?S\.?))\b",
        ),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            name = m.group(1).strip()
            if len(name) > 3:
                return name
    return None


# =====================================================================
#  Flexible age extraction
# =====================================================================

def extract_age(text: str) -> Optional[int]:
    """Extract patient age from structured fields or Spanish patterns."""
    # Structured field: "Edad: 43 ANOS"
    raw = extract_field(text, "Edad")
    if not raw:
        raw = extract_field(text, "Age")
    if not raw:
        raw = extract_field(text, "Edad ingreso")
    if raw:
        m = re.search(r"(\d+)", raw)
        if m:
            age = int(m.group(1))
            if 0 <= age <= 150:
                return age

    # Inline pattern: "Edad 71", "Edad ingreso 71 afios" (OCR for "años")
    m = re.search(
        r"(?:Edad(?:\s+ingreso)?|Age)\s*[:\s]\s*(\d{1,3})\s*"
        r"(?:a[nñ]os?|ANOS|afios|a\b|years?)?",
        text, re.IGNORECASE,
    )
    if m:
        age = int(m.group(1))
        if 0 <= age <= 150:
            return age

    # Footer pattern: "NAME, 40 a" or "40 anos"
    m = re.search(r",\s*(\d{1,3})\s*(?:a\b|anos|ANOS|afios)", text, re.IGNORECASE)
    if m:
        age = int(m.group(1))
        if 0 <= age <= 150:
            return age

    return None


# =====================================================================
#  Flexible sex / gender extraction
# =====================================================================

def extract_sex(text: str) -> Optional[str]:
    """Extract patient sex from structured fields or Spanish patterns.

    Returns ``'M'``, ``'F'``, or ``None``.
    Handles OCR artefacts like ``Sexoalnacer Mur`` (mangled from
    "Sexo al nacer Mujer").
    """
    raw = extract_field(text, "Sexo")
    if not raw:
        raw = extract_field(text, "Sex")
    if not raw:
        raw = extract_field(text, "Genero")
    if not raw:
        raw = extract_field(text, "Sexo al nacer")
    if raw:
        return _normalize_sex(raw)

    # Inline OCR pattern: "Sexo Mujer", "Sexo M", "Sexo al nacer Femenino"
    m = re.search(
        r"(?:Sexo(?:\s*al\s*nacer)?|Sex|Genero)\s*[:\s]\s*"
        r"(Masculino|Femenino|Mujer|Hombre|Male|Female|[MF])\b",
        text, re.IGNORECASE,
    )
    if m:
        return _normalize_sex(m.group(1))

    # OCR-mangled pattern: "Sexoalnacer Mur" / "Sexoalnacer Fem"
    m = re.search(
        r"Sexo\s*al\s*nacer\s+(Mu[a-z]*|Fe[a-z]*|Ho[a-z]*|Ma[a-z]*|[MF])\b",
        text, re.IGNORECASE,
    )
    if m:
        return _normalize_sex(m.group(1))

    return None


def _normalize_sex(raw: str) -> Optional[str]:
    """Normalize sex value to 'M' or 'F'.

    Handles OCR artefacts: truncated words like ``Mur`` (Mujer),
    ``Fem`` (Femenino), ``Mas`` (Masculino), ``Hom`` (Hombre).
    """
    if not raw:
        return None
    v = raw.strip().upper()
    if v in ("M", "MASCULINO", "HOMBRE", "MALE"):
        return "M"
    if v in ("F", "FEMENINO", "MUJER", "FEMALE"):
        return "F"
    # OCR partial matches
    if v.startswith("MU") or v.startswith("FE"):
        return "F"
    if v.startswith("HO") or v.startswith("MAS"):
        return "M"
    return None


# =====================================================================
#  Spanish section block extraction (findings, impression, etc.)
# =====================================================================

_FINDINGS_HEADERS = [
    "HALLAZGOS",
    "DESCRIPCION MICROSCOPICA",
    "DESCRIPCION MACROSCOPICA",
    "DESCRIPCION",
    "INFORME",
    "RESULTADOS",
    "Test Results",
    "Findings",
]

_IMPRESSION_HEADERS = [
    "IMPRESION",
    "DIAGNOSTICO HISTOLOGICO",
    "DIAGNOSTICO",
    "CONCLUSION",
    "CONCLUSIONES",
    "CONCEPTO",
    "INTERPRETACION",
    "Summary",
    "Impression",
]

# Boundaries that should stop block capture
_SECTION_BOUNDARY = re.compile(
    r"\n\s*(?:IMPRESION|DIAGNOSTICO|CONCLUSION|CONCEPTO|INTERPRETACION|"
    r"Atentamente|Cordialmente|Saludos|Powered\s+by)\s*[,:]?",
    re.IGNORECASE,
)


def extract_findings_block(text: str) -> Optional[str]:
    """Extract the findings/description section from the document."""
    for header in _FINDINGS_HEADERS:
        block = extract_block(text, header)
        if block:
            # Trim at next section boundary
            block = _SECTION_BOUNDARY.split(block)[0]
            return block.strip() if block.strip() else None
    return None


def extract_impression_block(text: str) -> Optional[str]:
    """Extract the impression/diagnosis/conclusion section."""
    for header in _IMPRESSION_HEADERS:
        block = extract_block(text, header)
        if block:
            # Trim at sign-off
            block = re.split(
                r"\n\s*(?:Atentamente|Cordialmente|Saludos|Powered\s+by"
                r"|Informe\s+Validado|Paciente\s*:)\s*[,:]?",
                block,
                flags=re.IGNORECASE,
            )[0]
            return block.strip() if block.strip() else None
    return None


# =====================================================================
#  Study area / specimen extraction
# =====================================================================

_SPECIMEN_ALIASES = [
    "Especimen(es)",
    "Especimen",
    "Muestra",
    "Muestras",
    "Material",
    "Tipo de Muestra",
    "Specimen",
]


def extract_study_area(text: str) -> Optional[str]:
    """Extract study area from specimen fields or radiology exam titles."""
    # Pathology specimens
    for alias in _SPECIMEN_ALIASES:
        val = extract_field(text, alias)
        if val:
            return val

    # Radiology exam title: "Resonancia Magnetica Columna Lumbar"
    m = re.search(
        r"(?:Resonancia\s+Magn[eé]tica|Radiograf[ií]a|Ecograf[ií]a|"
        r"Tomograf[ií]a|TAC|Electrocardiograma|MRI|CT|X-Ray|Ultrasound)"
        r"\s+(?:de\s+)?(.+?)$",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        area = m.group(1).strip().rstrip(".")
        if len(area) > 1:
            return area

    return None


# =====================================================================
#  Exam type inference (bilingual)
# =====================================================================

def infer_exam_type(text: str, test_results: Optional[list] = None) -> str:
    """Classify the exam type from test names and document text."""
    if test_results:
        names = [tr["test_name"].lower() for tr in test_results]
        all_names = " ".join(names)

        panels = [
            (["glucose", "glucosa", "hba1c", "insulin", "insulina", "glicemia"],
             "Blood Chemistry – Glucose Panel"),
            (["cholesterol", "colesterol", "ldl", "hdl", "triglyceride",
              "triglicerido", "lipid", "lipido"],
             "Blood Chemistry – Lipid Panel"),
            (["hemoglobin", "hemoglobina", "hematocrit", "hematocrito",
              "platelet", "plaqueta", "wbc", "rbc", "cbc", "hemograma",
              "leucocit", "eritrocit"],
             "Hematology – Complete Blood Count"),
            (["creatinine", "creatinina", "bun", "urea", "kidney", "renal"],
             "Blood Chemistry – Renal Panel"),
            (["alt", "ast", "bilirubin", "bilirrubina", "liver",
              "hepatic", "higado"],
             "Blood Chemistry – Liver Panel"),
            (["tsh", "t3", "t4", "thyroid", "tiroides", "tiroide"],
             "Blood Chemistry – Thyroid Panel"),
        ]
        for keywords, label in panels:
            if any(kw in all_names for kw in keywords):
                return label
        return "Laboratory Test"

    lower = text.lower()

    # Pathology (check first — long distinctive keywords)
    if any(kw in lower for kw in [
        "anatomopatologico", "histologico", "histopatolog",
        "biopsia", "trucut", "anatomia patologica",
    ]):
        return "Pathology – Histological Study"
    if "citolog" in lower:
        return "Pathology – Cytology"

    # Spanish radiology
    if "resonancia magn" in lower:
        return "Radiology – MRI"
    if "radiograf" in lower:
        return "Radiology – X-Ray"
    if "tomograf" in lower or re.search(r"\btac\b", lower):
        return "Radiology – CT Scan"
    if "ecograf" in lower or "ultrasonido" in lower:
        return "Radiology – Ultrasound"
    if "electrocardiograma" in lower or re.search(r"\b(?:ecg|ekg)\b", lower):
        return "Cardiology – ECG"
    if "endoscop" in lower:
        return "Gastroenterology – Endoscopy"
    if "densitometr" in lower:
        return "Radiology – Densitometry"
    if "mamograf" in lower:
        return "Radiology – Mammography"
    if "electroencefalograma" in lower or re.search(r"\beeg\b", lower):
        return "Neurology – EEG"

  