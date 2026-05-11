"""Shared helpers for rule-based entity extraction.

Supports both English and Spanish medical document formats from
across Latin America (Chile, Colombia, Mexico, etc.).

Accent flexibility
------------------
PaddleOCR with ``lang="en"`` strips accents from Spanish text
(e.g. ``"años"`` → ``"anos"``, ``"Cédula"`` → ``"Cedula"``).
PyMuPDF direct extraction preserves accents.  All alias matching
goes through :func:`_accent_flex` so that both paths are covered
by a single set of aliases.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
#  Accent-flexible regex builder
# =====================================================================

_ACCENT_MAP = {
    "a": "[aáàâã]", "A": "[AÁÀÂÃ]",
    "e": "[eéèê]",  "E": "[EÉÈÊ]",
    "i": "[iíìî]",  "I": "[IÍÌÎ]",
    "o": "[oóòôõ]", "O": "[OÓÒÔÕ]",
    "u": "[uúùû]",  "U": "[UÚÙÛ]",
    "n": "[nñ]",    "N": "[NÑ]",
}


def _accent_flex(key: str) -> str:
    """Convert a literal field label to an accent-flexible regex pattern.

    Each Latin vowel and ``n``/``ñ`` is replaced with a character class
    so that both accented (PyMuPDF) and unaccented (PaddleOCR) text is
    matched.  Non-alpha characters are regex-escaped normally.

    Example::

        >>> _accent_flex("Identificación")
        'Id[eéèê]nt[iíìî]f[iíìî]c[aáàâã]c[iíìî][oóòôõ][nñ]'
    """
    parts: list[str] = []
    for ch in key:
        if ch in _ACCENT_MAP:
            parts.append(_ACCENT_MAP[ch])
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


# =====================================================================
#  Core field extraction (language-agnostic)
# =====================================================================

def extract_field(text: str, key: str) -> Optional[str]:
    """Extract a single ``Key: Value`` field from *text*.

    Matches lines of the form ``Key: <value>`` and returns the stripped
    value.  Also handles OCR'd form layouts where the colon may be
    missing (``Key  Value``).  Returns ``None`` when the key is not found.

    Uses accent-flexible matching so that a single alias like
    ``"Identificacion"`` matches both ``"Identificación"`` (PyMuPDF)
    and ``"Identificacion"`` (PaddleOCR).
    """
    flex_key = _accent_flex(key)

    # Try with colon first (most reliable)
    pattern = re.compile(
        rf"^{flex_key}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        return value if value else None

    # Fallback: no colon — common in OCR'd tabular forms
    # Require at least 1 space (relaxed from 2 for PaddleOCR output
    # where spacing is tighter) between key and value.
    pattern_nocolon = re.compile(
        rf"(?:^|\n)\s*{flex_key}\s{{1,}}(.+?)(?:\s{{2,}}|\n|$)",
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
    Uses accent-flexible matching for the key.
    """
    flex_key = _accent_flex(key)

    # ── With colon ──
    # Try numeric date first: yyyy-mm-dd, dd-mm-yyyy, dd/mm/yyyy
    # Also handles dates with time suffix: "17/12/2024 8:10.00 a m"
    pattern = re.compile(
        rf"^{flex_key}\s*:\s*(\d{{2,4}}[\-/]\d{{2}}[\-/]\d{{2,4}})"
        rf"(?:\s+\d{{1,2}}[:.]\d{{2}})?",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return parse_spanish_date(match.group(1).strip()) or match.group(1).strip()

    # Try Spanish month name: "13-ene-2025", "13 de enero de 2025"
    pattern2 = re.compile(
        rf"^{flex_key}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE,
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
        rf"{flex_key}\s+"
        rf"(\d{{2,4}}[\-/]\d{{2}}[\-/]\d{{2,4}})"
        rf"(?:\s+\d{{1,2}}[:.]\d{{2}})?",  # optional time suffix
        re.IGNORECASE,
    )
    match3 = pattern_nocolon.search(text)
    if match3:
        return parse_spanish_date(match3.group(1).strip()) or match3.group(1).strip()

    # ── Bare digits (OCR'd date without separators): "Fecha nac. 27041953" ──
    pattern_bare = re.compile(
        rf"{flex_key}\s+(\d{{8}})\b",
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
    Uses accent-flexible matching for the header.
    """
    flex_header = _accent_flex(header)
    pattern = re.compile(
        rf"^{flex_header}\s*:?\s*\n((?:.+(?:\n|$))*)",
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

# Comprehensive aliases for patient name across Latin American documents.
# Accent-flex matching is applied automatically by extract_field(), so only
# one spelling per alias is needed (unaccented form preferred).
_PATIENT_NAME_ALIASES = [
    "Patient Name",
    "NOMBRE COMPLETO",
    "Nombre Completo",
    "Nombre del Paciente",
    "Nombre del paciente",
    "Nombre y Apellidos",
    "Nombres y Apellidos",
    "Nombre y Apellido",
    "NOMBRE",
    "Nombre",
    "Paciente",
    "Nombres",
    "Primer Apellido",          # Colombian EPS forms
    "Usuario",                  # some EPS/IPS forms
]


def extract_patient_name(text: str) -> Optional[str]:
    """Extract patient name using all known field aliases and patterns.

    Tries structured ``Key: Value`` fields first, then inline patterns
    common in Colombian/Chilean documents, then narrative sentence patterns.
    """
    # 1. Structured field aliases
    for alias in _PATIENT_NAME_ALIASES:
        val = extract_field(text, alias)
        if val:
            # Strip trailing age like ", 40 a" or ", 43 ANOS"
            val = re.sub(
                r"\s*,\s*\d+\s*(?:a[nñ]os?|ANOS|afios|a)\s*$", "", val,
            )
            # Strip trailing ID like "(24314628)" or "CC 24314628"
            val = re.sub(r"\s*\(?\d{5,12}\)?\s*$", "", val)
            return val.strip() if val.strip() else None

    # 2. Inline pattern: "NOMBRE GLORIA INES MONTANO" (no colon, single space)
    m = re.search(
        r"(?:NOMBRE|Nombre)\s+([A-Z][A-Z\s]{4,}?)(?:\s{2,}|\n|$)",
        text,
    )
    if m:
        name = m.group(1).strip()
        # Avoid matching labels like "NOMBRE COMPLETO"
        if name.upper() not in ("COMPLETO", "DEL PACIENTE", "Y APELLIDOS"):
            return name

    # 3. Narrative patterns: "paciente , NAME ha dado"
    val = extract_patient_name_from_sentence(text)
    if val:
        return val

    return None


def extract_patient_name_from_sentence(text: str) -> Optional[str]:
    """Extract a patient name from Spanish narrative sentence patterns.

    Uses broad character classes ``[A-Z\\w]`` to handle both accented
    (PyMuPDF) and unaccented (PaddleOCR) text.
    """
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
            r"([A-ZÀ-Ü][A-ZÀ-Ü\s]{3,}?)"
            r"(?:\s*[,.]|\s+de\s+\d)",
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
    "Tipo y No. de Identificacion",
    "Tipo y No de Identificacion",   # OCR without period
    "No. Identificacion",
    "No Identificacion",
    "Identificacion",
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
    "Cedula",
    "Cedula de Ciudadania",
    "RUT",
    "RUN",
    "DNI",
    "Documento",
    "No. Documento",
    "No Documento",
    "Afiliacion",
    "No. Afiliacion",
    "No Afiliacion",
    "Num. Afiliacion",
    "Num Afiliacion",
]


def extract_patient_id(text: str) -> Optional[str]:
    """Extract patient ID using all known aliases and Colombian patterns."""
    val = resolve_field_flexible(text, _PATIENT_ID_ALIASES)
    if val:
        # Often the value contains the doc-type prefix: "CC 24314628"
        m = re.search(r"(\d{5,12})", val)
        if m:
            return m.group(1)
        return val

    # Colombian CC / TI / CE / RC patterns: "CC 24314628", "C.C. 24314628"
    # Also handle "CC:24314628" or "CC. 24314628"
    m = re.search(
        r"(?:C\.?\s*C\.?|T\.?\s*I\.?|C\.?\s*E\.?|R\.?\s*C\.?|"
        r"C[eé]dula)\s*[.:\s]\s*(\d{5,12})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Bare long number near ID-related words (PaddleOCR may split across lines)
    m = re.search(
        r"(?:identificaci[oó]n|documento|cedula|afiliaci[oó]n)"
        r"[\s\S]{0,30}?(\d{5,12})",
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
    "Fecha Hora",                   # Colombian EPS: "Fecha Hora: 17/12/2024"
    "Fecha y Hora",
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
    "Medico Tratante",
    "Medico Responsable",
    "Medico Solicitante",
    "Medico",
    "Profesional",
    "Profesional de la Salud",
    "Dr",
    "Dra",
    "Validado por",
    "Informado por",
    "Firmado por",
    "Responsable",
    "Patologo",
    "Radiologo",
    "Nombre del Medico",
    "Nombre Medico",
    "Atendido por",
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
    """Extract the professional name from Spanish-format signatures.

    Handles both accented (PyMuPDF) and unaccented (PaddleOCR) text,
    and OCR artefacts like missing periods after ``Dr`` / ``Dra``.
    """
    patterns = [
        # "Informe Validado por: Dra. Fatima Mota"
        re.compile(
            r"(?:Informe\s+)?[Vv]alidado\s+por\s*:?\s*(.+?)$",
            re.MULTILINE,
        ),
        # "Firmado por: ..."
        re.compile(r"[Ff]irmado\s+por\s*:?\s*(.+?)$", re.MULTILINE),
        # "Informado por: ..."
        re.compile(r"[Ii]nformado\s+por\s*:?\s*(.+?)$", re.MULTILINE),
        # "Medico Tratante: ..." / "Medico: ..."
        re.compile(
            r"[Mm][eé]dico\s*(?:[Tt]ratante|[Rr]esponsable)?\s*:?\s*"
            r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.\s]+?)$",
            re.MULTILINE,
        ),
        # Standalone "Dr./Dra. Name" or "Dr Name" on its own line
        re.compile(
            r"^(Dra?\.?\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.\s]+?)$",
            re.MULTILINE,
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
    "Establecimiento",
    "Sede",
    "Unidad",
    "Prestador",
    "Nombre IPS",
    "Nombre EPS",
]


def extract_institution(text: str) -> Optional[str]:
    """Extract institution from structured fields or Spanish patterns.

    Tries the narrative/corporate-name detector first (catches ``SAS``,
    ``S.A.``, ``I.P.S.``, ``E.S.E.`` suffixes) then falls back to
    structured ``Key: Value`` aliases.
    """
    # Narrative / corporate names first — more specific
    val = extract_institution_spanish(text)
    if val:
        return val
    return resolve_field_flexible(text, _INSTITUTION_ALIASES)


def extract_institution_spanish(text: str) -> Optional[str]:
    """Detect institution names from narrative text.

    Patterns use character classes for accented/unaccented chars so that
    both PyMuPDF and PaddleOCR output are matched.
    """
    patterns = [
        # Standard institution names (accent-flex inline)
        re.compile(
            r"((?:Centro\s+(?:M[eé]dico|de\s+Salud)|Hospital|Cl[ií]nica|"
            r"Laboratorio|Instituto|Fundaci[oó]n|Sanatorio|Consultorio)"
            r"\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]+?)(?:\s{2,}|\n|$)",
            re.IGNORECASE,
        ),
        # Colombian corporate names: "MEIDE SAS", "COLSANITAS S.A."
        re.compile(
            r"([A-Z]{3,}(?:\s+[A-Z]+)*\s+"
            r"(?:S\.?A\.?S\.?|S\.?A\.?|LTDA|E\.?S\.?E\.?|I\.?P\.?S\.?))\b",
        ),
        # EPS/IPS names: "EPS FAMISANAR", "IPS MEIDE"
        re.compile(
            r"((?:EPS|IPS)\s+[A-Z][A-Za-zÀ-ÿ\s]+?)(?:\s{2,}|\n|$)",
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
    # Structured fields (accent-flex applied by extract_field)
    for key in ("Sexo al nacer", "Sexo", "Sex", "Genero"):
        raw = extract_field(text, key)
        if raw:
            return _normalize_sex(raw)

    # Inline OCR pattern: "Sexo Mujer", "Sexo M", "Sexo al nacer Femenino"
    # Also handles PaddleOCR where "Género" → "Genero"
    m = re.search(
        r"(?:Sexo(?:\s*al\s*nacer)?|Sex|G[eé]nero)\s*[:\s]\s*"
        r"(Masculino|Femenino|Mujer|Hombre|Male|Female|[MF])\b",
        text, re.IGNORECASE,
    )
    if m:
        return _normalize_sex(m.group(1))

    # OCR-mangled pattern: "Sexoalnacer Mur" / "Sexoalnacer Fem"
    # PaddleOCR may join words or truncate them
    m = re.search(
        r"Sexo\s*(?:al\s*nacer)?\s*(Mu[a-z]*|Fe[a-z]*|Ho[a-z]*|Ma[a-z]*|[MF])\b",
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

# Boundaries that should stop block capture (accent-flex inline)
_SECTION_BOUNDARY = re.compile(
    r"\n\s*(?:IMPRESI[OÓ]N|DIAGN[OÓ]STICO|CONCLUSI[OÓ]N|CONCEPTO|"
    r"INTERPRETACI[OÓ]N|Atentamente|Cordialmente|Saludos|Powered\s+by)\s*[,:]?",
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
    # Accent-flexible: Magnética/Magnetica, Radiografía/Radiografia, etc.
    m = re.search(
        r"(?:Resonancia\s+Magn[eé]tica|Radiograf[ií]a|Ecograf[ií]a|"
        r"Tomograf[ií]a|TAC|Electrocardiograma|Densitometr[ií]a|"
        r"Mamograf[ií]a|MRI|CT|X-Ray|Ultrasound)"
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

  