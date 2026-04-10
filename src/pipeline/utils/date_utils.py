from datetime import datetime
from functools import wraps

import re

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def convert_to_iso8601(date_str: str) -> str:
    """
    Convert a date string from 'dd-mm-yyyy' to ISO 8601 'yyyy-mm-dd'.

    Strings that are already in ISO 8601 format are returned unchanged.
    Non-date sentinel values (e.g. "Unknown", "Not specified") are passed
    through as-is so that downstream validators can handle them.
    """
    if not date_str or not date_str.strip():
        return date_str

    # Already ISO 8601 — pass through
    if _ISO_DATE_RE.match(date_str.strip()):
        return date_str.strip()

    try:
        return datetime.strptime(date_str.strip(), "%d-%m-%Y").date().isoformat()
    except ValueError:
        # Return the original string rather than crashing.  Pydantic
        # validation will surface the issue if the value is truly invalid.
        return date_str

def normalize_dates(*fields):
    """
    Decorator that normalizes specified date fields to ISO 8601 format.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(data: dict, *args, **kwargs):
            for field in fields:
                if field in data and isinstance(data[field], str):
                    data[field] = convert_to_iso8601(data[field])
            return func(data, *args, **kwargs)
        return wrapper
    return decorator
