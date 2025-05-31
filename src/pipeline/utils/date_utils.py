from datetime import datetime
from functools import wraps

def convert_to_iso8601(date_str: str) -> str:
    """
    Convert a date string from 'dd-mm-yyyy' to ISO 8601 'yyyy-mm-dd'.
    """
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").date().isoformat()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected dd-mm-yyyy.")

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
