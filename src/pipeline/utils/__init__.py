"""Shared utility functions for the DocIQ pipeline.

Convenience imports::

    from src.pipeline.utils import (
        detect_language, is_english, is_spanish,
        clean_text, lowercase, normalize_whitespace, strip_non_ascii, truncate,
        normalize_dates,
    )
"""

from src.pipeline.utils.date_utils import normalize_dates
from src.pipeline.utils.language import (
    detect_language,
    is_english,
    is_spanish,
)
from src.pipeline.utils.text_utils import (
    clean_text,
    lowercase,
    normalize_whitespace,
    remove_numbers,
    strip_non_ascii,
    truncate,
)

__all__ = [
    # Language detection
    "detect_language",
    "is_english",
    "is_spanish",
    # Text processing
    "clean_text",
    "lowercase",
    "normalize_whitespace",
    "remove_numbers",
    "strip_non_ascii",
    "truncate",
    # Date utils
    "normalize_dates",
]
