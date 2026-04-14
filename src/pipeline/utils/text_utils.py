"""Text-processing utilities shared across the DocIQ pipeline.

All functions are pure (no side-effects) and operate on plain strings.
They are intentionally kept lightweight — heavy NLP belongs in the
extractor or inference layers.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "clean_text",
    "lowercase",
    "normalize_whitespace",
    "remove_numbers",
    "strip_non_ascii",
    "truncate",
]


def clean_text(text: str) -> str:
    """Normalize Unicode, collapse whitespace, and strip non-word characters.

    Retains letters, digits, whitespace, and a small set of punctuation
    (``. , : / -``) that is meaningful in clinical documents.
    """
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,:/-]", "", text)
    return text.strip()


def remove_numbers(text: str) -> str:
    """Remove all digit sequences from *text*."""
    return re.sub(r"\d+", "", text)


def lowercase(text: str) -> str:
    """Return *text* in lowercase."""
    return text.lower()


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_non_ascii(text: str) -> str:
    """Remove characters outside the Basic Latin + Latin-1 Supplement range.

    Useful for cleaning OCR artefacts that introduce garbage codepoints.
    """
    return re.sub(r"[^\x00-\xff]", "", text)


def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate *text* to *max_length* characters, appending *suffix*.

    If *text* is already within the limit it is returned unchanged.
    Truncation respects word boundaries when possible.
    """
    if len(text) <= max_length:
        return text
    # Cut at the last space before the limit
    cut = max_length - len(suffix)
    boundary = text.rfind(" ", 0, cut)
    if boundary <= 0:
        boundary = cut
    return text[:boundary] + suffix
