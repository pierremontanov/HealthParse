"""Shared helpers for rule-based entity extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def extract_field(text: str, key: str) -> Optional[str]:
    """Extract a single ``Key: Value`` field from *text*.

    Matches lines of the form ``Key: <value>`` and returns the stripped
    value.  Returns ``None`` when the key is not found.
    """
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        return value if value else None
    return None


def extract_date(text: str, key: str) -> Optional[str]:
    """Extract a date value from a ``Key: <date>`` line.

    Supports ISO-8601 (``yyyy-mm-dd``), European (``dd-mm-yyyy``), and
    slash-separated variants.  Always returns the raw string as found;
    downstream normalisation happens in the validator decorator.
    """
    pattern = re.compile(
        rf"^{re.escape(key)}\s*:\s*(\d{{2,4}}[\-/]\d{{2}}[\-/]\d{{2,4}})",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
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
