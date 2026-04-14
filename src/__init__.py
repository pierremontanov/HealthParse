"""DocIQ – AI-powered medical document classification and extraction engine.

Top-level convenience imports::

    from src import DocIQEngine, DocIQSettings, settings, setup_logging
"""

from src.config import DocIQSettings, get_settings, settings
from src.logging_config import setup_logging

__all__ = [
    "DocIQSettings",
    "get_settings",
    "settings",
    "setup_logging",
]
