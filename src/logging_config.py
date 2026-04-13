"""DocIQ structured logging configuration.

Provides two output formats controlled by ``DocIQSettings.log_format``:

* **text** (default) – human-readable lines for local development::

      2024-03-15 09:12:34 [INFO   ] src.pipeline.core_engine: DocIQEngine initialised

* **json** – one JSON object per line for production / log aggregators::

      {"timestamp":"2024-03-15T09:12:34.123Z","level":"INFO","logger":"src.pipeline.core_engine","message":"DocIQEngine initialised"}

Usage
-----
    from src.logging_config import setup_logging
    setup_logging(level="DEBUG", fmt="json")
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Literal, Optional


__all__ = ["setup_logging", "JSONFormatter", "TEXT_FORMAT", "TEXT_DATEFMT"]

TEXT_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
TEXT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields: ``timestamp``, ``level``, ``logger``, ``message``,
    ``module``, ``function``, ``line``.  If the record contains an
    exception the formatted traceback is added as ``exception``.
    Extra keys attached to the record (via ``extra=`` or a
    :class:`logging.LoggerAdapter`) are merged at the top level.
    """

    # Keys that are part of the standard LogRecord and should not
    # appear in the "extra" section of the JSON output.
    _RESERVED = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        | {"message", "asctime"}
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge extra keys the caller attached to the record.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    fmt: Literal["text", "json"] = "text",
    stream: Optional[object] = None,
) -> None:
    """Configure the root logger for the entire application.

    Parameters
    ----------
    level:
        Standard Python log-level name (``DEBUG``, ``INFO``, etc.).
    fmt:
        ``"text"`` for human-readable output, ``"json"`` for structured
        JSON lines.
    stream:
        Output stream.  Defaults to ``sys.stderr``.
    """
    root = logging.getLogger()

    # Remove any existing handlers so this is idempotent.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(TEXT_FORMAT, datefmt=TEXT_DATEFMT)
        )

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
