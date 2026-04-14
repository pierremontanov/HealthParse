"""Thread-safe output collector for batch pipeline processing (#27).

:class:`OutputCollector` aggregates :class:`DocumentResult` dicts from
concurrent worker threads without external locking.  It supports optional
progress callbacks, real-time status counting, and snapshot-safe iteration.

Usage
-----
    from src.pipeline.output_collector import OutputCollector

    collector = OutputCollector(on_result=lambda r: print(r["file"]))
    collector.add({"file": "a.pdf", "status": "ok", ...})
    collector.add({"file": "b.pdf", "status": "extraction_error", ...})

    print(collector.count)        # 2
    print(collector.summary())    # {"ok": 1, "extraction_error": 1}
    results = collector.results() # thread-safe snapshot
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["OutputCollector"]


class OutputCollector:
    """Thread-safe container for pipeline results.

    Parameters
    ----------
    on_result : callable, optional
        Called with the result dict each time :meth:`add` is invoked.
        Useful for progress bars, live logging, or streaming output.
        The callback is invoked **inside** the lock, so keep it fast.
    """

    def __init__(
        self,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._on_result = on_result
        self._status_counts: Dict[str, int] = {}

    # ── Core API ─────────────────────────────────────────────────

    def add(self, result: Dict[str, Any]) -> None:
        """Append a result dict in a thread-safe manner.

        Parameters
        ----------
        result : dict
            A document-processing result.  Must contain at least a
            ``"status"`` key.
        """
        with self._lock:
            self._results.append(result)
            status = result.get("status", "unknown")
            self._status_counts[status] = self._status_counts.get(status, 0) + 1

            if self._on_result is not None:
                try:
                    self._on_result(result)
                except Exception as exc:
                    logger.warning("on_result callback raised: %s", exc)

    def add_many(self, results: List[Dict[str, Any]]) -> None:
        """Append multiple results atomically."""
        with self._lock:
            for result in results:
                self._results.append(result)
                status = result.get("status", "unknown")
                self._status_counts[status] = self._status_counts.get(status, 0) + 1

            if self._on_result is not None:
                for result in results:
                    try:
                        self._on_result(result)
                    except Exception as exc:
                        logger.warning("on_result callback raised: %s", exc)

    # ── Read API ─────────────────────────────────────────────────

    def results(self, *, sort_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a snapshot of all collected results.

        Parameters
        ----------
        sort_by : str, optional
            Key to sort by (e.g. ``"file"``).  ``None`` preserves
            insertion order.
        """
        with self._lock:
            snapshot = list(self._results)
        if sort_by:
            snapshot.sort(key=lambda r: r.get(sort_by, ""))
        return snapshot

    @property
    def count(self) -> int:
        """Total number of results collected so far."""
        with self._lock:
            return len(self._results)

    @property
    def ok_count(self) -> int:
        """Number of results with ``status == "ok"``."""
        with self._lock:
            return self._status_counts.get("ok", 0)

    @property
    def error_count(self) -> int:
        """Number of results with a non-ok status."""
        with self._lock:
            return sum(
                v for k, v in self._status_counts.items() if k != "ok"
            )

    def summary(self) -> Dict[str, int]:
        """Return a ``{status: count}`` dictionary."""
        with self._lock:
            return dict(self._status_counts)

    def clear(self) -> None:
        """Discard all collected results."""
        with self._lock:
            self._results.clear()
            self._status_counts.clear()

    # ── Dunder helpers ───────────────────────────────────────────

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self.results())

    def __repr__(self) -> str:
        return (
            f"OutputCollector(count={self.count}, "
            f"ok={self.ok_count}, errors={self.error_count})"
        )
