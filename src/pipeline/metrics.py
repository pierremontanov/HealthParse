"""Performance metrics collection for the DocIQ pipeline.

Provides lightweight, thread-safe utilities for tracking timing, counts,
and throughput across pipeline stages.

Key components:

* :class:`Timer` – context-manager / decorator that records elapsed time
* :class:`MetricsCollector` – thread-safe aggregator for named metrics
* :func:`get_collector` – module-level singleton accessor

Usage
-----
    from src.pipeline.metrics import get_collector, Timer

    # As a context manager
    with Timer("ocr_extraction") as t:
        text = extract_text_from_image(path)
    # elapsed available as t.elapsed_ms

    # Record to the global collector
    collector = get_collector()
    collector.record("ocr_extraction", t.elapsed_ms)

    # Retrieve summary
    print(collector.summary())
    # {"ocr_extraction": {"count": 5, "total_ms": 1234, "mean_ms": 246.8, ...}}
"""

from __future__ import annotations

import functools
import logging
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "Timer",
    "MetricSnapshot",
    "MetricsCollector",
    "get_collector",
    "reset_collector",
    "timed",
]


# ═══════════════════════════════════════════════════════════════════
# Timer
# ═══════════════════════════════════════════════════════════════════

class Timer:
    """Lightweight stopwatch usable as a context manager.

    Attributes
    ----------
    name : str
        Label for this timing measurement.
    elapsed_ms : float
        Milliseconds elapsed between enter and exit (0.0 until exited).
    """

    __slots__ = ("name", "elapsed_ms", "_start")

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed_ms={self.elapsed_ms:.1f})"


# ═══════════════════════════════════════════════════════════════════
# Decorator
# ═══════════════════════════════════════════════════════════════════

def timed(name: str | None = None, *, log: bool = True) -> Callable:
    """Decorator that times a function and records to the global collector.

    Parameters
    ----------
    name : str, optional
        Metric name.  Defaults to ``<module>.<function>``.
    log : bool
        When ``True`` (default), emit a DEBUG log after each call.
    """

    def decorator(fn: Callable) -> Callable:
        metric_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(metric_name) as t:
                result = fn(*args, **kwargs)
            get_collector().record(metric_name, t.elapsed_ms)
            if log:
                logger.debug(
                    "%s completed in %.1f ms", metric_name, t.elapsed_ms
                )
            return result

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════
# Metric snapshot (immutable result)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MetricSnapshot:
    """Immutable statistical summary of a single named metric."""

    name: str
    count: int
    total_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════
# Collector
# ═══════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Thread-safe collector that aggregates named timing measurements.

    Stores raw values per metric name and computes statistical summaries
    on demand.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}
        self._errors: Dict[str, int] = {}

    # ── Recording ──────────────────────────────────────────────

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement for *name*."""
        with self._lock:
            self._data.setdefault(name, []).append(elapsed_ms)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named counter (e.g. processed docs, retries)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def record_error(self, name: str) -> None:
        """Increment the error counter for *name*."""
        with self._lock:
            self._errors[name] = self._errors.get(name, 0) + 1

    # ── Querying ───────────────────────────────────────────────

    def snapshot(self, name: str) -> Optional[MetricSnapshot]:
        """Compute a statistical snapshot for *name*.

        Returns ``None`` if no measurements have been recorded.
        """
        with self._lock:
            values = list(self._data.get(name, []))
        if not values:
            return None
        return self._compute(name, values)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Return statistical summaries for all recorded metrics."""
        with self._lock:
            names = list(self._data.keys())
            all_values = {n: list(v) for n, v in self._data.items()}
        result: Dict[str, Dict[str, Any]] = {}
        for name in names:
            snap = self._compute(name, all_values[name])
            result[name] = snap.as_dict()
        return result

    def counters(self) -> Dict[str, int]:
        """Return a copy of all named counters."""
        with self._lock:
            return dict(self._counters)

    def errors(self) -> Dict[str, int]:
        """Return a copy of all error counters."""
        with self._lock:
            return dict(self._errors)

    def report(self) -> Dict[str, Any]:
        """Full metrics report: timings, counters, and errors."""
        return {
            "timings": self.summary(),
            "counters": self.counters(),
            "errors": self.errors(),
        }

    # ── Lifecycle ──────────────────────────────────────────────

    def clear(self) -> None:
        """Reset all recorded data."""
        with self._lock:
            self._data.clear()
            self._counters.clear()
            self._errors.clear()

    @property
    def metric_names(self) -> List[str]:
        """List of metric names that have timing data."""
        with self._lock:
            return list(self._data.keys())

    # ── Internal ───────────────────────────────────────────────

    @staticmethod
    def _compute(name: str, values: List[float]) -> MetricSnapshot:
        n = len(values)
        sorted_vals = sorted(values)
        return MetricSnapshot(
            name=name,
            count=n,
            total_ms=sum(values),
            mean_ms=statistics.mean(values),
            min_ms=sorted_vals[0],
            max_ms=sorted_vals[-1],
            p50_ms=_percentile(sorted_vals, 50),
            p95_ms=_percentile(sorted_vals, 95),
            p99_ms=_percentile(sorted_vals, 99),
        )

    def __repr__(self) -> str:
        with self._lock:
            n_metrics = len(self._data)
            n_counters = len(self._counters)
        return f"MetricsCollector(metrics={n_metrics}, counters={n_counters})"


# ═══════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════

_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    """Return the module-level :class:`MetricsCollector` singleton."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def reset_collector() -> None:
    """Replace the singleton with a fresh instance. Useful for tests."""
    global _collector
    with _collector_lock:
        _collector = MetricsCollector()


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════

def _percentile(sorted_values: List[float], pct: float) -> float:
    """Compute the *pct*-th percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (pct / 100) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
