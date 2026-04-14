"""Tests for src.pipeline.output_collector – thread-safe result aggregation (#27)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pytest

from src.pipeline.output_collector import OutputCollector


# ═══════════════════════════════════════════════════════════════════
# Basic API
# ═══════════════════════════════════════════════════════════════════

class TestOutputCollectorBasic:
    def test_add_single(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        assert c.count == 1

    def test_add_many(self):
        c = OutputCollector()
        c.add_many([
            {"file": "a.pdf", "status": "ok"},
            {"file": "b.pdf", "status": "ok"},
        ])
        assert c.count == 2

    def test_results_returns_snapshot(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        snap = c.results()
        c.add({"file": "b.pdf", "status": "ok"})
        # snapshot should be independent
        assert len(snap) == 1
        assert c.count == 2

    def test_results_sort_by(self):
        c = OutputCollector()
        c.add({"file": "c.pdf", "status": "ok"})
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "ok"})
        sorted_results = c.results(sort_by="file")
        assert [r["file"] for r in sorted_results] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_empty_collector(self):
        c = OutputCollector()
        assert c.count == 0
        assert c.ok_count == 0
        assert c.error_count == 0
        assert c.results() == []
        assert c.summary() == {}


# ═══════════════════════════════════════════════════════════════════
# Status counting
# ═══════════════════════════════════════════════════════════════════

class TestStatusCounting:
    def test_ok_count(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "ok"})
        c.add({"file": "c.pdf", "status": "extraction_error"})
        assert c.ok_count == 2

    def test_error_count(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "extraction_error"})
        c.add({"file": "c.pdf", "status": "inference_error"})
        assert c.error_count == 2

    def test_summary(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "ok"})
        c.add({"file": "c.pdf", "status": "extraction_error"})
        assert c.summary() == {"ok": 2, "extraction_error": 1}

    def test_missing_status_defaults_to_unknown(self):
        c = OutputCollector()
        c.add({"file": "a.pdf"})
        assert c.summary() == {"unknown": 1}


# ═══════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════

class TestCallbacks:
    def test_on_result_called(self):
        received: List[Dict[str, Any]] = []
        c = OutputCollector(on_result=lambda r: received.append(r))
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "ok"})
        assert len(received) == 2
        assert received[0]["file"] == "a.pdf"

    def test_on_result_called_for_add_many(self):
        received: List[Dict[str, Any]] = []
        c = OutputCollector(on_result=lambda r: received.append(r))
        c.add_many([
            {"file": "a.pdf", "status": "ok"},
            {"file": "b.pdf", "status": "ok"},
        ])
        assert len(received) == 2

    def test_callback_exception_does_not_break_add(self):
        """A failing callback should not prevent the result from being stored."""
        def bad_callback(r):
            raise ValueError("boom")

        c = OutputCollector(on_result=bad_callback)
        c.add({"file": "a.pdf", "status": "ok"})
        assert c.count == 1  # result still stored


# ═══════════════════════════════════════════════════════════════════
# Clear
# ═══════════════════════════════════════════════════════════════════

class TestClear:
    def test_clear_empties_everything(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.clear()
        assert c.count == 0
        assert c.summary() == {}

    def test_add_after_clear(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.clear()
        c.add({"file": "b.pdf", "status": "extraction_error"})
        assert c.count == 1
        assert c.ok_count == 0
        assert c.error_count == 1


# ═══════════════════════════════════════════════════════════════════
# Dunder methods
# ═══════════════════════════════════════════════════════════════════

class TestDunderMethods:
    def test_len(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        assert len(c) == 1

    def test_iter(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "ok"})
        items = list(c)
        assert len(items) == 2

    def test_repr(self):
        c = OutputCollector()
        c.add({"file": "a.pdf", "status": "ok"})
        c.add({"file": "b.pdf", "status": "extraction_error"})
        r = repr(c)
        assert "count=2" in r
        assert "ok=1" in r
        assert "errors=1" in r


# ═══════════════════════════════════════════════════════════════════
# Thread safety
# ═══════════════════════════════════════════════════════════════════

class TestThreadSafety:
    def test_concurrent_adds(self):
        """Multiple threads adding simultaneously should not lose results."""
        c = OutputCollector()
        n_threads = 8
        n_per_thread = 100

        def worker(thread_id: int):
            for i in range(n_per_thread):
                c.add({"file": f"t{thread_id}_{i}.pdf", "status": "ok"})

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(n_threads)]
            for f in futures:
                f.result()

        assert c.count == n_threads * n_per_thread
        assert c.ok_count == n_threads * n_per_thread

    def test_concurrent_add_many(self):
        c = OutputCollector()
        n_threads = 4
        batch_size = 50

        def worker(thread_id: int):
            batch = [
                {"file": f"t{thread_id}_{i}.pdf", "status": "ok"}
                for i in range(batch_size)
            ]
            c.add_many(batch)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(n_threads)]
            for f in futures:
                f.result()

        assert c.count == n_threads * batch_size

    def test_concurrent_reads_and_writes(self):
        """Readers and writers running concurrently should not raise."""
        c = OutputCollector()
        errors: List[Exception] = []

        def writer():
            for i in range(100):
                c.add({"file": f"w_{i}.pdf", "status": "ok"})

        def reader():
            for _ in range(100):
                try:
                    _ = c.results()
                    _ = c.count
                    _ = c.summary()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert c.count == 100

    def test_snapshot_isolation(self):
        """A snapshot taken mid-flight should be a stable copy."""
        c = OutputCollector()
        c.add({"file": "initial.pdf", "status": "ok"})
        snap = c.results()

        # Mutating the internal list after snapshot should not affect it
        c.add({"file": "later.pdf", "status": "ok"})
        assert len(snap) == 1
