"""Tests for src.pipeline.metrics – Timer, MetricsCollector, timed decorator."""
import threading
import time

import pytest

from src.pipeline.metrics import (
    MetricSnapshot,
    MetricsCollector,
    Timer,
    _percentile,
    get_collector,
    reset_collector,
    timed,
)


# ═══════════════════════════════════════════════════════════════════
# Timer
# ═══════════════════════════════════════════════════════════════════


class TestTimer:
    """Tests for the Timer context-manager stopwatch."""

    def test_elapsed_ms_is_zero_before_use(self):
        t = Timer("unused")
        assert t.elapsed_ms == 0.0

    def test_measures_elapsed_time(self):
        with Timer("sleep") as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40  # generous lower bound
        assert t.elapsed_ms < 500  # generous upper bound

    def test_name_attribute(self):
        t = Timer("my_stage")
        assert t.name == "my_stage"

    def test_default_name_is_empty(self):
        t = Timer()
        assert t.name == ""

    def test_repr(self):
        t = Timer("r")
        r = repr(t)
        assert "Timer" in r
        assert "r" in r

    def test_reusable(self):
        t = Timer("reuse")
        with t:
            time.sleep(0.01)
        first = t.elapsed_ms
        with t:
            time.sleep(0.01)
        second = t.elapsed_ms
        # second measurement is independent
        assert second > 0
        assert first > 0


# ═══════════════════════════════════════════════════════════════════
# MetricSnapshot
# ═══════════════════════════════════════════════════════════════════


class TestMetricSnapshot:
    """Tests for the frozen MetricSnapshot dataclass."""

    def _make(self, **overrides):
        defaults = dict(
            name="test",
            count=10,
            total_ms=1000.0,
            mean_ms=100.0,
            min_ms=50.0,
            max_ms=200.0,
            p50_ms=95.0,
            p95_ms=180.0,
            p99_ms=195.0,
        )
        defaults.update(overrides)
        return MetricSnapshot(**defaults)

    def test_frozen(self):
        snap = self._make()
        with pytest.raises(AttributeError):
            snap.count = 99

    def test_as_dict_keys(self):
        snap = self._make()
        d = snap.as_dict()
        expected_keys = {
            "name", "count", "total_ms", "mean_ms",
            "min_ms", "max_ms", "p50_ms", "p95_ms", "p99_ms",
        }
        assert set(d.keys()) == expected_keys

    def test_as_dict_rounds_floats(self):
        snap = self._make(total_ms=1000.12345)
        d = snap.as_dict()
        assert d["total_ms"] == 1000.12


# ═══════════════════════════════════════════════════════════════════
# MetricsCollector
# ═══════════════════════════════════════════════════════════════════


class TestMetricsCollector:
    """Tests for the thread-safe MetricsCollector."""

    def test_record_and_snapshot(self):
        c = MetricsCollector()
        c.record("stage_a", 100.0)
        c.record("stage_a", 200.0)
        snap = c.snapshot("stage_a")
        assert snap is not None
        assert snap.count == 2
        assert snap.mean_ms == 150.0

    def test_snapshot_returns_none_when_empty(self):
        c = MetricsCollector()
        assert c.snapshot("nonexistent") is None

    def test_increment_counter(self):
        c = MetricsCollector()
        c.increment("docs")
        c.increment("docs")
        c.increment("docs", amount=3)
        assert c.counters()["docs"] == 5

    def test_record_error(self):
        c = MetricsCollector()
        c.record_error("ocr")
        c.record_error("ocr")
        assert c.errors()["ocr"] == 2

    def test_summary_returns_all_metrics(self):
        c = MetricsCollector()
        c.record("a", 10.0)
        c.record("b", 20.0)
        s = c.summary()
        assert "a" in s
        assert "b" in s
        assert s["a"]["count"] == 1

    def test_report_combines_all(self):
        c = MetricsCollector()
        c.record("x", 5.0)
        c.increment("y")
        c.record_error("z")
        r = c.report()
        assert "timings" in r
        assert "counters" in r
        assert "errors" in r
        assert r["counters"]["y"] == 1
        assert r["errors"]["z"] == 1

    def test_clear_resets_everything(self):
        c = MetricsCollector()
        c.record("a", 1.0)
        c.increment("b")
        c.record_error("c")
        c.clear()
        assert c.summary() == {}
        assert c.counters() == {}
        assert c.errors() == {}

    def test_metric_names(self):
        c = MetricsCollector()
        c.record("alpha", 1.0)
        c.record("beta", 2.0)
        names = c.metric_names
        assert set(names) == {"alpha", "beta"}

    def test_repr(self):
        c = MetricsCollector()
        c.record("m", 1.0)
        c.increment("c")
        r = repr(c)
        assert "metrics=1" in r
        assert "counters=1" in r

    def test_snapshot_statistics(self):
        c = MetricsCollector()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            c.record("s", v)
        snap = c.snapshot("s")
        assert snap.count == 5
        assert snap.min_ms == 10.0
        assert snap.max_ms == 50.0
        assert snap.mean_ms == 30.0
        assert snap.total_ms == 150.0

    def test_thread_safety(self):
        """Multiple threads recording concurrently should not lose data."""
        c = MetricsCollector()
        n_threads = 8
        n_records = 200
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for i in range(n_records):
                c.record("concurrent", float(i))
                c.increment("count")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = c.snapshot("concurrent")
        assert snap.count == n_threads * n_records
        assert c.counters()["count"] == n_threads * n_records


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════


class TestSingleton:
    """Tests for the module-level collector singleton."""

    def test_get_collector_returns_same_instance(self):
        reset_collector()
        a = get_collector()
        b = get_collector()
        assert a is b

    def test_reset_collector_creates_new_instance(self):
        reset_collector()
        a = get_collector()
        reset_collector()
        b = get_collector()
        assert a is not b

    def test_reset_clears_data(self):
        reset_collector()
        c = get_collector()
        c.record("temp", 42.0)
        reset_collector()
        c2 = get_collector()
        assert c2.snapshot("temp") is None


# ═══════════════════════════════════════════════════════════════════
# timed decorator
# ═══════════════════════════════════════════════════════════════════


class TestTimedDecorator:
    """Tests for the @timed decorator."""

    def test_records_to_global_collector(self):
        reset_collector()

        @timed("test_fn", log=False)
        def do_work():
            time.sleep(0.01)
            return 42

        result = do_work()
        assert result == 42
        snap = get_collector().snapshot("test_fn")
        assert snap is not None
        assert snap.count == 1
        assert snap.mean_ms >= 5

    def test_default_name_uses_qualname(self):
        reset_collector()

        @timed(log=False)
        def my_func():
            pass

        my_func()
        names = get_collector().metric_names
        assert any("my_func" in n for n in names)

    def test_preserves_function_metadata(self):
        @timed("meta_test", log=False)
        def documented():
            """My docstring."""
            pass

        assert documented.__name__ == "documented"
        assert "My docstring" in (documented.__doc__ or "")

    def test_multiple_calls_accumulate(self):
        reset_collector()

        @timed("accum", log=False)
        def noop():
            pass

        for _ in range(5):
            noop()

        snap = get_collector().snapshot("accum")
        assert snap.count == 5


# ═══════════════════════════════════════════════════════════════════
# _percentile helper
# ═══════════════════════════════════════════════════════════════════


class TestPercentile:
    """Tests for the _percentile utility function."""

    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_median_odd(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(vals, 50) == 30.0

    def test_p0_returns_min(self):
        vals = [5.0, 10.0, 15.0]
        assert _percentile(vals, 0) == 5.0

    def test_p100_returns_max(self):
        vals = [5.0, 10.0, 15.0]
        assert _percentile(vals, 100) == 15.0

    def test_interpolation(self):
        vals = [0.0, 100.0]
        assert _percentile(vals, 50) == 50.0
        assert _percentile(vals, 25) == 25.0
        assert _percentile(vals, 75) == 75.0

    def test_p95_large_dataset(self):
        vals = sorted(float(i) for i in range(100))
        p95 = _percentile(vals, 95)
        assert 93.0 <= p95 <= 96.0
