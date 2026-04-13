"""Tests for src.logging_config – structured logging setup."""

from __future__ import annotations

import io
import json
import logging

import pytest

from src.logging_config import JSONFormatter, setup_logging, TEXT_FORMAT


# ═══════════════════════════════════════════════════════════════════
# JSONFormatter
# ═══════════════════════════════════════════════════════════════════

class TestJSONFormatter:
    @pytest.fixture
    def formatter(self):
        return JSONFormatter()

    def _make_record(self, msg="hello", level=logging.INFO, **kwargs):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        return record

    def test_output_is_valid_json(self, formatter):
        record = self._make_record()
        line = formatter.format(record)
        data = json.loads(line)
        assert isinstance(data, dict)

    def test_contains_required_fields(self, formatter):
        record = self._make_record("test message")
        data = json.loads(formatter.format(record))
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data
        assert data["line"] == 42

    def test_timestamp_is_iso_format(self, formatter):
        record = self._make_record()
        data = json.loads(formatter.format(record))
        # Must parse as ISO-8601
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"])

    def test_debug_level(self, formatter):
        record = self._make_record(level=logging.DEBUG)
        data = json.loads(formatter.format(record))
        assert data["level"] == "DEBUG"

    def test_error_level(self, formatter):
        record = self._make_record(level=logging.ERROR)
        data = json.loads(formatter.format(record))
        assert data["level"] == "ERROR"

    def test_exception_included(self, formatter):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = self._make_record()
        record.exc_info = exc_info
        data = json.loads(formatter.format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "boom" in data["exception"]

    def test_extra_keys_merged(self, formatter):
        record = self._make_record()
        record.request_id = "abc-123"
        record.doc_type = "prescription"
        data = json.loads(formatter.format(record))
        assert data["request_id"] == "abc-123"
        assert data["doc_type"] == "prescription"

    def test_single_line_output(self, formatter):
        record = self._make_record("multi\nline\nmessage")
        line = formatter.format(record)
        # JSON serialisation escapes newlines → stays on one line
        assert "\n" not in line


# ═══════════════════════════════════════════════════════════════════
# setup_logging
# ═══════════════════════════════════════════════════════════════════

class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def _restore_root_logger(self):
        """Restore root logger state after each test."""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        yield
        root.handlers = original_handlers
        root.level = original_level

    def test_text_format(self):
        buf = io.StringIO()
        setup_logging(level="DEBUG", fmt="text", stream=buf)
        logging.getLogger("test.text").info("hello")
        output = buf.getvalue()
        assert "hello" in output
        assert "[INFO" in output

    def test_json_format(self):
        buf = io.StringIO()
        setup_logging(level="DEBUG", fmt="json", stream=buf)
        logging.getLogger("test.json").warning("structured")
        output = buf.getvalue().strip()
        data = json.loads(output)
        assert data["message"] == "structured"
        assert data["level"] == "WARNING"
        assert data["logger"] == "test.json"

    def test_respects_level(self):
        buf = io.StringIO()
        setup_logging(level="WARNING", fmt="text", stream=buf)
        logging.getLogger("test.level").info("should not appear")
        logging.getLogger("test.level").warning("should appear")
        output = buf.getvalue()
        assert "should not appear" not in output
        assert "should appear" in output

    def test_idempotent(self):
        """Calling setup_logging twice should not duplicate handlers."""
        buf = io.StringIO()
        setup_logging(level="INFO", fmt="text", stream=buf)
        setup_logging(level="INFO", fmt="text", stream=buf)
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_json_then_text_switches_formatter(self):
        buf = io.StringIO()
        setup_logging(level="INFO", fmt="json", stream=buf)
        setup_logging(level="INFO", fmt="text", stream=buf)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, JSONFormatter)


# ═══════════════════════════════════════════════════════════════════
# Config integration
# ═══════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    def test_log_format_setting_default(self):
        from src.config import DocIQSettings
        s = DocIQSettings()
        assert s.log_format == "text"

    def test_log_format_setting_json(self):
        from src.config import DocIQSettings
        s = DocIQSettings(log_format="json")
        assert s.log_format == "json"

    def test_log_format_setting_invalid_rejected(self):
        from src.config import DocIQSettings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocIQSettings(log_format="xml")
