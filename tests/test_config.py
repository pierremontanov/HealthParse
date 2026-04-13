"""Tests for src.config – DocIQ configuration system."""

import os
import textwrap

import pytest

from src.config import DocIQSettings, get_settings


# ── Defaults ─────────────────────────────────────────────────────


class TestDefaults:
    """Built-in defaults should be sensible out of the box."""

    def test_version(self):
        cfg = DocIQSettings()
        assert cfg.version == "1.0.0"

    def test_log_level(self):
        cfg = DocIQSettings()
        assert cfg.log_level == "INFO"

    def test_ocr_dpi(self):
        cfg = DocIQSettings()
        assert cfg.ocr_dpi == 300

    def test_output_dir(self):
        cfg = DocIQSettings()
        assert cfg.output_dir == "output"

    def test_export_format(self):
        cfg = DocIQSettings()
        assert cfg.export_format == "json"

    def test_api_port(self):
        cfg = DocIQSettings()
        assert cfg.api_port == 8000

    def test_run_inference_default_true(self):
        cfg = DocIQSettings()
        assert cfg.run_inference is True

    def test_supported_extensions_is_set(self):
        cfg = DocIQSettings()
        assert ".pdf" in cfg.supported_extensions
        assert ".png" in cfg.supported_extensions


# ── Constructor overrides ────────────────────────────────────────


class TestOverrides:
    """Keyword arguments to the constructor have highest priority."""

    def test_override_ocr_dpi(self):
        cfg = DocIQSettings(ocr_dpi=600)
        assert cfg.ocr_dpi == 600

    def test_override_log_level_normalises(self):
        cfg = DocIQSettings(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_override_export_format(self):
        cfg = DocIQSettings(export_format="fhir")
        assert cfg.export_format == "fhir"

    def test_override_run_inference_false(self):
        cfg = DocIQSettings(run_inference=False)
        assert cfg.run_inference is False

    def test_override_api_port(self):
        cfg = DocIQSettings(api_port=9000)
        assert cfg.api_port == 9000

    def test_override_tesseract_cmd(self):
        cfg = DocIQSettings(tesseract_cmd="/usr/local/bin/tesseract")
        assert cfg.tesseract_cmd == "/usr/local/bin/tesseract"


# ── Environment variable loading ─────────────────────────────────


class TestEnvVars:
    """DOCIQ_ prefixed env vars should be picked up."""

    def test_env_ocr_dpi(self, monkeypatch):
        monkeypatch.setenv("DOCIQ_OCR_DPI", "600")
        cfg = DocIQSettings()
        assert cfg.ocr_dpi == 600

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("DOCIQ_LOG_LEVEL", "debug")
        cfg = DocIQSettings()
        assert cfg.log_level == "DEBUG"

    def test_env_run_inference_false(self, monkeypatch):
        monkeypatch.setenv("DOCIQ_RUN_INFERENCE", "false")
        cfg = DocIQSettings()
        assert cfg.run_inference is False

    def test_env_api_port(self, monkeypatch):
        monkeypatch.setenv("DOCIQ_API_PORT", "9090")
        cfg = DocIQSettings()
        assert cfg.api_port == 9090


# ── YAML config loading via get_settings ─────────────────────────


class TestYAMLConfig:
    """get_settings() should load values from a YAML file."""

    def test_load_yaml_basic(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            ocr_dpi: 450
            log_level: WARNING
            output_dir: /tmp/dociq_out
        """))
        cfg = get_settings(config_path=str(cfg_file))
        assert cfg.ocr_dpi == 450
        assert cfg.log_level == "WARNING"
        assert cfg.output_dir == "/tmp/dociq_out"

    def test_yaml_values_overridden_by_kwargs(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("ocr_dpi: 450\n")
        cfg = get_settings(config_path=str(cfg_file), ocr_dpi=200)
        assert cfg.ocr_dpi == 200

    def test_yaml_values_overridden_by_env(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("ocr_dpi: 450\n")
        monkeypatch.setenv("DOCIQ_OCR_DPI", "600")
        cfg = get_settings(config_path=str(cfg_file))
        assert cfg.ocr_dpi == 600

    def test_missing_yaml_raises(self):
        from src.pipeline.exceptions import ConfigFileNotFoundError
        with pytest.raises(ConfigFileNotFoundError):
            get_settings(config_path="/nonexistent/config.yaml")

    def test_yaml_with_hyphens_normalised(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("log-level: DEBUG\n")
        cfg = get_settings(config_path=str(cfg_file))
        assert cfg.log_level == "DEBUG"

    def test_yaml_extra_keys_ignored(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            ocr_dpi: 300
            this_key_does_not_exist: hello
        """))
        cfg = get_settings(config_path=str(cfg_file))
        assert cfg.ocr_dpi == 300


# ── Validation ───────────────────────────────────────────────────


class TestValidation:
    """Field constraints should be enforced."""

    def test_ocr_dpi_too_low(self):
        with pytest.raises(Exception):
            DocIQSettings(ocr_dpi=10)

    def test_ocr_dpi_too_high(self):
        with pytest.raises(Exception):
            DocIQSettings(ocr_dpi=5000)

    def test_api_port_zero(self):
        with pytest.raises(Exception):
            DocIQSettings(api_port=0)

    def test_invalid_log_level(self):
        with pytest.raises(Exception):
            DocIQSettings(log_level="VERBOSE")

    def test_invalid_export_format(self):
        with pytest.raises(Exception):
            DocIQSettings(export_format="xml")

    def test_extensions_from_list(self):
        cfg = DocIQSettings(supported_extensions=[".pdf", ".tiff"])
        assert cfg.supported_extensions == {".pdf", ".tiff"}
