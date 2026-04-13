"""DocIQ centralised configuration.

Settings are loaded with the following priority (highest wins):

1. **Explicit constructor kwargs** (used by tests or programmatic callers)
2. **Environment variables** prefixed with ``DOCIQ_`` (e.g. ``DOCIQ_LOG_LEVEL=DEBUG``)
3. **``.env`` file** in the project root (loaded automatically by *pydantic-settings*)
4. **``config.yaml``** file when passed via CLI ``--config`` or ``DOCIQ_CONFIG_PATH``
5. **Built-in defaults** defined below

Usage
-----
    from src.config import settings          # singleton, ready to use
    print(settings.ocr_dpi)                  # 300

    # Reload from a YAML file at runtime
    settings = get_settings("config.yaml")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

__all__ = ["DocIQSettings", "get_settings", "settings"]


# ── Helpers ──────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file, with a fallback key:value parser when PyYAML is absent."""
    from src.pipeline.exceptions import ConfigFileNotFoundError, ConfigParseError

    p = Path(path)
    if not p.exists():
        raise ConfigFileNotFoundError(str(path))

    try:
        import yaml  # type: ignore[import-untyped]

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        data: Dict[str, Any] = {}  # type: ignore[no-redef]
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if value.lower() in ("true", "yes"):
                        value = True
                    elif value.lower() in ("false", "no"):
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.lower() in ("null", "~", ""):
                        value = None
                    data[key] = value

    # Normalise hyphens to underscores
    normalised = {k.replace("-", "_"): v for k, v in data.items()}

    # Backward-compatible aliases from CLI / old config keys
    _ALIASES = {
        "input": "input_dir",
        "format": "export_format",
    }
    for old_key, new_key in _ALIASES.items():
        if old_key in normalised and new_key not in normalised:
            normalised[new_key] = normalised.pop(old_key)
        elif old_key in normalised:
            normalised.pop(old_key)  # new_key already present, drop old

    # Translate no_inference → run_inference
    if "no_inference" in normalised:
        normalised.setdefault("run_inference", not normalised.pop("no_inference"))

    return normalised


# ── Settings model ───────────────────────────────────────────────

class DocIQSettings(BaseSettings):
    """Application-wide configuration for DocIQ.

    Environment variables are prefixed with ``DOCIQ_`` and use uppercase
    (e.g. ``DOCIQ_OCR_DPI=600``).  A ``.env`` file in the project root is
    loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCIQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────
    version: str = Field("1.0.0", description="DocIQ version string.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging verbosity."
    )
    supported_languages: Set[str] = Field(
        {"en", "es"}, description="ISO-639-1 codes for supported languages."
    )

    # ── Paths / I/O ──────────────────────────────────────────────
    input_dir: Optional[str] = Field(None, description="Default input path.")
    output_dir: str = Field("output", description="Default output directory.")
    temp_dir: Optional[str] = Field(
        None,
        description="Temporary directory for file uploads. Defaults to system temp.",
    )

    # ── Pipeline – OCR / Extraction ──────────────────────────────
    ocr_dpi: int = Field(300, ge=72, le=1200, description="DPI for PDF-to-image conversion.")
    ocr_lang: str = Field("eng+spa", description="Tesseract language pack string.")
    min_char_threshold: int = Field(
        10, ge=0, description="Minimum characters for a PDF to be considered text-based."
    )
    preprocessing_threshold: int = Field(
        120, ge=0, le=255, description="Binary threshold for image preprocessing."
    )

    # ── Pipeline – Language detection ────────────────────────────
    lang_detect_max_pages: int = Field(
        3, ge=1, description="Max PDF pages to sample for language detection."
    )
    lang_detect_min_chars: int = Field(
        120, ge=1, description="Minimum characters required for reliable detection."
    )

    # ── Pipeline – Concurrency ───────────────────────────────────
    max_workers: Optional[int] = Field(
        None,
        ge=1,
        description="Thread-pool cap for batch processing. None = auto (cpu_count * 2).",
    )
    page_timeout: int = Field(
        300,
        ge=10,
        description="Seconds to wait for a single page extraction before timing out.",
    )

    # ── Pipeline – Inference ─────────────────────────────────────
    run_inference: bool = Field(
        True,
        description="Run classification + NER. False = extraction-only.",
    )
    supported_extensions: Set[str] = Field(
        {".pdf", ".png", ".jpg", ".jpeg"},
        description="File extensions the pipeline will accept.",
    )

    # ── Export ────────────────────────────────────────────────────
    export_format: Literal["json", "csv", "fhir"] = Field(
        "json", description="Default export format."
    )

    # ── FHIR ─────────────────────────────────────────────────────
    fhir_bundle: bool = Field(
        True,
        description="Write a combined FHIR Bundle alongside individual resources.",
    )
    fhir_output_dir: str = Field(
        "dociq_fhir",
        description="Sub-directory name for FHIR output within output_dir.",
    )

    # ── API ───────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", description="FastAPI bind host.")
    api_port: int = Field(8000, ge=1, le=65535, description="FastAPI bind port.")
    api_workers: int = Field(1, ge=1, description="Uvicorn worker count.")
    api_reload: bool = Field(False, description="Enable Uvicorn auto-reload.")

    # ── Secrets / external paths (typically from .env) ────────────
    tesseract_cmd: Optional[str] = Field(
        None,
        description="Explicit path to the Tesseract binary. Overrides PATH lookup.",
    )
    poppler_path: Optional[str] = Field(
        None,
        description="Path to the Poppler bin directory (for pdf2image on Windows).",
    )

    # ── Validators ───────────────────────────────────────────────

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def _coerce_extensions(cls, v: Any) -> Any:
        if isinstance(v, list):
            return set(v)
        return v

    @field_validator("supported_languages", mode="before")
    @classmethod
    def _coerce_languages(cls, v: Any) -> Any:
        if isinstance(v, list):
            return set(v)
        return v


# ── Factory & singleton ──────────────────────────────────────────

def _env_overrides() -> Dict[str, Any]:
    """Return DOCIQ_* env vars as a dict (lowercased, prefix stripped)."""
    prefix = "DOCIQ_"
    result: Dict[str, Any] = {}
    import os
    for key, value in os.environ.items():
        if key.startswith(prefix):
            result[key[len(prefix):].lower()] = value
    return result


def get_settings(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> DocIQSettings:
    """Build a :class:`DocIQSettings` instance.

    Priority (highest first):

    1. **overrides** – explicit kwargs from calling code / CLI
    2. **environment** – ``DOCIQ_*`` env vars (+ ``.env`` file)
    3. **config_path** – YAML file values
    4. **defaults** – built into the model
    """
    yaml_values: Dict[str, Any] = {}
    if config_path is not None:
        yaml_values = _load_yaml(config_path)
        logger.debug("Loaded config from %s: %s", config_path, list(yaml_values.keys()))

    # Merge: yaml is the base, env vars override, explicit overrides win
    env_vars = _env_overrides()
    merged = {**yaml_values, **env_vars, **overrides}
    return DocIQSettings(**merged)


#: Module-level singleton.  Import this in production code.
settings: DocIQSettings = get_settings()
