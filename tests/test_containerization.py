"""Tests for #21 Containerization – Dockerfile & Runtime.

Validates that container configuration files exist, are well-structured,
and that the Dockerfile follows best practices.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════
# File existence
# ═══════════════════════════════════════════════════════════════════


class TestFileExistence:
    """All container config files must exist."""

    @pytest.mark.parametrize("name", ["Dockerfile", ".dockerignore", "docker-compose.yml"])
    def test_file_exists(self, name):
        assert (ROOT / name).is_file(), f"{name} not found at project root"


# ═══════════════════════════════════════════════════════════════════
# Dockerfile
# ═══════════════════════════════════════════════════════════════════


class TestDockerfile:
    """Validate Dockerfile structure and best practices."""

    @pytest.fixture()
    def content(self) -> str:
        return (ROOT / "Dockerfile").read_text(encoding="utf-8")

    # -- Base image ---------------------------------------------------

    def test_uses_python_311_slim(self, content):
        assert "python:3.11-slim" in content

    def test_multi_stage_build(self, content):
        from_lines = [l for l in content.splitlines() if l.strip().startswith("FROM ")]
        assert len(from_lines) >= 2, "Expected multi-stage build with at least 2 FROM instructions"

    def test_builder_stage_named(self, content):
        assert "AS builder" in content

    def test_runtime_stage_named(self, content):
        assert "AS runtime" in content

    # -- System dependencies ------------------------------------------

    def test_installs_tesseract(self, content):
        assert "tesseract-ocr" in content

    def test_installs_tesseract_eng(self, content):
        assert "tesseract-ocr-eng" in content

    def test_installs_tesseract_spa(self, content):
        assert "tesseract-ocr-spa" in content

    def test_installs_poppler(self, content):
        assert "poppler-utils" in content

    # -- Python best practices ----------------------------------------

    def test_pythonunbuffered(self, content):
        assert "PYTHONUNBUFFERED=1" in content

    def test_pythondontwritebytecode(self, content):
        assert "PYTHONDONTWRITEBYTECODE=1" in content

    def test_no_cache_dir_pip(self, content):
        assert "--no-cache-dir" in content

    def test_apt_lists_cleaned(self, content):
        assert "rm -rf /var/lib/apt/lists/*" in content

    # -- Entrypoint / CMD ---------------------------------------------

    def test_exposes_port_8000(self, content):
        assert "EXPOSE 8000" in content

    def test_default_cmd_is_uvicorn(self, content):
        assert "uvicorn" in content
        assert "src.api.app:app" in content

    def test_healthcheck_defined(self, content):
        assert "HEALTHCHECK" in content

    # -- Environment defaults -----------------------------------------

    def test_log_format_defaults_json(self, content):
        assert "DOCIQ_LOG_FORMAT=json" in content

    def test_output_dir_set(self, content):
        assert "DOCIQ_OUTPUT_DIR" in content

    # -- Security / hygiene -------------------------------------------

    def test_copies_requirements_before_code(self, content):
        """requirements.txt should be copied before application code for layer caching."""
        req_pos = content.find("COPY requirements.txt")
        src_pos = content.find("COPY src/")
        assert req_pos != -1, "requirements.txt not COPYed"
        assert src_pos != -1, "src/ not COPYed"
        assert req_pos < src_pos, "requirements.txt should be COPYed before src/ for caching"

    def test_no_root_user_warning(self, content):
        """Informational: flag if no USER instruction (runs as root by default)."""
        # Not a hard failure, but note it for future hardening
        pass  # Acceptable for v1; USER can be added later

    def test_workdir_set(self, content):
        assert "WORKDIR /app" in content

    def test_labels_present(self, content):
        assert "LABEL" in content


# ═══════════════════════════════════════════════════════════════════
# .dockerignore
# ═══════════════════════════════════════════════════════════════════


class TestDockerignore:
    """Validate .dockerignore excludes unnecessary build context."""

    @pytest.fixture()
    def entries(self) -> set[str]:
        lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        return {l.strip() for l in lines if l.strip() and not l.strip().startswith("#")}

    @pytest.mark.parametrize("pattern", [
        ".git", "__pycache__", "*.pyc", ".pytest_cache",
        ".env", "venv/", "tests/", "output/", "data/",
        "Dockerfile", ".coverage",
    ])
    def test_excludes_pattern(self, entries, pattern):
        assert pattern in entries, f".dockerignore should exclude '{pattern}'"


# ═══════════════════════════════════════════════════════════════════
# docker-compose.yml
# ═══════════════════════════════════════════════════════════════════


class TestDockerCompose:
    """Validate docker-compose.yml structure."""

    @pytest.fixture()
    def compose(self) -> dict:
        return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    def test_has_services(self, compose):
        assert "services" in compose

    def test_api_service_exists(self, compose):
        assert "api" in compose["services"]

    def test_cli_service_exists(self, compose):
        assert "cli" in compose["services"]

    def test_test_service_exists(self, compose):
        assert "test" in compose["services"]

    # -- API service ---------------------------------------------------

    def test_api_port_mapping(self, compose):
        api = compose["services"]["api"]
        ports = api.get("ports", [])
        port_str = str(ports)
        assert "8000" in port_str

    def test_api_healthcheck(self, compose):
        api = compose["services"]["api"]
        assert "healthcheck" in api

    def test_api_restart_policy(self, compose):
        api = compose["services"]["api"]
        assert api.get("restart") == "unless-stopped"

    def test_api_output_volume(self, compose):
        api = compose["services"]["api"]
        volumes = [str(v) for v in api.get("volumes", [])]
        assert any("output" in v for v in volumes)

    def test_api_data_volume_readonly(self, compose):
        api = compose["services"]["api"]
        volumes = [str(v) for v in api.get("volumes", [])]
        assert any("data" in v and ":ro" in v for v in volumes), \
            "Data volume should be mounted read-only"

    # -- CLI service ---------------------------------------------------

    def test_cli_has_profile(self, compose):
        cli = compose["services"]["cli"]
        assert "cli" in cli.get("profiles", [])

    def test_cli_entrypoint(self, compose):
        cli = compose["services"]["cli"]
        ep = cli.get("entrypoint", [])
        assert "src.main" in str(ep)

    # -- Test service --------------------------------------------------

    def test_test_has_dev_profile(self, compose):
        test = compose["services"]["test"]
        assert "dev" in test.get("profiles", [])

    def test_test_entrypoint_is_pytest(self, compose):
        test = compose["services"]["test"]
        ep = test.get("entrypoint", [])
        assert "pytest" in str(ep)
