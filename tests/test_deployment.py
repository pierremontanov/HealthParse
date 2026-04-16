"""Tests for #6 Deployment – Single-Host Docker Setup.

Validates that nginx config, production compose overlay, deploy script,
and environment templates are well-structured and production-ready.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


# ═══════════════════════════════════════════════════════════════════
# File existence
# ═══════════════════════════════════════════════════════════════════


class TestDeployFilesExist:
    """All deployment artefacts must be present."""

    @pytest.mark.parametrize("rel_path", [
        "docker-compose.prod.yml",
        "deploy/nginx/nginx.conf",
        "deploy/.env.production",
        "deploy/deploy.sh",
    ])
    def test_file_exists(self, rel_path):
        assert (ROOT / rel_path).is_file(), f"{rel_path} not found"


# ═══════════════════════════════════════════════════════════════════
# Nginx configuration
# ═══════════════════════════════════════════════════════════════════


class TestNginxConfig:
    """Validate the reverse-proxy configuration."""

    @pytest.fixture()
    def conf(self) -> str:
        return (DEPLOY / "nginx" / "nginx.conf").read_text(encoding="utf-8")

    # -- Upstream & proxy ──────────────────────────────────────────

    def test_upstream_points_to_api(self, conf):
        assert "server api:8000" in conf

    def test_proxy_pass_to_upstream(self, conf):
        assert "proxy_pass http://dociq_api" in conf

    def test_sets_x_real_ip(self, conf):
        assert "X-Real-IP" in conf

    def test_sets_x_forwarded_for(self, conf):
        assert "X-Forwarded-For" in conf

    def test_sets_x_forwarded_proto(self, conf):
        assert "X-Forwarded-Proto" in conf

    # -- Rate limiting ─────────────────────────────────────────────

    def test_general_rate_limit_zone(self, conf):
        assert "zone=api_general" in conf

    def test_process_rate_limit_zone(self, conf):
        assert "zone=api_process" in conf

    def test_rate_limit_status_429(self, conf):
        assert "limit_req_status 429" in conf

    # -- Security headers ──────────────────────────────────────────

    @pytest.mark.parametrize("header", [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Referrer-Policy",
        "Content-Security-Policy",
    ])
    def test_security_header_present(self, conf, header):
        assert header in conf

    def test_x_frame_options_deny(self, conf):
        assert '"DENY"' in conf

    def test_nosniff(self, conf):
        assert '"nosniff"' in conf

    # -- Upload limits ─────────────────────────────────────────────

    def test_client_max_body_size(self, conf):
        assert "client_max_body_size" in conf
        match = re.search(r"client_max_body_size\s+(\d+)m", conf)
        assert match, "client_max_body_size should be set in megabytes"
        assert int(match.group(1)) >= 10, "Upload limit should be at least 10 MB"

    # -- Health endpoints no rate limit ────────────────────────────

    def test_health_location_block(self, conf):
        assert "location /health" in conf

    def test_ready_location_block(self, conf):
        assert "location /ready" in conf

    # -- Process endpoint timeout ──────────────────────────────────

    def test_process_proxy_read_timeout(self, conf):
        assert "proxy_read_timeout 300s" in conf

    # -- Logging ───────────────────────────────────────────────────

    def test_json_log_format(self, conf):
        assert "json_combined" in conf

    # -- TLS preparation ───────────────────────────────────────────

    def test_tls_config_commented(self, conf):
        """TLS block should exist as commented-out template."""
        assert "ssl_certificate" in conf
        assert "ssl_protocols" in conf

    def test_listens_on_port_80(self, conf):
        assert "listen 80" in conf


# ═══════════════════════════════════════════════════════════════════
# docker-compose.prod.yml
# ═══════════════════════════════════════════════════════════════════


class TestProdCompose:
    """Validate the production compose overlay."""

    @pytest.fixture()
    def compose(self) -> dict:
        return yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"))

    # -- Nginx service ─────────────────────────────────────────────

    def test_nginx_service_exists(self, compose):
        assert "nginx" in compose["services"]

    def test_nginx_image_is_alpine(self, compose):
        image = compose["services"]["nginx"]["image"]
        assert "nginx" in image
        assert "alpine" in image

    def test_nginx_depends_on_api(self, compose):
        deps = compose["services"]["nginx"]["depends_on"]
        assert "api" in deps

    def test_nginx_waits_for_healthy_api(self, compose):
        deps = compose["services"]["nginx"]["depends_on"]
        assert deps["api"]["condition"] == "service_healthy"

    def test_nginx_mounts_config_readonly(self, compose):
        volumes = compose["services"]["nginx"]["volumes"]
        conf_vol = [v for v in volumes if "nginx.conf" in v]
        assert conf_vol, "Nginx config must be mounted"
        assert ":ro" in conf_vol[0]

    def test_nginx_port_80(self, compose):
        ports = [str(p) for p in compose["services"]["nginx"]["ports"]]
        assert any("80" in p for p in ports)

    def test_nginx_log_rotation(self, compose):
        logging = compose["services"]["nginx"]["logging"]
        assert logging["driver"] == "json-file"
        assert "max-size" in logging["options"]

    # -- API overrides ─────────────────────────────────────────────

    def test_api_ports_removed(self, compose):
        api = compose["services"]["api"]
        assert api["ports"] == [], "API should not expose ports directly in prod"

    def test_api_exposes_8000_internally(self, compose):
        api = compose["services"]["api"]
        assert "8000" in [str(e) for e in api["expose"]]

    def test_api_log_format_json(self, compose):
        api = compose["services"]["api"]
        env = api["environment"]
        assert any("DOCIQ_LOG_FORMAT=json" in str(e) for e in env)

    def test_api_resource_limits(self, compose):
        resources = compose["services"]["api"]["deploy"]["resources"]
        assert "limits" in resources
        assert "memory" in resources["limits"]
        assert "cpus" in resources["limits"]

    def test_api_resource_reservations(self, compose):
        resources = compose["services"]["api"]["deploy"]["resources"]
        assert "reservations" in resources

    def test_api_log_rotation(self, compose):
        logging = compose["services"]["api"]["logging"]
        assert logging["driver"] == "json-file"
        assert "max-size" in logging["options"]
        assert "max-file" in logging["options"]

    # -- Named volume ──────────────────────────────────────────────

    def test_named_output_volume(self, compose):
        assert "dociq_output" in compose.get("volumes", {})

    def test_api_uses_named_volume(self, compose):
        volumes = compose["services"]["api"]["volumes"]
        assert any("dociq_output" in str(v) for v in volumes)


# ═══════════════════════════════════════════════════════════════════
# .env.production
# ═══════════════════════════════════════════════════════════════════


class TestEnvProduction:
    """Validate the production environment template."""

    @pytest.fixture()
    def env_lines(self) -> list[str]:
        return (DEPLOY / ".env.production").read_text(encoding="utf-8").splitlines()

    @pytest.fixture()
    def env_vars(self, env_lines) -> dict[str, str]:
        """Parse uncommented KEY=VALUE pairs."""
        result = {}
        for line in env_lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        return result

    @pytest.mark.parametrize("var", [
        "DOCIQ_LOG_LEVEL",
        "DOCIQ_LOG_FORMAT",
        "DOCIQ_API_WORKERS",
        "DOCIQ_OCR_DPI",
        "DOCIQ_OUTPUT_DIR",
        "NGINX_PORT",
    ])
    def test_key_variable_present(self, env_vars, var):
        assert var in env_vars, f"{var} should be set in .env.production"

    def test_log_format_is_json(self, env_vars):
        assert env_vars["DOCIQ_LOG_FORMAT"] == "json"

    def test_workers_at_least_two(self, env_vars):
        assert int(env_vars["DOCIQ_API_WORKERS"]) >= 2

    def test_reload_disabled(self, env_vars):
        assert env_vars.get("DOCIQ_API_RELOAD") == "false"

    def test_no_secrets_in_template(self, env_lines):
        """Template should not contain actual secret values."""
        for line in env_lines:
            low = line.lower()
            assert "password=" not in low or line.strip().startswith("#"), \
                "Passwords should not appear in env template"
            assert "secret=" not in low or line.strip().startswith("#"), \
                "Secrets should not appear in env template"


# ═══════════════════════════════════════════════════════════════════
# deploy.sh
# ═══════════════════════════════════════════════════════════════════


class TestDeployScript:
    """Validate the deployment bootstrap script."""

    @pytest.fixture()
    def script(self) -> str:
        return (DEPLOY / "deploy.sh").read_text(encoding="utf-8")

    def test_executable_permission(self):
        mode = os.stat(DEPLOY / "deploy.sh").st_mode
        assert mode & stat.S_IXUSR, "deploy.sh should be executable"

    def test_has_shebang(self, script):
        assert script.startswith("#!/")

    def test_set_euo_pipefail(self, script):
        assert "set -euo pipefail" in script

    def test_uses_both_compose_files(self, script):
        assert "docker-compose.yml" in script
        assert "docker-compose.prod.yml" in script

    def test_preflight_checks_docker(self, script):
        assert "command -v docker" in script

    def test_preflight_checks_compose(self, script):
        assert "docker compose version" in script

    def test_copies_env_template(self, script):
        assert ".env.production" in script

    def test_creates_output_dir(self, script):
        assert 'mkdir -p' in script
        assert "output" in script

    def test_health_check_loop(self, script):
        assert "/health" in script
        assert "max_retries" in script

    def test_supports_build_flag(self, script):
        assert "--build" in script

    def test_supports_down_flag(self, script):
        assert "--down" in script

    def test_supports_status_flag(self, script):
        assert "--status" in script

    def test_supports_logs_flag(self, script):
        assert "--logs" in script

    def test_supports_help_flag(self, script):
        assert "--help" in script
