"""Tests for the DocIQ FastAPI REST API."""

import io
import pytest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app, _engine, _get_engine
from src.api.models import HealthResponse, ReadinessResponse, ProcessingResponse


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton engine between tests."""
    import src.api.app as app_module
    app_module._engine = None
    yield
    app_module._engine = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def _mock_extraction():
    """Patch low-level extraction for upload tests."""
    with (
        patch("src.pipeline.process_folder.detect_pdf_language") as m_pdf_lang,
        patch("src.pipeline.process_folder.is_pdf_text_based") as m_text,
        patch("src.pipeline.process_folder.extract_text_directly") as m_direct,
        patch("src.pipeline.process_folder.extract_text_from_pdf_ocr") as m_ocr,
        patch("src.pipeline.process_folder.extract_text_from_image") as m_img,
        patch("src.pipeline.process_folder.detect_language") as m_lang,
    ):
        m_pdf_lang.return_value = SimpleNamespace(language="en", text_sample="sample")
        m_text.return_value = True
        m_direct.return_value = "PDF extracted text"
        m_ocr.return_value = "OCR text"
        m_img.return_value = "Image text"
        m_lang.return_value = "en"
        yield


# ═══════════════════════════════════════════════════════════════════
# Health endpoint
# ═══════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_includes_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_includes_iso_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data
        # Must be valid ISO-8601
        datetime.fromisoformat(data["timestamp"])

    def test_health_matches_model(self, client):
        resp = client.get("/health")
        HealthResponse(**resp.json())


# ═══════════════════════════════════════════════════════════════════
# Readiness endpoint
# ═══════════════════════════════════════════════════════════════════

class TestReadinessEndpoint:
    """Tests for GET /ready."""

    EXPECTED_CHECKS = {"tesseract", "poppler", "inference_engine", "config", "disk"}

    def test_ready_returns_200_or_503(self, client):
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)

    def test_ready_payload_shape(self, client):
        data = client.get("/ready").json()
        assert "ready" in data
        assert "checks" in data
        assert "total_elapsed_ms" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) >= 5

    def test_ready_check_names(self, client):
        names = {c["name"] for c in client.get("/ready").json()["checks"]}
        assert self.EXPECTED_CHECKS.issubset(names)

    def test_ready_each_check_has_elapsed(self, client):
        for check in client.get("/ready").json()["checks"]:
            assert "elapsed_ms" in check
            assert check["elapsed_ms"] >= 0

    def test_ready_total_elapsed_positive(self, client):
        data = client.get("/ready").json()
        assert data["total_elapsed_ms"] >= 0

    def test_ready_config_check_passes(self, client):
        """Config check should pass since settings already loaded."""
        checks = {c["name"]: c for c in client.get("/ready").json()["checks"]}
        assert checks["config"]["available"] is True

    def test_ready_disk_check_passes(self, client):
        checks = {c["name"]: c for c in client.get("/ready").json()["checks"]}
        assert checks["disk"]["available"] is True
        assert "MB free" in checks["disk"]["detail"]

    def test_ready_matches_model(self, client):
        ReadinessResponse(**client.get("/ready").json())

    def test_ready_returns_503_when_dependency_fails(self, client):
        """When a dependency is unavailable, /ready returns 503."""
        with patch("src.api.app._check_tesseract", return_value=(False, "missing")):
            resp = client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["ready"] is False

    def test_ready_returns_200_when_all_pass(self, client):
        """Force every check to pass → 200."""
        with (
            patch("src.api.app._check_tesseract", return_value=(True, "/usr/bin/tesseract")),
            patch("src.api.app._check_poppler", return_value=(True, "/usr/bin/pdftoppm")),
            patch("src.api.app._check_inference", return_value=(True, "registered types: result")),
            patch("src.api.app._check_config", return_value=(True, "settings loaded")),
            patch("src.api.app._check_disk", return_value=(True, "5000 MB free")),
        ):
            resp = client.get("/ready")
            assert resp.status_code == 200
            assert resp.json()["ready"] is True


# ═══════════════════════════════════════════════════════════════════
# Process endpoint
# ═══════════════════════════════════════════════════════════════════

class TestProcessEndpoint:
    def test_upload_pdf(self, client, _mock_extraction):
        import src.api.app as app_module
        app_module._engine = _make_no_inference_engine()

        file = io.BytesIO(b"fake pdf content")
        resp = client.post(
            "/process",
            files=[("files", ("test.pdf", file, "application/pdf"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["file"] == "test.pdf"
        assert data["results"][0]["status"] == "ok"
        assert "summary" in data

    def test_upload_image(self, client, _mock_extraction):
        import src.api.app as app_module
        app_module._engine = _make_no_inference_engine()

        file = io.BytesIO(b"fake png content")
        resp = client.post(
            "/process",
            files=[("files", ("scan.png", file, "image/png"))],
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["file"] == "scan.png"

    def test_unsupported_file_type(self, client):
        file = io.BytesIO(b"not a doc")
        resp = client.post(
            "/process",
            files=[("files", ("data.xlsx", file, "application/octet-stream"))],
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["status"] == "unsupported_type"
        assert ".xlsx" in result["error"]

    def test_multiple_files(self, client, _mock_extraction):
        import src.api.app as app_module
        app_module._engine = _make_no_inference_engine()

        files = [
            ("files", ("a.pdf", io.BytesIO(b"pdf1"), "application/pdf")),
            ("files", ("b.png", io.BytesIO(b"png1"), "image/png")),
            ("files", ("c.txt", io.BytesIO(b"txt1"), "text/plain")),
        ]
        resp = client.post("/process", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        statuses = [r["status"] for r in data["results"]]
        assert "unsupported_type" in statuses  # c.txt

    def test_summary_counts(self, client, _mock_extraction):
        import src.api.app as app_module
        app_module._engine = _make_no_inference_engine()

        files = [
            ("files", ("a.pdf", io.BytesIO(b"pdf"), "application/pdf")),
            ("files", ("b.txt", io.BytesIO(b"txt"), "text/plain")),
        ]
        resp = client.post("/process", files=files)
        summary = resp.json()["summary"]
        assert "unsupported_type" in summary

    def test_response_matches_model(self, client, _mock_extraction):
        import src.api.app as app_module
        app_module._engine = _make_no_inference_engine()

        file = io.BytesIO(b"fake pdf")
        resp = client.post(
            "/process",
            files=[("files", ("test.pdf", file, "application/pdf"))],
        )
        ProcessingResponse(**resp.json())


# ── Helpers ───────────────────────────────────────────────────────

def _make_no_inference_engine():
    """Create a DocIQEngine with inference disabled (extraction only)."""
    from src.pipeline.core_engine import DocIQEngine
    return DocIQEngine(run_inference=False)
