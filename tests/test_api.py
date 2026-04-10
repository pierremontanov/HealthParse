"""Tests for the DocIQ FastAPI REST API."""

import io
import pytest
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
# Health endpoints
# ═══════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_matches_model(self, client):
        resp = client.get("/health")
        HealthResponse(**resp.json())  # validates against schema


class TestReadinessEndpoint:
    def test_ready_returns_200(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "ready" in data
        assert "checks" in data
        assert len(data["checks"]) >= 2

    def test_ready_checks_tesseract(self, client):
        resp = client.get("/ready")
        names = [c["name"] for c in resp.json()["checks"]]
        assert "tesseract" in names

    def test_ready_checks_inference_engine(self, client):
        resp = client.get("/ready")
        names = [c["name"] for c in resp.json()["checks"]]
        assert "inference_engine" in names

    def test_ready_matches_model(self, client):
        resp = client.get("/ready")
        ReadinessResponse(**resp.json())


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
