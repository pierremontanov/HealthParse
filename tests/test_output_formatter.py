"""Tests for src.pipeline.output_formatter – unified export module."""

import csv
import json

import pytest

from src.pipeline.output_formatter import (
    export_csv,
    export_fhir,
    export_json,
    export_results,
    format_document,
    save_json_output,
)
from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_result_model():
    return ResultSchema(
        patient_name="John Doe",
        patient_id="12345",
        age=50,
        sex="M",
        exam_type="CR",
        exam_date="01-06-2024",
        findings="Normal study.",
        professional="Dr. Smith",
        institution="General Hospital",
    )


@pytest.fixture
def sample_result_dicts():
    """Two result dicts as they come from DocumentResult.as_dict()."""
    return [
        {
            "file": "doc_a.pdf",
            "status": "ok",
            "language": "en",
            "method": "direct",
            "document_type": "result",
            "extracted_data": {
                "patient_name": "John Doe",
                "findings": "Normal.",
                "exam_type": "CR",
                "exam_date": "2024-06-01",
                "professional": "Dr. Smith",
                "institution": "General Hospital",
            },
            "validated": True,
            "error": None,
            "elapsed_ms": 42,
        },
        {
            "file": "doc_b.pdf",
            "status": "ok",
            "language": "es",
            "method": "ocr",
            "document_type": None,
            "extracted_data": None,
            "validated": False,
            "error": None,
            "elapsed_ms": 100,
        },
    ]


# ── format_document ──────────────────────────────────────────────

class TestFormatDocument:
    def test_returns_dict(self, sample_result_model):
        d = format_document(sample_result_model)
        assert isinstance(d, dict)
        assert d["patient_name"] == "John Doe"

    def test_excludes_none(self, sample_result_model):
        d = format_document(sample_result_model)
        # notes was not set, so it should not appear
        assert "notes" not in d


# ── save_json_output ─────────────────────────────────────────────

class TestSaveJsonOutput:
    def test_creates_file(self, tmp_path, sample_result_model):
        path = str(tmp_path / "out.json")
        result = save_json_output(sample_result_model, path)
        assert (tmp_path / "out.json").exists()
        assert result.endswith("out.json")

    def test_content_is_valid_json(self, tmp_path, sample_result_model):
        path = str(tmp_path / "out.json")
        save_json_output(sample_result_model, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["patient_name"] == "John Doe"

    def test_creates_parent_dirs(self, tmp_path, sample_result_model):
        path = str(tmp_path / "sub" / "dir" / "out.json")
        save_json_output(sample_result_model, path)
        assert (tmp_path / "sub" / "dir" / "out.json").exists()


# ── export_json ──────────────────────────────────────────────────

class TestExportJson:
    def test_creates_one_file_per_item(self, tmp_path, sample_result_dicts):
        path = export_json(sample_result_dicts, str(tmp_path))
        files = list((tmp_path / "dociq_results").glob("*.json"))
        assert len(files) == 2

    def test_custom_dirname(self, tmp_path, sample_result_dicts):
        export_json(sample_result_dicts, str(tmp_path), dirname="my_output")
        assert (tmp_path / "my_output").is_dir()

    def test_file_content(self, tmp_path, sample_result_dicts):
        export_json(sample_result_dicts, str(tmp_path))
        with open(tmp_path / "dociq_results" / "doc_a.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["file"] == "doc_a.pdf"
        assert data["status"] == "ok"


# ── export_csv ───────────────────────────────────────────────────

class TestExportCsv:
    def test_creates_csv(self, tmp_path, sample_result_dicts):
        path = export_csv(sample_result_dicts, str(tmp_path))
        assert path.endswith(".csv")
        assert (tmp_path / "dociq_results.csv").exists()

    def test_csv_has_header_and_rows(self, tmp_path, sample_result_dicts):
        export_csv(sample_result_dicts, str(tmp_path))
        with open(tmp_path / "dociq_results.csv", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert len(reader) == 3  # header + 2 rows

    def test_csv_flattens_extracted_data(self, tmp_path, sample_result_dicts):
        export_csv(sample_result_dicts, str(tmp_path))
        with open(tmp_path / "dociq_results.csv", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        # extracted_data should be a JSON string, not a dict
        cell = reader[0]["extracted_data"]
        parsed = json.loads(cell)
        assert parsed["patient_name"] == "John Doe"

    def test_empty_items_creates_empty_file(self, tmp_path):
        path = export_csv([], str(tmp_path))
        assert (tmp_path / "dociq_results.csv").exists()
        assert (tmp_path / "dociq_results.csv").read_text() == ""

    def test_custom_filename(self, tmp_path, sample_result_dicts):
        export_csv(sample_result_dicts, str(tmp_path), filename="custom.csv")
        assert (tmp_path / "custom.csv").exists()


# ── export_fhir ──────────────────────────────────────────────────

class TestExportFhir:
    def test_exports_recognised_types(self, tmp_path, sample_result_dicts):
        path = export_fhir(sample_result_dicts, str(tmp_path))
        fhir_dir = tmp_path / "dociq_fhir"
        files = list(fhir_dir.glob("*_fhir.json"))
        # Only doc_a has document_type="result" + extracted_data
        assert len(files) == 1

    def test_fhir_content(self, tmp_path, sample_result_dicts):
        export_fhir(sample_result_dicts, str(tmp_path))
        with open(tmp_path / "dociq_fhir" / "doc_a_fhir.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["resourceType"] == "DiagnosticReport"

    def test_skips_unclassified(self, tmp_path, sample_result_dicts):
        export_fhir(sample_result_dicts, str(tmp_path))
        fhir_dir = tmp_path / "dociq_fhir"
        # doc_b has no document_type, should be skipped
        assert not (fhir_dir / "doc_b_fhir.json").exists()


# ── export_results dispatcher ────────────────────────────────────

class TestExportResults:
    def test_dispatches_json(self, tmp_path, sample_result_dicts):
        path = export_results(sample_result_dicts, output_dir=str(tmp_path), fmt="json")
        assert (tmp_path / "dociq_results").is_dir()

    def test_dispatches_csv(self, tmp_path, sample_result_dicts):
        path = export_results(sample_result_dicts, output_dir=str(tmp_path), fmt="csv")
        assert path.endswith(".csv")

    def test_dispatches_fhir(self, tmp_path, sample_result_dicts):
        path = export_results(sample_result_dicts, output_dir=str(tmp_path), fmt="fhir")
        assert "fhir" in path

    def test_invalid_format_raises(self, tmp_path, sample_result_dicts):
        with pytest.raises(ValueError, match="Unsupported"):
            export_results(sample_result_dicts, output_dir=str(tmp_path), fmt="xml")
