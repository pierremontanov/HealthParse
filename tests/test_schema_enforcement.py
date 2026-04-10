"""Tests for #23 – Output Schema Enforcement / Field Checks.

Covers:
- Prescription extra-field rejection (extra='forbid')
- validate_output() single-item re-validation
- validate_batch() batch gating (default and strict modes)
- export_results() pre-export validation integration
"""

import json

import pytest
from pydantic import ValidationError

from src.pipeline.validation.prescription_schema import Prescription
from src.pipeline.validation.validator import (
    SCHEMA_REGISTRY,
    OutputValidationResult,
    validate_batch,
    validate_output,
)
from src.pipeline.output_formatter import export_results


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def valid_result_item():
    """A correctly validated result dict."""
    return {
        "file": "test_result.pdf",
        "status": "ok",
        "language": "en",
        "method": "direct",
        "document_type": "result",
        "extracted_data": {
            "patient_name": "John Doe",
            "findings": "Normal study.",
            "exam_type": "CR",
            "exam_date": "2024-06-01",
            "professional": "Dr. Smith",
            "institution": "General Hospital",
        },
        "validated": True,
        "error": None,
        "elapsed_ms": 50,
    }


@pytest.fixture
def invalid_result_item():
    """A result dict whose extracted_data is missing required fields."""
    return {
        "file": "bad_result.pdf",
        "status": "ok",
        "language": "en",
        "method": "direct",
        "document_type": "result",
        "extracted_data": {
            # Missing patient_name and findings (both required)
            "exam_type": "CR",
        },
        "validated": False,
        "error": None,
        "elapsed_ms": 30,
    }


@pytest.fixture
def extraction_only_item():
    """An item with no inference (no document_type or extracted_data)."""
    return {
        "file": "raw.pdf",
        "status": "ok",
        "language": "es",
        "method": "ocr",
        "document_type": None,
        "extracted_data": None,
        "validated": False,
        "error": None,
        "elapsed_ms": 80,
    }


@pytest.fixture
def valid_prescription_item():
    return {
        "file": "rx.pdf",
        "status": "ok",
        "language": "es",
        "method": "direct",
        "document_type": "prescription",
        "extracted_data": {
            "patient_name": "Ana Torres",
            "items": [
                {"type": "medicine", "name": "Paracetamol 500mg"},
            ],
        },
        "validated": True,
        "error": None,
        "elapsed_ms": 40,
    }


# ── Prescription extra='forbid' ─────────────────────────────────

class TestPrescriptionExtraForbid:
    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="extra_field"):
            Prescription(
                items=[{"type": "medicine", "name": "Ibuprofen"}],
                extra_field="should not be here",
            )

    def test_valid_prescription_still_works(self):
        rx = Prescription(
            patient_name="Test Patient",
            items=[{"type": "medicine", "name": "Aspirin"}],
        )
        assert rx.patient_name == "Test Patient"


# ── SCHEMA_REGISTRY ──────────────────────────────────────────────

class TestSchemaRegistry:
    def test_all_types_registered(self):
        assert "result" in SCHEMA_REGISTRY
        assert "prescription" in SCHEMA_REGISTRY
        assert "clinical_history" in SCHEMA_REGISTRY

    def test_registry_length(self):
        assert len(SCHEMA_REGISTRY) == 3


# ── validate_output ──────────────────────────────────────────────

class TestValidateOutput:
    def test_valid_result_passes(self, valid_result_item):
        r = validate_output(valid_result_item)
        assert isinstance(r, OutputValidationResult)
        assert r.valid is True
        assert r.error is None

    def test_invalid_result_fails(self, invalid_result_item):
        r = validate_output(invalid_result_item)
        assert r.valid is False
        assert r.error is not None
        assert "validation error" in r.error

    def test_extraction_only_passes(self, extraction_only_item):
        r = validate_output(extraction_only_item)
        assert r.valid is True

    def test_no_extracted_data_passes(self):
        item = {"file": "x.pdf", "document_type": "result", "extracted_data": None}
        r = validate_output(item)
        assert r.valid is True

    def test_unknown_document_type_passes(self):
        item = {
            "file": "x.pdf",
            "document_type": "unknown_type",
            "extracted_data": {"some": "data"},
        }
        r = validate_output(item)
        assert r.valid is True
        assert "No schema registered" in r.error

    def test_valid_prescription_passes(self, valid_prescription_item):
        r = validate_output(valid_prescription_item)
        assert r.valid is True

    def test_prescription_extra_field_fails(self):
        item = {
            "file": "rx.pdf",
            "document_type": "prescription",
            "extracted_data": {
                "items": [{"type": "medicine", "name": "Aspirin"}],
                "bogus_field": "not allowed",
            },
        }
        r = validate_output(item)
        assert r.valid is False
        assert "bogus_field" in r.error

    def test_result_with_extra_field_fails(self):
        item = {
            "file": "res.pdf",
            "document_type": "result",
            "extracted_data": {
                "patient_name": "Jane",
                "findings": "Normal",
                "not_a_field": "oops",
            },
        }
        r = validate_output(item)
        assert r.valid is False


# ── validate_batch ───────────────────────────────────────────────

class TestValidateBatch:
    def test_default_mode_keeps_all(self, valid_result_item, invalid_result_item):
        items = [valid_result_item, invalid_result_item]
        out = validate_batch(items)
        assert len(out) == 2

    def test_default_mode_tags_invalid(self, invalid_result_item):
        out = validate_batch([invalid_result_item])
        assert "_validation_error" in out[0]

    def test_default_mode_no_tag_on_valid(self, valid_result_item):
        out = validate_batch([valid_result_item])
        assert "_validation_error" not in out[0]

    def test_strict_mode_drops_invalid(self, valid_result_item, invalid_result_item):
        items = [valid_result_item, invalid_result_item]
        out = validate_batch(items, strict=True)
        assert len(out) == 1
        assert out[0]["file"] == "test_result.pdf"

    def test_strict_mode_keeps_valid(self, valid_result_item):
        out = validate_batch([valid_result_item], strict=True)
        assert len(out) == 1

    def test_extraction_only_passes_through(self, extraction_only_item):
        out = validate_batch([extraction_only_item], strict=True)
        assert len(out) == 1

    def test_does_not_mutate_original(self, invalid_result_item):
        original = dict(invalid_result_item)
        validate_batch([invalid_result_item])
        # Original should not have _validation_error
        assert "_validation_error" not in original


# ── export_results with validate=True ────────────────────────────

class TestExportWithValidation:
    def test_json_export_with_validation(self, tmp_path, valid_result_item, invalid_result_item):
        items = [valid_result_item, invalid_result_item]
        path = export_results(items, output_dir=str(tmp_path), fmt="json", validate=True)
        # Both items should be exported (default non-strict)
        files = list((tmp_path / "dociq_results").glob("*.json"))
        assert len(files) == 2

    def test_json_export_strict_drops_invalid(self, tmp_path, valid_result_item, invalid_result_item):
        items = [valid_result_item, invalid_result_item]
        path = export_results(
            items, output_dir=str(tmp_path), fmt="json", validate=True, strict=True,
        )
        files = list((tmp_path / "dociq_results").glob("*.json"))
        assert len(files) == 1

    def test_csv_export_with_validation(self, tmp_path, valid_result_item, invalid_result_item):
        items = [valid_result_item, invalid_result_item]
        path = export_results(items, output_dir=str(tmp_path), fmt="csv", validate=True)
        assert (tmp_path / "dociq_results.csv").exists()

    def test_validation_error_field_in_json(self, tmp_path, invalid_result_item):
        export_results([invalid_result_item], output_dir=str(tmp_path), fmt="json")
        with open(tmp_path / "dociq_results" / "bad_result.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "_validation_error" in data

    def test_no_validation_when_disabled(self, tmp_path, invalid_result_item):
        export_results(
            [invalid_result_item], output_dir=str(tmp_path), fmt="json", validate=False,
        )
        with open(tmp_path / "dociq_results" / "bad_result.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "_validation_error" not in data
