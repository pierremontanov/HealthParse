"""Tests for threaded save & export (#28).

Validates that export_json, export_fhir, and export_results correctly
use ThreadPoolExecutor for parallel writes and produce identical output
to single-threaded execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.pipeline.output_formatter import (
    export_csv,
    export_fhir,
    export_json,
    export_results,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_items(n: int = 10, *, with_inference: bool = False) -> List[Dict[str, Any]]:
    """Generate N synthetic result dicts."""
    items = []
    for i in range(n):
        item: Dict[str, Any] = {
            "file": f"doc_{i:03d}.pdf",
            "status": "ok",
            "text": f"Sample text for document {i}.",
            "language": "en",
            "method": "direct",
            "elapsed_ms": 100 + i,
        }
        if with_inference:
            item["document_type"] = "result"
            item["extracted_data"] = {
                "patient_name": f"Patient {i}",
                "date": "2024-01-15",
                "findings": f"Normal findings for patient {i}.",
            }
            item["validated"] = True
        items.append(item)
    return items


def _make_fhir_items(n: int = 5) -> List[Dict[str, Any]]:
    """Generate N items with extracted_data suitable for FHIR mapping."""
    doc_types = ["result", "prescription", "clinical_history"]
    items = []
    for i in range(n):
        dt = doc_types[i % 3]
        item: Dict[str, Any] = {
            "file": f"fhir_{i:03d}.pdf",
            "status": "ok",
            "document_type": dt,
        }
        if dt == "result":
            item["extracted_data"] = {
                "patient_name": f"Patient {i}",
                "exam_type": "Blood Panel",
                "exam_date": "2024-01-15",
                "findings": f"Finding {i}.",
                "professional": f"Dr. Lab {i}",
                "institution": "City Hospital",
            }
        elif dt == "prescription":
            item["extracted_data"] = {
                "patient_name": f"Patient {i}",
                "date": "2024-02-20",
                "items": [{"type": "medicine", "name": f"Drug{i}", "dosage": "10mg"}],
            }
        else:  # clinical_history
            item["extracted_data"] = {
                "patient_name": f"Patient {i}",
                "consultation_date": "2024-03-10",
                "doctor_name": f"Dr. Smith {i}",
            }
        items.append(item)
    return items


# ═══════════════════════════════════════════════════════════════════
# export_json – threaded
# ═══════════════════════════════════════════════════════════════════

class TestExportJsonThreaded:
    def test_creates_one_file_per_item(self, tmp_path: Path):
        items = _make_items(5)
        out = export_json(items, str(tmp_path), max_workers=4)
        json_files = list(Path(out).glob("*.json"))
        assert len(json_files) == 5

    def test_file_contents_valid_json(self, tmp_path: Path):
        items = _make_items(3)
        out = export_json(items, str(tmp_path), max_workers=2)
        for f in Path(out).glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            assert "file" in data
            assert "status" in data

    def test_sequential_matches_threaded(self, tmp_path: Path):
        items = _make_items(8)
        seq_dir = tmp_path / "seq"
        thr_dir = tmp_path / "thr"
        export_json(items, str(seq_dir), max_workers=1)
        export_json(items, str(thr_dir), max_workers=4)

        seq_files = sorted((seq_dir / "dociq_results").glob("*.json"))
        thr_files = sorted((thr_dir / "dociq_results").glob("*.json"))
        assert len(seq_files) == len(thr_files)

        for sf, tf in zip(seq_files, thr_files):
            assert sf.name == tf.name
            assert json.loads(sf.read_text()) == json.loads(tf.read_text())

    def test_empty_items(self, tmp_path: Path):
        out = export_json([], str(tmp_path), max_workers=2)
        json_files = list(Path(out).glob("*.json"))
        assert len(json_files) == 0

    def test_custom_dirname(self, tmp_path: Path):
        items = _make_items(2)
        out = export_json(items, str(tmp_path), dirname="my_output", max_workers=2)
        assert "my_output" in out

    def test_many_files_threaded(self, tmp_path: Path):
        """Stress test with many files to verify thread safety."""
        items = _make_items(50)
        out = export_json(items, str(tmp_path), max_workers=8)
        json_files = list(Path(out).glob("*.json"))
        assert len(json_files) == 50


# ═══════════════════════════════════════════════════════════════════
# export_fhir – threaded
# ═══════════════════════════════════════════════════════════════════

class TestExportFhirThreaded:
    def test_creates_fhir_files(self, tmp_path: Path):
        items = _make_fhir_items(6)
        out = export_fhir(items, str(tmp_path), max_workers=4)
        fhir_files = list(Path(out).glob("*_fhir.json"))
        assert len(fhir_files) == 6

    def test_bundle_created(self, tmp_path: Path):
        items = _make_fhir_items(3)
        out = export_fhir(items, str(tmp_path), bundle=True, max_workers=2)
        bundle_path = Path(out) / "bundle.json"
        assert bundle_path.exists()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle["resourceType"] == "Bundle"
        assert len(bundle["entry"]) == 3

    def test_no_bundle_when_disabled(self, tmp_path: Path):
        items = _make_fhir_items(3)
        out = export_fhir(items, str(tmp_path), bundle=False, max_workers=2)
        bundle_path = Path(out) / "bundle.json"
        assert not bundle_path.exists()

    def test_sequential_matches_threaded(self, tmp_path: Path):
        items = _make_fhir_items(6)
        seq_dir = tmp_path / "seq"
        thr_dir = tmp_path / "thr"

        export_fhir(items, str(seq_dir), bundle=False, max_workers=1)
        export_fhir(items, str(thr_dir), bundle=False, max_workers=4)

        seq_files = sorted((seq_dir / "dociq_fhir").glob("*_fhir.json"))
        thr_files = sorted((thr_dir / "dociq_fhir").glob("*_fhir.json"))
        assert len(seq_files) == len(thr_files)

        for sf, tf in zip(seq_files, thr_files):
            assert sf.name == tf.name
            sd = json.loads(sf.read_text())
            td = json.loads(tf.read_text())
            # FHIR resources contain generated UUIDs and timestamps
            # that differ between runs; compare structure keys and
            # stable fields only.
            assert sd["resourceType"] == td["resourceType"]
            assert sd.get("subject") == td.get("subject")

    def test_invalid_items_skipped(self, tmp_path: Path):
        items = _make_fhir_items(3) + [
            {"file": "bad.pdf", "document_type": "result", "extracted_data": {}},
        ]
        out = export_fhir(items, str(tmp_path), bundle=False, max_workers=2)
        fhir_files = list(Path(out).glob("*_fhir.json"))
        assert len(fhir_files) == 3  # bad item skipped

    def test_items_without_inference_skipped(self, tmp_path: Path):
        items = _make_items(3, with_inference=False)
        out = export_fhir(items, str(tmp_path), max_workers=2)
        fhir_files = list(Path(out).glob("*_fhir.json"))
        assert len(fhir_files) == 0

    def test_many_fhir_threaded(self, tmp_path: Path):
        """Stress test for FHIR threading."""
        items = _make_fhir_items(30)
        out = export_fhir(items, str(tmp_path), max_workers=8)
        fhir_files = list(Path(out).glob("*_fhir.json"))
        assert len(fhir_files) == 30


# ═══════════════════════════════════════════════════════════════════
# export_results – dispatcher with max_workers
# ═══════════════════════════════════════════════════════════════════

class TestExportResultsMaxWorkers:
    def test_json_with_workers(self, tmp_path: Path):
        items = _make_items(5)
        out = export_results(
            items, output_dir=str(tmp_path), fmt="json",
            validate=False, max_workers=2,
        )
        assert len(list(Path(out).glob("*.json"))) == 5

    def test_fhir_with_workers(self, tmp_path: Path):
        items = _make_fhir_items(3)
        out = export_results(
            items, output_dir=str(tmp_path), fmt="fhir",
            validate=False, max_workers=2,
        )
        assert len(list(Path(out).glob("*_fhir.json"))) == 3

    def test_csv_ignores_workers(self, tmp_path: Path):
        """CSV export has no threading; max_workers is silently ignored."""
        items = _make_items(5)
        out = export_results(
            items, output_dir=str(tmp_path), fmt="csv",
            validate=False, max_workers=4,
        )
        assert Path(out).exists()
        assert Path(out).suffix == ".csv"

    def test_default_workers_none(self, tmp_path: Path):
        """Works without explicit max_workers (backwards-compatible)."""
        items = _make_items(3)
        out = export_results(
            items, output_dir=str(tmp_path), fmt="json", validate=False,
        )
        assert len(list(Path(out).glob("*.json"))) == 3
