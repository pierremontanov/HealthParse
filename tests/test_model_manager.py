"""Tests for src.pipeline.model_manager – model loading & saving (#22)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.pipeline.model_manager import ModelManager, ModelMeta, _FORMAT_VERSION


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture()
def saved_model(model_dir: Path) -> Path:
    """Save a model artefact and return its path."""
    mgr = ModelManager()
    p = model_dir / "v1.json"
    mgr.save(p, version="v1-test", description="fixture model")
    return p


def _make_raw_model(
    path: Path,
    *,
    weights: Dict[str, Dict[str, float]] | None = None,
    meta: Dict[str, Any] | None = None,
    ocr: Dict[str, Any] | None = None,
) -> Path:
    """Write a minimal model JSON file manually."""
    payload: Dict[str, Any] = {
        "meta": meta or {"version": "raw", "format_version": _FORMAT_VERSION},
        "classifier_weights": weights or {"prescription": {"medication": 5.0}},
        "ocr_config": ocr or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════

class TestSave:
    def test_creates_file(self, model_dir: Path):
        mgr = ModelManager()
        p = model_dir / "out.json"
        mgr.save(p)
        assert p.exists()

    def test_creates_parent_dirs(self, model_dir: Path):
        mgr = ModelManager()
        p = model_dir / "deep" / "nested" / "model.json"
        mgr.save(p)
        assert p.exists()

    def test_returns_meta(self, model_dir: Path):
        mgr = ModelManager()
        meta = mgr.save(model_dir / "m.json", version="v42", description="test")
        assert isinstance(meta, ModelMeta)
        assert meta.version == "v42"
        assert meta.description == "test"

    def test_auto_version_when_empty(self, model_dir: Path):
        mgr = ModelManager()
        meta = mgr.save(model_dir / "m.json")
        assert meta.version.startswith("v2")  # v20260414_...

    def test_checksum_populated(self, model_dir: Path):
        mgr = ModelManager()
        meta = mgr.save(model_dir / "m.json")
        assert len(meta.checksum) == 16

    def test_json_is_valid(self, saved_model: Path):
        data = json.loads(saved_model.read_text(encoding="utf-8"))
        assert "meta" in data
        assert "classifier_weights" in data
        assert "ocr_config" in data

    def test_captures_classifier_weights(self, saved_model: Path):
        data = json.loads(saved_model.read_text(encoding="utf-8"))
        weights = data["classifier_weights"]
        # Should contain the built-in document types
        assert "prescription" in weights
        assert isinstance(weights["prescription"], dict)
        assert len(weights["prescription"]) > 0

    def test_captures_ocr_config(self, saved_model: Path):
        data = json.loads(saved_model.read_text(encoding="utf-8"))
        ocr = data["ocr_config"]
        assert "ocr_dpi" in ocr
        assert "ocr_lang" in ocr

    def test_training_metadata_saved(self, model_dir: Path):
        mgr = ModelManager()
        meta = mgr.save(
            model_dir / "m.json",
            training_examples_count=150,
            eval_scores={"prescription": 0.95, "result": 0.88},
        )
        assert meta.training_examples_count == 150
        assert meta.eval_scores["prescription"] == 0.95

    def test_registered_types_in_meta(self, saved_model: Path):
        data = json.loads(saved_model.read_text(encoding="utf-8"))
        types = data["meta"]["registered_types"]
        assert "prescription" in types
        assert "result" in types
        assert "clinical_history" in types


# ═══════════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════════

class TestLoad:
    def test_round_trip(self, saved_model: Path):
        mgr = ModelManager.load(saved_model)
        assert mgr.meta is not None
        assert mgr.meta.version == "v1-test"
        assert mgr.meta.description == "fixture model"

    def test_weights_loaded(self, saved_model: Path):
        mgr = ModelManager.load(saved_model)
        assert "prescription" in mgr.classifier_weights
        assert len(mgr.classifier_weights["prescription"]) > 0

    def test_ocr_config_loaded(self, saved_model: Path):
        mgr = ModelManager.load(saved_model)
        assert "ocr_dpi" in mgr.ocr_config

    def test_registered_types_loaded(self, saved_model: Path):
        mgr = ModelManager.load(saved_model)
        assert "prescription" in mgr.registered_types

    def test_file_not_found(self, tmp_path: Path):
        from src.pipeline.exceptions import ModelLoadError

        with pytest.raises(ModelLoadError, match="not found"):
            ModelManager.load(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path: Path):
        from src.pipeline.exceptions import ModelLoadError

        bad = tmp_path / "bad.json"
        bad.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ModelLoadError, match="parse"):
            ModelManager.load(bad)

    def test_missing_weights_key(self, tmp_path: Path):
        from src.pipeline.exceptions import ModelLoadError

        p = tmp_path / "noweights.json"
        p.write_text(json.dumps({"meta": {}}), encoding="utf-8")
        with pytest.raises(ModelLoadError, match="classifier_weights"):
            ModelManager.load(p)

    def test_load_raw_minimal(self, tmp_path: Path):
        """Load a hand-crafted minimal model file."""
        p = _make_raw_model(
            tmp_path / "minimal.json",
            weights={"prescription": {"rx": 10.0}},
        )
        mgr = ModelManager.load(p)
        assert mgr.classifier_weights == {"prescription": {"rx": 10.0}}

    def test_format_version_preserved(self, saved_model: Path):
        mgr = ModelManager.load(saved_model)
        assert mgr.meta is not None
        assert mgr.meta.format_version == _FORMAT_VERSION


# ═══════════════════════════════════════════════════════════════════
# Apply
# ═══════════════════════════════════════════════════════════════════

class TestApply:
    def test_apply_patches_classifier(self, tmp_path: Path):
        """After apply(), DOCUMENT_TYPES should have the new weights."""
        from src.pipeline.extractors.document_classifier import DOCUMENT_TYPES

        original_weight = DOCUMENT_TYPES["prescription"].get("_test_sentinel_kw")
        assert original_weight is None  # shouldn't exist yet

        p = _make_raw_model(
            tmp_path / "patch.json",
            weights={"prescription": {"_test_sentinel_kw": 99.0}},
        )
        mgr = ModelManager.load(p)
        mgr.apply()

        assert DOCUMENT_TYPES["prescription"]["_test_sentinel_kw"] == 99.0

        # Cleanup
        DOCUMENT_TYPES["prescription"].pop("_test_sentinel_kw", None)

    def test_apply_adds_new_type(self, tmp_path: Path):
        from src.pipeline.extractors.document_classifier import DOCUMENT_TYPES

        p = _make_raw_model(
            tmp_path / "new_type.json",
            weights={"radiology_report": {"x-ray": 5.0, "mri": 4.0}},
        )
        mgr = ModelManager.load(p)
        mgr.apply()

        assert "radiology_report" in DOCUMENT_TYPES
        assert DOCUMENT_TYPES["radiology_report"]["x-ray"] == 5.0

        # Cleanup
        DOCUMENT_TYPES.pop("radiology_report", None)

    def test_apply_empty_weights_is_noop(self, tmp_path: Path):
        p = _make_raw_model(tmp_path / "empty.json", weights={})
        mgr = ModelManager.load(p)
        # Should not raise
        mgr.apply()


# ═══════════════════════════════════════════════════════════════════
# load_engine
# ═══════════════════════════════════════════════════════════════════

class TestLoadEngine:
    def test_returns_inference_engine(self, saved_model: Path):
        from src.pipeline.inference import InferenceEngine

        engine = ModelManager.load_engine(saved_model)
        assert isinstance(engine, InferenceEngine)

    def test_engine_has_registered_types(self, saved_model: Path):
        engine = ModelManager.load_engine(saved_model)
        assert "prescription" in engine.registered_types
        assert "result" in engine.registered_types
        assert "clinical_history" in engine.registered_types

    def test_engine_can_classify(self, saved_model: Path):
        engine = ModelManager.load_engine(saved_model)
        text = "Patient Name: John\nMedication: Amoxicillin 500mg\nDosage: 1 tablet"
        result = engine.classify(text)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════
# Inspect
# ═══════════════════════════════════════════════════════════════════

class TestInspect:
    def test_inspect_returns_meta(self, saved_model: Path):
        meta = ModelManager.inspect(saved_model)
        assert isinstance(meta, ModelMeta)
        assert meta.version == "v1-test"

    def test_inspect_invalid_file(self, tmp_path: Path):
        from src.pipeline.exceptions import ModelLoadError

        with pytest.raises(ModelLoadError):
            ModelManager.inspect(tmp_path / "nope.json")


# ═══════════════════════════════════════════════════════════════════
# ModelMeta
# ═══════════════════════════════════════════════════════════════════

class TestModelMeta:
    def test_as_dict(self):
        meta = ModelMeta(
            version="v1",
            format_version="1.0",
            created_utc="2026-04-14T00:00:00+00:00",
            description="test",
            registered_types=["prescription"],
            training_examples_count=10,
            eval_scores={"prescription": 0.9},
            checksum="abc123",
        )
        d = meta.as_dict()
        assert d["version"] == "v1"
        assert d["checksum"] == "abc123"
        assert d["registered_types"] == ["prescription"]

    def test_defaults(self):
        meta = ModelMeta()
        assert meta.version == ""
        assert meta.registered_types == []
        assert meta.eval_scores == {}


# ═══════════════════════════════════════════════════════════════════
# Integration: create_default_engine with model_path
# ═══════════════════════════════════════════════════════════════════

class TestCreateDefaultEngineWithModel:
    def test_explicit_model_path(self, saved_model: Path):
        from src.pipeline.inference import create_default_engine

        engine = create_default_engine(model_path=str(saved_model))
        assert "prescription" in engine.registered_types

    def test_no_model_path_still_works(self):
        from src.pipeline.inference import create_default_engine

        engine = create_default_engine()
        assert "prescription" in engine.registered_types
