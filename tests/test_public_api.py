"""Tests for public API surface – __init__.py exports (NEW-7).

Verifies that every package exposes the documented symbols via its
``__init__.py`` and that ``__all__`` is consistent with actual exports.
"""

from __future__ import annotations

import importlib

import pytest


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def _check_exports(module_path: str, expected_names: list[str]) -> None:
    """Assert that *module_path* exports every name in *expected_names*."""
    mod = importlib.import_module(module_path)
    for name in expected_names:
        assert hasattr(mod, name), f"{module_path} missing export: {name}"
    # __all__ should list at least these names
    if hasattr(mod, "__all__"):
        all_set = set(mod.__all__)
        for name in expected_names:
            assert name in all_set, (
                f"{name!r} exported by {module_path} but missing from __all__"
            )


# ═══════════════════════════════════════════════════════════════════
# src
# ═══════════════════════════════════════════════════════════════════

class TestSrcPackage:
    def test_exports(self):
        _check_exports("src", [
            "DocIQSettings",
            "get_settings",
            "settings",
            "setup_logging",
        ])

    def test_settings_is_instance(self):
        from src import settings, DocIQSettings
        assert isinstance(settings, DocIQSettings)


# ═══════════════════════════════════════════════════════════════════
# src.pipeline
# ═══════════════════════════════════════════════════════════════════

class TestPipelinePackage:
    def test_exports(self):
        _check_exports("src.pipeline", [
            "DocIQEngine",
            "EngineResult",
            "InferenceEngine",
            "InferenceResult",
            "ModelBundle",
            "ModelRegistry",
            "create_default_engine",
            "ModelManager",
            "ModelMeta",
            "OutputCollector",
        ])

    def test_process_folder_submodule_accessible(self):
        """process_folder is available as a submodule (not re-exported to
        avoid shadowing the module name)."""
        from src.pipeline.process_folder import process_folder, DocumentResult
        assert callable(process_folder)
        assert DocumentResult is not None

    def test_create_default_engine_callable(self):
        from src.pipeline import create_default_engine
        engine = create_default_engine()
        assert hasattr(engine, "process_document")

    def test_output_collector_usable(self):
        from src.pipeline import OutputCollector
        c = OutputCollector()
        c.add({"status": "ok"})
        assert c.count == 1


# ═══════════════════════════════════════════════════════════════════
# src.pipeline.extractors
# ═══════════════════════════════════════════════════════════════════

class TestExtractorsPackage:
    def test_exports(self):
        _check_exports("src.pipeline.extractors", [
            "PrescriptionExtractor",
            "LabResultExtractor",
            "ClinicalHistoryExtractor",
        ])

    def test_extractors_have_extract_method(self):
        from src.pipeline.extractors import (
            PrescriptionExtractor,
            LabResultExtractor,
            ClinicalHistoryExtractor,
        )
        for cls in (PrescriptionExtractor, LabResultExtractor, ClinicalHistoryExtractor):
            assert hasattr(cls(), "extract"), f"{cls.__name__} missing extract()"


# ═══════════════════════════════════════════════════════════════════
# src.pipeline.validation
# ═══════════════════════════════════════════════════════════════════

class TestValidationPackage:
    def test_exports(self):
        _check_exports("src.pipeline.validation", [
            "ResultSchema",
            "Prescription",
            "ClinicalHistorySchema",
            "validate_result_schema",
            "validate_prescription",
            "validate_clinical_history",
            "validate_output",
            "validate_batch",
            "OutputValidationResult",
            "SCHEMA_REGISTRY",
        ])

    def test_schema_registry_has_types(self):
        from src.pipeline.validation import SCHEMA_REGISTRY
        assert "result" in SCHEMA_REGISTRY
        assert "prescription" in SCHEMA_REGISTRY
        assert "clinical_history" in SCHEMA_REGISTRY


# ═══════════════════════════════════════════════════════════════════
# src.pipeline.utils
# ═══════════════════════════════════════════════════════════════════

class TestUtilsPackage:
    def test_exports(self):
        _check_exports("src.pipeline.utils", [
            "detect_language",
            "is_english",
            "is_spanish",
            "clean_text",
            "lowercase",
            "normalize_whitespace",
            "remove_numbers",
            "strip_non_ascii",
            "truncate",
            "normalize_dates",
        ])

    def test_text_functions_callable(self):
        from src.pipeline.utils import clean_text, normalize_whitespace, truncate
        assert clean_text("  hello  ") == "hello"
        assert normalize_whitespace("a  b") == "a b"
        assert truncate("short", max_length=100) == "short"


# ═══════════════════════════════════════════════════════════════════
# src.api
# ═══════════════════════════════════════════════════════════════════

class TestApiPackage:
    def test_exports(self):
        _check_exports("src.api", [
            "HealthResponse",
            "ProcessingResponse",
            "ReadinessResponse",
        ])

    def test_models_are_pydantic(self):
        from pydantic import BaseModel
        from src.api import HealthResponse, ReadinessResponse
        assert issubclass(HealthResponse, BaseModel)
        assert issubclass(ReadinessResponse, BaseModel)


# ═══════════════════════════════════════════════════════════════════
# __all__ consistency: every name in __all__ must actually exist
# ═══════════════════════════════════════════════════════════════════

_PACKAGES = [
    "src",
    "src.pipeline",
    "src.pipeline.extractors",
    "src.pipeline.validation",
    "src.pipeline.utils",
    "src.api",
]

@pytest.mark.parametrize("pkg", _PACKAGES)
def test_all_entries_exist(pkg: str):
    """Every name listed in __all__ must be an actual attribute."""
    mod = importlib.import_module(pkg)
    if not hasattr(mod, "__all__"):
        pytest.skip(f"{pkg} has no __all__")
    for name in mod.__all__:
        assert hasattr(mod, name), f"{pkg}.__all__ lists {name!r} but it does not exist"
