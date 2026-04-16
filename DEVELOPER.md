# Developer Guide

This guide covers the internal architecture, extension points, and development workflows for contributing to DocIQ.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Internals](#pipeline-internals)
- [Adding a New Document Type](#adding-a-new-document-type)
- [Validation Schemas](#validation-schemas)
- [FHIR Mapping](#fhir-mapping)
- [Relation Mapping](#relation-mapping)
- [Configuration System](#configuration-system)
- [Exception Hierarchy](#exception-hierarchy)
- [Metrics and Observability](#metrics-and-observability)
- [Testing Guide](#testing-guide)
- [Code Conventions](#code-conventions)

## Architecture Overview

DocIQ follows a layered pipeline architecture. Each layer has a single responsibility and communicates through plain dictionaries or Pydantic models:

```
CLI / API
    |
DocIQEngine (orchestrator)
    |
InferenceEngine (classify + extract + validate)
    |
    +-- DocumentClassifier (keyword scoring)
    +-- Extractors (rule-based NER per document type)
    +-- RelationMapper (entity wiring for ML NER)
    +-- Validators (Pydantic schema enforcement)
    |
OutputFormatter (JSON / CSV / FHIR export)
```

The `DocIQEngine` in `src/pipeline/core_engine.py` is the top-level orchestrator. It owns the extraction step (PDF/image I/O) and delegates classification, NER, and validation to the `InferenceEngine`. Export is handled by `output_formatter.py`.

The `InferenceEngine` in `src/pipeline/inference.py` is the inference orchestrator. It manages a `ModelRegistry` of `ModelBundle` objects, each containing a classifier and NER model for a document type. The engine preprocesses text, runs both models, optionally applies relation mapping, merges outputs, and validates against the registered Pydantic schema.

## Pipeline Internals

### Document flow

1. **File intake** -- `DocIQEngine.process_file()` checks the file extension, then calls `_extract_pdf()` or `_extract_image()` from `process_folder.py`.

2. **PDF extraction** -- `_extract_pdf()` samples the first pages for language detection, checks if the PDF is text-based (`pdf_type_detector.py`), and routes to `extract_text_directly()` (PyMuPDF) or `extract_text_from_pdf_ocr()` (pdf2image + Tesseract).

3. **Image extraction** -- `_extract_image()` calls `extract_text_from_image()` in `ocr.py`, which loads the image, converts colour modes, and runs Tesseract.

4. **Classification** -- The `DocumentClassifier` scores the text against keyword sets for each document type. It returns the highest-scoring type above a configurable threshold, or `None` for unclassifiable text.

5. **NER extraction** -- The `InferenceEngine` retrieves the `ModelBundle` for the classified type and runs the NER model on the raw text (preserving casing for proper nouns and dates). The classifier model runs on preprocessed (lowercased) text.

6. **Relation mapping** -- If the NER output contains a flat `"entities"` list (from an ML model rather than a rule-based extractor), the engine applies `RelationMapper` with domain-specific configs to wire entities into structured relations.

7. **Validation** -- The merged extraction dict is validated against the Pydantic schema for that document type. The validator normalises dates and strips whitespace before validation.

8. **Export** -- `export_results()` dispatches to `export_json()`, `export_csv()`, or `export_fhir()`. JSON and FHIR exports use `ThreadPoolExecutor` for parallel writes.

### Model dispatch

The `InferenceEngine._apply_model()` method supports multiple model interfaces. It checks in order: `model.predict()`, `model.extract()`, `model.extract_entities()`, or `model()` (callable). This allows both rule-based extractors (which use `extract()`) and ML models (which typically use `predict()`) to plug in without adapter code.

## Adding a New Document Type

To add support for a new document type (e.g. `imaging_report`):

### 1. Create the Pydantic schema

```python
# src/pipeline/validation/imaging_report_schema.py
from pydantic import BaseModel, Field
from typing import Optional

class ImagingReportSchema(BaseModel):
    patient_name: str = Field(..., min_length=1)
    patient_id: Optional[str] = None
    modality: str = Field(...)       # "MRI", "CT", "X-Ray", etc.
    body_part: Optional[str] = None
    findings: str = Field(...)
    impression: Optional[str] = None
    radiologist: Optional[str] = None
    institution: Optional[str] = None
```

### 2. Register the schema in the validator

```python
# src/pipeline/validation/validator.py
from src.pipeline.validation.imaging_report_schema import ImagingReportSchema

SCHEMA_REGISTRY["imaging_report"] = ImagingReportSchema

def validate_imaging_report(data: dict) -> ImagingReportSchema:
    return ImagingReportSchema(**data)
```

Update `validation/__init__.py` to re-export the new schema and validator.

### 3. Create the extractor

```python
# src/pipeline/extractors/imaging_report_extractor.py
from src.pipeline.extractors.base import extract_field, extract_block
from src.pipeline.extractors.field_aliases import resolve_institution

class ImagingReportExtractor:
    def extract(self, text: str) -> dict:
        return {
            "patient_name": extract_field(text, "Patient Name"),
            "modality": extract_field(text, "Modality"),
            "body_part": extract_field(text, "Body Part"),
            "findings": extract_block(text, "Findings") or "",
            "impression": extract_field(text, "Impression"),
            "radiologist": extract_field(text, "Radiologist"),
            "institution": resolve_institution(text),
        }
```

### 4. Add classification keywords

In `src/pipeline/extractors/document_classifier.py`, add a keyword entry for the new type in the `_KEYWORDS` dict so the classifier can detect it.

### 5. Register in the default engine

```python
# src/pipeline/inference.py, inside create_default_engine()
from src.pipeline.extractors.imaging_report_extractor import ImagingReportExtractor

registry["imaging_report"] = ModelBundle(
    classifier=classifier,
    ner=ImagingReportExtractor(),
)
```

Add the validator to `InferenceEngine.DEFAULT_VALIDATORS`.

### 6. Add FHIR mapping (optional)

In `src/pipeline/fhir_mapper.py`, add an `isinstance` check in `map_to_fhir_loose()` and a `imaging_report_to_fhir()` function.

### 7. Write tests

Create `tests/test_imaging_report.py` with extraction, validation, and integration tests following the patterns in existing test files.

## Validation Schemas

All schemas live in `src/pipeline/validation/` and use Pydantic v2. Key conventions:

- `model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")` -- whitespace is stripped and unexpected fields raise errors.
- Required fields use `Field(..., min_length=1)` to reject empty strings.
- Optional fields default to `None`.
- The `SCHEMA_REGISTRY` in `validator.py` maps document type strings to schema classes. This registry is used by the output formatter and API for FHIR mapping.
- The `validate_*` functions in `validator.py` apply date normalisation (via `@normalize_dates` decorator) before constructing the model.

## FHIR Mapping

`src/pipeline/fhir_mapper.py` converts validated Pydantic models to loose FHIR R4 resources:

| Schema | FHIR Resource |
|---|---|
| `ResultSchema` | `DiagnosticReport` |
| `Prescription` | `MedicationRequest` |
| `ClinicalHistorySchema` | `Encounter` |

The `prune_none()` helper strips all `None` values recursively so exported JSON contains only populated fields.

`build_fhir_bundle()` wraps a list of resources into a FHIR `Bundle` of type `collection`, each entry with a `fullUrl` of `urn:uuid:<generated>`.

## Relation Mapping

The relation mapping system (`src/pipeline/relation_mapper.py`) handles flat entity lists produced by ML-based NER models. It groups entities into structured relations using proximity-based anchor-dependent configs.

Domain configs are defined in `src/pipeline/relation_configs.py`:

- `PRESCRIPTION_RELATIONS` -- 5 anchor types (MEDICATION, RADIOLOGY, LAB_TEST, SPECIALIST, THERAPY)
- `RESULT_RELATIONS` -- 2 anchors (TEST_NAME, EXAM_TYPE)
- `CLINICAL_HISTORY_RELATIONS` -- 3 anchors (DIAGNOSIS, MEDICATION, COMPLAINT)

The `RelationMapper` takes a config and a list of entities, groups them by proximity window, and returns a `RelationMappingResult` with structured relations and orphaned entities. The `InferenceEngine` auto-detects when to apply mapping based on the NER output format.

## Configuration System

The `DocIQSettings` class in `src/config.py` uses a multi-source resolution strategy:

```python
from src.config import get_settings

# Load from defaults + environment
cfg = get_settings()

# Load from YAML file with overrides
cfg = get_settings(config_path="dociq.yaml", log_level="DEBUG")
```

Settings are resolved in this priority order: constructor kwargs > environment variables (`DOCIQ_*`) > `.env` file > YAML config > defaults. The `get_settings()` factory handles merging and returns a frozen `DocIQSettings` instance.

All settings have validators (value ranges, allowed values) defined via Pydantic's `Field` constraints.

## Exception Hierarchy

All pipeline exceptions inherit from `DocIQError` for blanket catching. The hierarchy in `src/pipeline/exceptions.py`:

```
DocIQError
+-- ConfigurationError (ConfigFileNotFoundError, ConfigParseError)
+-- DocumentExtractionError (PDFOpenError, PDFExtractionError, PageTimeoutError)
+-- OCRError (ImageLoadError, TesseractError)
+-- ClassificationError
+-- NERExtractionError
+-- ModelError (ModelLoadError, ModelExecutionError)
+-- SchemaValidationError
+-- ExportError (FHIRMappingError)
+-- UnsupportedFileError
```

The API layer maps these to HTTP status codes (400, 422, 500, 503, 504) with structured error responses.

## Metrics and Observability

### Timing metrics

The `MetricsCollector` in `src/pipeline/metrics.py` provides thread-safe aggregation:

```python
from src.pipeline.metrics import get_collector, Timer

with Timer("ocr_extraction") as t:
    text = extract_text_from_image(path)

collector = get_collector()
collector.record("ocr_extraction", t.elapsed_ms)

summary = collector.summary()
# {"ocr_extraction": {"count": 5, "total_ms": 1234, "mean_ms": 246.8, "p95_ms": ...}}
```

The `@timed` decorator records function execution time automatically:

```python
from src.pipeline.metrics import timed

@timed("my_operation")
def do_work():
    ...
```

### Logging

DocIQ supports two log formats configured via `DOCIQ_LOG_FORMAT`:

- `text` -- human-readable format for development
- `json` -- structured JSON lines for log aggregators (includes timestamp, level, logger, module, function, line number, and exception traceback)

### Health checks

The `/ready` endpoint checks five dependencies and reports timing for each. It returns 503 if any check fails, making it compatible with Kubernetes readiness probes.

## Testing Guide

### Running tests

```bash
# Full suite
pytest

# Verbose with short tracebacks
pytest -v --tb=short

# Single file
pytest tests/test_integration.py

# By keyword
pytest -k "test_prescription"

# Stop on first failure
pytest -x
```

### Test organisation

Tests are organised by pipeline stage and concern:

- `test_e2e_pipeline.py` -- happy-path end-to-end flows through the real inference engine
- `test_integration.py` -- cross-cutting integration (error propagation, API endpoints, metrics, batch failures)
- `test_architecture.py` -- structural validation (import coupling, file naming, exports)
- `test_core_engine.py` -- `DocIQEngine` and `EngineResult`
- `test_inference_engine.py` -- `InferenceEngine` classification and extraction
- `test_extractors.py` -- individual rule-based extractors
- `test_ner_comprehensive.py` -- comprehensive NER across all extractors
- `test_fhir_mapper.py`, `test_fhir_integration.py`, `test_fhir_mapping_comprehensive.py` -- FHIR mapping
- `test_containerization.py`, `test_deployment.py` -- Docker and deploy script validation
- `test_api.py` -- FastAPI endpoint tests

### Test patterns

The test suite uses several patterns:

**Mocked extraction fixture** -- `_mock_extraction` patches file I/O (PDF extraction, image OCR, language detection) so tests can feed synthetic text directly without real files.

**Module-scoped engine** -- `inference_engine` is created once per module with `scope="module"` since `create_default_engine()` is deterministic and stateless.

**MagicMock with spec=[]** -- When testing the InferenceEngine with mock models, use `MagicMock(spec=[])` to prevent auto-generated `predict`/`extract` attributes from interfering with the model dispatch logic.

**AST-based validation** -- Architecture tests parse source files as ASTs to verify import structure without executing the code.

### Writing new tests

When adding tests for a new feature, prefer integration-level tests that exercise the real inference engine over unit tests with extensive mocking. Only mock file I/O and external services. Use the synthetic text constants from `test_e2e_pipeline.py` or `test_integration.py` as templates for realistic document text.

## Code Conventions

- **Imports** -- Consumer modules import schemas from `src.pipeline.validation`, not from individual schema submodules. Only the validation layer itself uses direct submodule imports.
- **Field aliases** -- Shared field-name fallback chains (e.g. `Clinic` / `Institution`) go in `extractors/field_aliases.py`, not inline in each extractor.
- **File naming** -- All Python files use `snake_case`. Schema files follow the pattern `<type>_schema.py`.
- **Module-level imports** -- Avoid late imports inside function bodies unless there is a genuine circular dependency risk. Import at module level for clarity.
- **Docstrings** -- Public functions and classes use NumPy-style docstrings with Parameters, Returns, and Raises sections.
- **Type hints** -- All public APIs are fully typed. Use `from __future__ import annotations` for forward references.
- **Error handling** -- Pipeline errors use the custom exception hierarchy. Per-file errors in batch processing are caught and recorded in the result dict without stopping the batch.
