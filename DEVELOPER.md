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
    +-- LLM Enhancement (opt-in, Ollama via OllamaClient)
    +-- Merge Layer (rule-based wins, LLM fills gaps)
    +-- RelationMapper (entity wiring for ML NER)
    +-- Validators (Pydantic schema enforcement)
    |
OutputFormatter (JSON / CSV / FHIR export)
```

The `DocIQEngine` in `src/pipeline/core_engine.py` is the top-level orchestrator. It owns the extraction step (PDF/image I/O) and delegates classification, NER, and validation to the `InferenceEngine`. Export is handled by `output_formatter.py`.

The `InferenceEngine` in `src/pipeline/inference.py` is the inference orchestrator. It manages a `ModelRegistry` of `ModelBundle` objects, each containing a classifier and NER model for a document type. The engine preprocesses text, runs both models, optionally applies relation mapping, merges outputs, optionally enhances via the LLM layer, and validates against the registered Pydantic schema.

## Pipeline Internals

### Document flow

1. **File intake** -- `DocIQEngine.process_file()` checks the file extension, then calls `_extract_pdf()` or `_extract_image()` from `process_folder.py`.

2. **PDF extraction** -- `_extract_pdf()` samples the first pages for language detection, checks if the PDF is text-based (`pdf_type_detector.py`), and routes to `extract_text_directly()` (PyMuPDF, thread-pooled) or `extract_text_from_pdf_ocr()` (pdf2image + PaddleOCR, sequential since PaddleOCR is not thread-safe).

3. **Image extraction** -- `_extract_image()` calls `extract_text_from_image()` in `ocr.py`, which delegates to the PaddleOCR singleton in `ocr_paddle.py`. PaddleOCR handles its own image preprocessing (binarisation, deskew, orientation correction) internally.

4. **Classification** -- The `DocumentClassifier` scores the text against keyword sets for each document type. It returns the highest-scoring type above a configurable threshold, or `None` for unclassifiable text.

5. **NER extraction** -- The `InferenceEngine` retrieves the `ModelBundle` for the classified type and runs the NER model on the raw text (preserving casing for proper nouns and dates). The classifier model runs on preprocessed (lowercased) text. Rule-based extractors use accent-flexible regex patterns (see below) so that both accented text from PyMuPDF and unaccented text from PaddleOCR are matched by the same alias lists. Each extractor also includes the full page text in a `raw_text` field to preserve information that falls outside the structured schema.

6. **Relation mapping** -- If the NER output contains a flat `"entities"` list (from an ML model rather than a rule-based extractor), the engine applies `RelationMapper` with domain-specific configs to wire entities into structured relations.

7. **Validation** -- The merged extraction dict is validated against the Pydantic schema for that document type. The validator normalises dates and strips whitespace before validation.

8. **Export** -- `export_results()` dispatches to `export_json()`, `export_csv()`, or `export_fhir()`. JSON and FHIR exports use `ThreadPoolExecutor` for parallel writes.

### Model dispatch

The `InferenceEngine._apply_model()` method supports multiple model interfaces. It checks in order: `model.predict()`, `model.extract()`, `model.extract_entities()`, or `model()` (callable). This allows both rule-based extractors (which use `extract()`) and ML models (which typically use `predict()`) to plug in without adapter code.

### OCR engine

DocIQ uses PaddleOCR 3.4+ (PP-OCRv4 models) for all OCR tasks. The engine is managed as a thread-safe singleton in `src/pipeline/ocr_paddle.py` with double-checked locking. Key design decisions:

- **`enable_mkldnn=False`** -- Critical fix that bypasses the broken oneDNN/PIR code path in PaddlePaddle 3.x. This flows through PaddleX's config chain to force `run_mode="paddle"`.
- **`lang="en"`** -- The English/Latin model handles all Latin-script languages including Spanish. PP-OCRv4 has no dedicated Spanish model.
- **Sequential page processing** -- PaddleOCR is not thread-safe, so pages are processed one at a time (unlike PyMuPDF direct extraction which uses a thread pool).
- **Accent stripping** -- PaddleOCR with the Latin model strips accents from Spanish text (e.g. "años" → "anos", "Cédula" → "Cedula"). The accent-flexible regex system compensates for this.

### Accent-flexible regex

The `_accent_flex()` function in `src/pipeline/extractors/base.py` converts literal field alias strings into regex patterns where each Latin vowel and `ñ` is replaced with a character class:

```python
_accent_flex("Identificacion")
# → '[Iíìî]d[eéèê][nñ]t[iíìî]f[iíìî]c[aáàâã]c[iíìî][oóòôõ][nñ]'
```

This is applied automatically inside `extract_field()`, `extract_date()`, and `extract_block()`, so all alias lists work transparently with both text sources. Developers only need to add one spelling per alias (unaccented preferred).

### Raw text preservation

All extractors include a `raw_text` field in their output containing the full page text. This ensures that information not captured by the structured regex patterns is still available for the LLM enhancement layer. The corresponding Pydantic schemas accept `raw_text` as an optional field.

### LLM enhancement layer

When `llm_enabled` is `true` in the config, the `InferenceEngine.process_document()` method runs an additional enhancement step after rule-based extraction. The flow is:

1. **Rule-based extraction runs first** and produces a complete dict with all schema fields (some may be `None` or empty).
2. **OllamaClient** (`src/pipeline/llm_client.py`) sends the raw text to a local Ollama server and asks it to extract structured fields.
3. **`_merge_llm_fields()`** walks both dicts field by field. Rule-based values always win when non-empty. The LLM fills fields that are `None`, empty strings, or empty lists. Fields in `_LLM_SKIP_FIELDS` (`raw_text`, `document_type`) are never overwritten. Fields returned by the LLM that do not exist in the rule-based dict are silently ignored to avoid Pydantic `extra="forbid"` violations.
4. **Special items merge** -- for prescription `items` lists, the LLM list replaces the rule-based list only if it contains more items, or if the items have richer sub-fields (more non-null values per item).
5. **Re-validation** -- if the LLM contributed any fields, the merged dict is re-validated against the Pydantic schema.

The entire block is wrapped in a `try/except` so LLM failure never blocks the pipeline. If the Ollama server is unreachable, the pipeline returns rule-based results silently. The `InferenceResult` dataclass exposes `llm_output` (raw LLM dict) and `llm_fields_used` (list of field names where LLM values were used) for observability.

#### OllamaClient

`src/pipeline/llm_client.py` wraps the Ollama `/api/generate` endpoint with retry logic and exponential backoff. It provides three methods:

- `is_available()` -- quick health check against the server root.
- `classify(text)` -- returns a `LLMClassifyResult(document_type, confidence, language)`.
- `extract(document_type, text)` -- returns a structured dict matching the Pydantic schema for that type.

Text is truncated to 3000 characters to stay within the model's context window (4096 tokens). JSON responses are parsed with automatic markdown fence stripping.

#### Custom model

The `dociq-medical` Ollama model (defined in `models/Modelfile`) uses `qwen2.5:3b` as a base with a system prompt containing the three Pydantic schemas and instructions for two modes (classify and extract). Temperature is set to 0.1 for deterministic output. See `models/benchmark.py` for latency benchmarks.

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

### 7. Update the LLM model (if LLM is enabled)

If you use the LLM enhancement layer, update the `models/Modelfile` system prompt to include the new Pydantic schema. The LLM must use the exact field names from the schema — any names it returns that are not in the rule-based extractor's output dict will be silently ignored by the merge layer.

### 8. Write tests

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
+-- OCRError (ImageLoadError, PaddleOCRError)
+-- ClassificationError
+-- NERExtractionError
+-- ModelError (ModelLoadError, ModelExecutionError)
+-- LLMError (LLMConnectionError, LLMTimeoutError, LLMResponseError)
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

The `/ready` endpoint checks dependencies (Poppler, PaddleOCR, inference engine, config, disk space) and reports timing for each. It returns 503 if any check fails, making it compatible with Kubernetes readiness probes.

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
- `test_llm_integration.py` -- LLM client, merge logic, pipeline integration with mocked Ollama
- `test_containerization.py`, `test_deployment.py` -- Docker and deploy script validation
- `test_api.py` -- FastAPI endpoint tests

### Test patterns

The test suite uses several patterns:

**Mocked extraction fixture** -- `_mock_extraction` patches file I/O (PDF extraction, image OCR, language detection) so tests can feed synthetic text directly without real files.

**Module-scoped engine** -- `inference_engine` is created once per module with `scope="module"` since `create_default_engine()` is deterministic and stateless.

**MagicMock with spec=[]** -- When testing the InferenceEngine with mock models, use `MagicMock(spec=[])` to prevent auto-generated `predict`/`extract` attributes from interfering with the model dispatch logic.

**AST-based validation** -- Architecture tests parse source files as ASTs to verify import structure without executing the code.

### LLM integration tests

`test_llm_integration.py` (46 tests) covers the full LLM layer without requiring a live Ollama server:

- **Client tests** -- `@patch("src.pipeline.llm_client.requests.post")` mocks the HTTP layer. Tests cover classify, extract, JSON parsing, retries, errors, and text truncation.
- **Merge logic tests** -- call `InferenceEngine._merge_llm_fields()` directly with synthetic dicts. Tests cover gap-filling, skip fields, items richness comparison, and schema-safety (unknown fields ignored).
- **Pipeline integration tests** -- use `@patch("src.pipeline.llm_client.OllamaClient")` to mock the client class and `patch.object(settings, "llm_enabled", True)` to enable LLM without modifying global config. Tests verify that the engine fills gaps, falls back when unavailable, and falls back on exceptions.

Key mocking pattern: because both `OllamaClient` and `settings` are imported locally inside `_enhance_with_llm()` and `process_document()`, the patch targets must be `src.pipeline.llm_client.OllamaClient` (not `src.pipeline.inference.OllamaClient`) and `patch.object` on the real settings singleton (not a module-level attribute).

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
