# DocIQ

DocIQ is a document classification and information extraction engine designed for medical prescriptions, lab results, and clinical histories. It combines OCR, lightweight language detection, structured validation, and FHIR mapping to turn semi-structured clinical documents into interoperable JSON artifacts.

## Table of Contents
- [Key Capabilities](#key-capabilities)
- [Repository Layout](#repository-layout)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup](#setup)
  - [Python Environment](#python-environment)
  - [System Dependencies](#system-dependencies)
- [Usage](#usage)
  - [Process a Folder of Documents](#process-a-folder-of-documents)
  - [Working with Structured Outputs](#working-with-structured-outputs)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Key Capabilities
- **Mixed-mode PDF handling** – Determines whether PDFs are text-based or scanned before selecting direct extraction or OCR workflows.【F:src/pipeline/process_folder.py†L54-L82】【F:src/pipeline/pdf_extractor.py†L94-L181】
- **Language triage** – Samples PDF text and runs `langdetect` with deterministic seeding to flag English and Spanish content used for downstream routing.【F:src/pipeline/language.py†L14-L77】
- **Image preprocessing and OCR** – Normalises images with OpenCV before extracting text through Tesseract, ensuring higher-quality OCR on noisy scans.【F:src/pipeline/preprocess.py†L1-L19】【F:src/pipeline/pdf_extractor.py†L21-L72】
- **FHIR-ready outputs** – Validated clinical entities are converted into loose FHIR resources for interoperability and saved as JSON payloads.【F:src/pipeline/fhir_mapper.py†L11-L89】【F:src/pipeline/output_formatter.py†L1-L22】

## Repository Layout

```
.
├── data/                     # Sample or generated input documents
├── output/                   # Destination for processed JSON/CSV artifacts
├── src/
│   ├── main.py               # Example entry point that orchestrates folder processing
│   ├── pipeline/
│   │   ├── process_folder.py # Batch ingestion & routing between PDF and image handlers
│   │   ├── pdf_extractor.py  # Direct text extraction and OCR utilities for PDFs
│   │   ├── ocr.py            # OCR helpers for standalone image files
│   │   ├── language.py       # Lightweight language detection and PDF sampling logic
│   │   ├── preprocess.py     # Image/text normalisation helpers
│   │   ├── utils/            # Shared utilities (text cleanup, language helpers)
│   │   ├── validation/       # Pydantic schemas for prescription/history/result payloads
│   │   └── fhir_*            # Mapping and persistence utilities for FHIR-compatible output
└── tests/                    # Pytest coverage for extraction utilities
```

## Pipeline Architecture

```mermaid
graph TD
    A[Document Intake] --> B[Extension Filter]
    B --> C{PDF?}
    C -- Yes --> D[Sample & Detect PDF Language]
    D --> E{Text-based?}
    E -- Yes --> F[Direct Text Extraction (PyMuPDF)]
    E -- No --> G[OCR Pipeline (pdf2image + Tesseract)]
    F --> H[Language Finalisation]
    G --> H
    C -- No --> I[Image OCR]
    H --> J[Preprocess & Normalise Text]
    J --> K[Validation & Structuring]
    K --> L[FHIR Mapping]
    L --> M[Persist Outputs]
```

- **Document intake** enumerates supported extensions and dispatches PDFs to a worker pool while processing images synchronously.【F:src/pipeline/process_folder.py†L39-L76】
- **PDF language sampling** builds an early language hint by reading the first few pages before deciding on an extraction strategy.【F:src/pipeline/process_folder.py†L57-L74】【F:src/pipeline/language.py†L63-L103】
- **Extraction strategy** chooses direct text reads for digital PDFs or spins up multi-threaded OCR for scanned documents.【F:src/pipeline/process_folder.py†L67-L74】【F:src/pipeline/pdf_extractor.py†L94-L181】
- **Image OCR** routes non-PDF assets to the image OCR module which handles preprocessing and Tesseract execution.【F:src/pipeline/process_folder.py†L78-L93】【F:src/pipeline/ocr.py†L1-L80】
- **Post-processing & validation** prepares cleaned text and validates structured entities before mapping to FHIR resources.【F:src/pipeline/preprocess.py†L5-L19】【F:src/pipeline/output_formatter.py†L7-L22】
- **FHIR mapping & persistence** transforms validated payloads into DiagnosticReport, MedicationRequest, or Encounter resources and saves JSON artifacts.【F:src/pipeline/fhir_mapper.py†L11-L89】【F:src/pipeline/output_formatter.py†L7-L22】

## Setup

### Python Environment
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
   ```
3. Install the Python dependencies:
   ```bash
   pip install langdetect pymupdf pdf2image pytesseract opencv-python-headless numpy pandas pydantic pillow pytest
   ```

### System Dependencies
- **Tesseract OCR** – Required for both PDF OCR and standalone image extraction. Install via your package manager (e.g., `sudo apt-get install tesseract-ocr`) and ensure the binary is on your `PATH`.
- **Poppler** – `pdf2image` relies on Poppler utilities such as `pdftoppm`. Install with `sudo apt-get install poppler-utils` (Linux) or via Homebrew on macOS (`brew install poppler`).

## Usage

### Process a Folder of Documents
Use the `process_folder` helper to scan a directory of PDFs and images and return metadata for each file. The snippet below writes the aggregated results to a CSV file:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
from src.pipeline.process_folder import process_folder

folder = Path("data/generated")  # Update with your input directory
results = process_folder(str(folder))

output_csv = Path("output/ocr_results.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")
print(f"Saved {len(results)} rows to {output_csv}")
PY
```

The resulting CSV contains the extracted text, inferred language, the hint captured during PDF sampling, and the extraction method for traceability.【F:src/pipeline/process_folder.py†L54-L96】

### Working with Structured Outputs
Once OCR and text parsing populate Pydantic models (e.g., `ResultSchema`, `Prescription`), convert them into FHIR-aligned JSON:

```python
from src.pipeline.fhir_mapper import map_to_fhir_loose
from src.pipeline.output_formatter import save_json_output

fhir_payload = map_to_fhir_loose(prescription_document)
save_json_output(prescription_document, "output/prescription.json")
```

`map_to_fhir_loose` dispatches to the appropriate mapper based on the document type and returns a dictionary ready to be serialised or transmitted.【F:src/pipeline/fhir_mapper.py†L11-L89】

## Testing
Run the test suite after installing the optional dependencies:

```bash
pytest
```

Tests cover PDF extraction helpers including direct-text threading and classifier hooks. Some tests require `data/generated/sample_text_based.pdf`; they skip automatically if the fixture is missing.【F:tests/test_pdf_extractor.py†L1-L39】

## Troubleshooting
- **`fitz` import errors** – Ensure `PyMuPDF` installed successfully; reinstall with `pip install --upgrade pymupdf` if necessary.
- **`pdf2image` cannot find Poppler** – Add the Poppler binary directory to your `PATH`. On Windows, download Poppler for Windows and update the `PATH` environment variable to include the `bin` folder.
- **OCR output is noisy** – Confirm that Tesseract is installed with the appropriate language packs and consider adjusting the preprocessing thresholds inside `src/pipeline/preprocess.py`.

