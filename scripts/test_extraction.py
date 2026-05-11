#!/usr/bin/env python
"""Quick extraction test against a real PDF.

Usage (from project root, with .venv activated):

    python scripts/test_extraction.py data/PRUEBAS.pdf

Processes the PDF through the full pipeline (OCR or direct text),
classifies each page, runs extraction, and prints a summary of
which fields were captured vs. null for each page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.pdf_extractor import extract_text_from_pdf
from src.pipeline.extractors.document_classifier import DocumentClassifier
from src.pipeline.extractors.clinical_history_extractor import ClinicalHistoryExtractor
from src.pipeline.extractors.prescription_extractor import PrescriptionExtractor
from src.pipeline.extractors.result_extractor import LabResultExtractor

_classifier = DocumentClassifier()

_EXTRACTORS = {
    "clinical_history": ClinicalHistoryExtractor(),
    "prescription": PrescriptionExtractor(),
    "lab_result": LabResultExtractor(),
    "result": LabResultExtractor(),
    "receipt": None,    # no extractor yet
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_extraction.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"  Extraction test: {pdf_path}")
    print(f"{'='*60}\n")

    # Extract with page results
    full_text, page_results = extract_text_from_pdf(
        pdf_path, return_page_results=True,
    )

    total_fields = 0
    captured_fields = 0

    for pr in page_results:
        page_num = pr["page"]
        text = pr["text"]

        if not text.strip():
            print(f"Page {page_num:2d}: (empty)\n")
            continue

        # Classify
        doc_type = _classifier.classify(text) or "unknown"
        extractor = _EXTRACTORS.get(doc_type)

        print(f"Page {page_num:2d}: [{doc_type}]")

        if extractor is None:
            print(f"  (no extractor for '{doc_type}')\n")
            continue

        result = extractor.extract(text)

        filled = []
        empty = []
        for key, val in result.items():
            total_fields += 1
            if val is not None and val != [] and val != "Unknown" and val != "No findings recorded":
                captured_fields += 1
                # Truncate long values for display
                display = str(val)
                if len(display) > 80:
                    display = display[:77] + "..."
                filled.append(f"    ✓ {key}: {display}")
            else:
                empty.append(f"    ✗ {key}")

        for line in filled:
            print(line)
        for line in empty:
            print(line)
        print()

    # Summary
    pct = (captured_fields / total_fields * 100) if total_fields else 0
    print(f"{'='*60}")
    print(f"  SUMMARY: {captured_fields}/{total_fields} fields captured ({pct:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
