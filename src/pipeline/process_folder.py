"""Batch document ingestion.

Scans a folder for supported documents (PDF and image files), extracts text
via the appropriate method (direct extraction or OCR), detects the language,
classifies the document type, runs NER extraction through the inference
engine, and returns structured results.

Usage
-----
    from src.pipeline.process_folder import process_folder

    results = process_folder("data/generated")
    for r in results:
        print(r["file"], r["document_type"], r["status"])
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.language import detect_language, detect_pdf_language
from src.pipeline.ocr import extract_text_from_image
from src.pipeline.output_collector import OutputCollector
from src.pipeline.pdf_extractor import (
    extract_text_directly,
    extract_text_from_pdf_ocr,
)
from src.pipeline.pdf_type_detector import is_pdf_text_based

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}


# ── Result dataclass ──────────────────────────────────────────────

@dataclass
class DocumentResult:
    """Structured result for a single processed document."""

    file: str
    status: str  # "ok", "extraction_error", "inference_error", "skipped"
    text: str = ""
    language: str = "unknown"
    language_hint: str = "unknown"
    language_sample: str = ""
    method: str = ""  # "direct", "ocr", "image"
    document_type: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    validated: bool = False
    error: Optional[str] = None
    elapsed_ms: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dictionary."""
        return asdict(self)


# ── Internal helpers ──────────────────────────────────────────────

def _finalise_language(full_text: str, fallback: str) -> str:
    detected = detect_language(full_text)
    if detected == "unknown":
        return fallback
    return detected


def _extract_pdf(file_path: str, filename: str) -> Dict[str, Any]:
    """Extract text and metadata from a PDF file."""
    language_result = detect_pdf_language(file_path)
    language_hint = language_result.language
    text_sample = language_result.text_sample

    if is_pdf_text_based(file_path):
        text = extract_text_directly(file_path)
        method = "direct"
    else:
        text = extract_text_from_pdf_ocr(file_path)
        method = "ocr"

    language = _finalise_language(text, language_hint)

    return {
        "text": text.strip(),
        "language": language,
        "language_hint": language_hint,
        "language_sample": text_sample,
        "method": method,
    }


def _extract_image(file_path: str, filename: str) -> Dict[str, Any]:
    """Extract text from an image file via OCR."""
    text = extract_text_from_image(file_path)
    language = detect_language(text)

    return {
        "text": text.strip(),
        "language": language,
        "language_hint": "unknown",
        "language_sample": "",
        "method": "image",
    }


# ── Public API ────────────────────────────────────────────────────

def process_folder(
    folder_path: str,
    *,
    run_inference: bool = False,
    engine: object = None,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process every supported document in *folder_path*.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing documents to process.
    run_inference : bool, optional
        When ``True`` each document is also classified and run through NER
        extraction via the inference engine.  Defaults to ``False`` for
        backward-compatibility with the existing CSV-export workflow.
    engine : InferenceEngine, optional
        A pre-built inference engine instance.  If *run_inference* is ``True``
        and no engine is supplied, :func:`create_default_engine` is called to
        build one automatically.
    max_workers : int, optional
        Cap the number of threads.  ``None`` lets the executor decide.

    Returns
    -------
    list[dict]
        One dictionary per processed file (see :class:`DocumentResult`).
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Lazy-import to avoid circular imports and heavy deps when not needed.
    if run_inference and engine is None:
        from src.pipeline.inference import create_default_engine
        engine = create_default_engine()

    filenames = sorted(
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not filenames:
        logger.warning("No supported documents found in %s", folder_path)
        return []

    logger.info(
        "Starting ingestion of %d document(s) from %s", len(filenames), folder_path
    )

    collector = OutputCollector()
    extraction_futures: List[Tuple[str, Future]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit extraction jobs
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            file_path = os.path.join(folder_path, filename)

            if ext == ".pdf":
                future = executor.submit(_extract_pdf, file_path, filename)
            else:
                future = executor.submit(_extract_image, file_path, filename)
            extraction_futures.append((filename, future))

        # Collect extraction results and optionally run inference
        # File-level timeout: generous (page_timeout * 2) since a file may
        # contain many pages, each with their own page-level timeout.
        try:
            from src.config import settings as _cfg
            file_timeout = _cfg.page_timeout * 2
        except Exception:
            file_timeout = 600

        for filename, future in extraction_futures:
            t0 = time.monotonic()
            try:
                extraction = future.result(timeout=file_timeout)
            except TimeoutError:
                logger.error("Extraction timed out for %s after %ds", filename, file_timeout)
                collector.add(
                    DocumentResult(
                        file=filename,
                        status="extraction_error",
                        error=f"Timed out after {file_timeout}s",
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                    ).as_dict()
                )
                continue
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", filename, exc)
                collector.add(
                    DocumentResult(
                        file=filename,
                        status="extraction_error",
                        error=str(exc),
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                    ).as_dict()
                )
                continue

            doc = DocumentResult(
                file=filename,
                status="ok",
                text=extraction["text"],
                language=extraction["language"],
                language_hint=extraction["language_hint"],
                language_sample=extraction["language_sample"],
                method=extraction["method"],
            )

            # ── Inference pass ────────────────────────────────────
            if run_inference and engine is not None and doc.text:
                try:
                    doc_type = engine.classify(doc.text)

                    if doc_type is None:
                        doc.document_type = "unknown"
                        doc.status = "ok"
                        logger.warning(
                            "Could not classify %s; skipping NER.", filename
                        )
                    else:
                        doc.document_type = doc_type
                        inference_result = engine.process_document(doc_type, doc.text)
                        doc.extracted_data = inference_result.as_dict()
                        doc.validated = inference_result.validated_data is not None
                except Exception as exc:
                    logger.error("Inference failed for %s: %s", filename, exc)
                    doc.status = "inference_error"
                    doc.error = str(exc)

            doc.elapsed_ms = int((time.monotonic() - t0) * 1000)
            collector.add(doc.as_dict())

    # Summary
    logger.info(
        "Ingestion complete: %d ok, %d errors out of %d documents.",
        collector.ok_count,
        collector.error_count,
        collector.count,
    )

    return collector.results()
