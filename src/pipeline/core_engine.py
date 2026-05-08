"""DocIQ Core Engine – single entry-point for document processing.

:class:`DocIQEngine` wraps the inference engine, folder ingestion,
and output persistence behind a clean interface that can be called from
the CLI, a FastAPI endpoint, or any other integration point.

Usage
-----
    from src.pipeline.core_engine import DocIQEngine

    engine = DocIQEngine()

    # Process one file
    result = engine.process_file("data/generated/prescription_1.pdf")

    # Process a folder
    results = engine.process_batch("data/generated")

    # Export results
    engine.export(results, output_dir="output", fmt="json")
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.pipeline.metrics import Timer, get_collector

from src.pipeline.exceptions import (
    ClassificationError,
    DocIQError,
    DocumentExtractionError,
    ExportError,
    UnsupportedFileError,
)
from src.pipeline.inference import InferenceEngine, InferenceResult, create_default_engine
from src.pipeline.process_folder import (
    DocumentResult,
    SUPPORTED_EXTENSIONS,
    _extract_image,
    _extract_pdf,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = ["DocIQEngine", "EngineResult"]


class EngineResult:
    """Thin wrapper around a batch of :class:`DocumentResult` dicts.

    Provides convenience accessors for filtering, counting, and exporting.
    """

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self._results = results

    # ── Accessors ──

    @property
    def all(self) -> List[Dict[str, Any]]:
        return list(self._results)

    @property
    def ok(self) -> List[Dict[str, Any]]:
        return [r for r in self._results if r["status"] == "ok"]

    @property
    def errors(self) -> List[Dict[str, Any]]:
        return [r for r in self._results if r["status"] != "ok"]

    @property
    def count(self) -> int:
        return len(self._results)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return counts

    # ── Iteration / indexing ──

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, idx):
        return self._results[idx]

    def __repr__(self) -> str:
        s = self.summary()
        parts = [f"{k}={v}" for k, v in s.items()]
        return f"EngineResult({', '.join(parts)})"


class DocIQEngine:
    """High-level orchestrator for DocIQ document processing.

    Parameters
    ----------
    inference_engine : InferenceEngine, optional
        A pre-built inference engine.  When ``None`` the default engine
        with rule-based extractors is created automatically.
    run_inference : bool
        Whether to classify and extract entities from documents.
        ``True`` by default — set to ``False`` to get extraction-only
        output (text + language + method metadata).
    """

    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        *,
        run_inference: bool = True,
    ) -> None:
        self._run_inference = run_inference
        if run_inference:
            self._engine = inference_engine or create_default_engine()
        else:
            self._engine = inference_engine
        logger.info("DocIQEngine initialised (inference=%s).", run_inference)

    # ── Page splitting ───────────────────────────────────────────

    _PAGE_SPLIT_RE = re.compile(r"\n---\s*Page\s+\d+\s*---\n")

    @staticmethod
    def _split_pages(text: str) -> List[str]:
        """Split OCR / direct-extracted text into per-page chunks.

        The PDF extractor inserts ``--- Page N ---`` markers between
        pages.  This method splits on those markers and returns a list
        of non-empty page texts.
        """
        parts = DocIQEngine._PAGE_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    # ── Single-file processing ────────────────────────────────────

    def process_file(
        self, file_path: str, *, split_pages: bool = True,
    ) -> Dict[str, Any]:
        """Process a single document and return a result dict.

        For multi-page PDFs (e.g. a bundle of scanned documents), each
        page is classified and extracted independently when
        *split_pages* is ``True``.  The returned dict contains a
        ``"pages"`` key with per-page results alongside the top-level
        combined result.

        Parameters
        ----------
        file_path : str
            Path to a PDF or image file.
        split_pages : bool
            When ``True`` (default), multi-page documents are split and
            each page is processed independently.

        Returns
        -------
        dict
            A :class:`DocumentResult`-compatible dictionary.  When
            page splitting is active, the dict also includes a
            ``"pages"`` list with one result per page.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(path.name, ext)

        filename = path.name
        t0 = time.monotonic()
        collector = get_collector()

        # ── Extraction ──
        try:
            with Timer("extraction") as t_ext:
                if ext == ".pdf":
                    extraction = _extract_pdf(str(path), filename)
                else:
                    extraction = _extract_image(str(path), filename)
            collector.record("extraction", t_ext.elapsed_ms)
            collector.record(f"extraction.{extraction['method']}", t_ext.elapsed_ms)
        except Exception as exc:
            collector.record_error("extraction")
            logger.error("Extraction failed for %s: %s", filename, exc)
            raise DocumentExtractionError(filename, str(exc)) from exc

        full_text = extraction["text"]

        # ── Multi-page splitting ──
        pages = self._split_pages(full_text) if split_pages else []
        is_multipage = len(pages) > 1

        if is_multipage and self._run_inference and self._engine is not None:
            # Process each page independently
            page_results = self._process_pages(
                pages, filename, extraction, collector,
            )

            # Build top-level summary from the first successful page
            first_ok = next(
                (p for p in page_results if p.get("extracted_data")), None,
            )
            doc = DocumentResult(
                file=filename,
                status="ok",
                text="",  # omit full text to keep response compact
                language=extraction["language"],
                language_hint=extraction["language_hint"],
                language_sample=extraction["language_sample"],
                method=extraction["method"],
                document_type=first_ok["document_type"] if first_ok else "unknown",
                extracted_data=first_ok.get("extracted_data") if first_ok else None,
                validated=first_ok.get("validated", False) if first_ok else False,
            )
            doc.elapsed_ms = int((time.monotonic() - t0) * 1000)
            result = doc.as_dict()
            result["pages"] = page_results
            result["page_count"] = len(pages)
            collector.record("process_file", doc.elapsed_ms)
            collector.increment("documents_processed")
            return result

        # ── Single-page / single-document flow ──
        doc = DocumentResult(
            file=filename,
            status="ok",
            text=full_text,
            language=extraction["language"],
            language_hint=extraction["language_hint"],
            language_sample=extraction["language_sample"],
            method=extraction["method"],
        )

        if self._run_inference and self._engine is not None and doc.text:
            try:
                with Timer("classification") as t_cls:
                    doc_type = self._engine.classify(doc.text)
                collector.record("classification", t_cls.elapsed_ms)

                if doc_type is None:
                    doc.document_type = "unknown"
                    logger.warning("Could not classify %s.", filename)
                elif doc_type not in self._engine.registered_types:
                    doc.document_type = doc_type
                    logger.info(
                        "Classified %s as '%s' but no extractor registered.",
                        filename, doc_type,
                    )
                else:
                    doc.document_type = doc_type
                    with Timer("ner_extraction") as t_ner:
                        inference_result = self._engine.process_document(doc_type, doc.text)
                    collector.record("ner_extraction", t_ner.elapsed_ms)
                    doc.extracted_data = inference_result.as_dict()
                    doc.validated = inference_result.validated_data is not None
            except Exception as exc:
                collector.record_error("inference")
                logger.error("Inference failed for %s: %s", filename, exc)
                doc.status = "inference_error"
                doc.error = str(exc)

        doc.elapsed_ms = int((time.monotonic() - t0) * 1000)
        collector.record("process_file", doc.elapsed_ms)
        collector.increment("documents_processed")
        return doc.as_dict()

    def _process_pages(
        self,
        pages: List[str],
        filename: str,
        extraction: Dict[str, Any],
        collector: Any,
    ) -> List[Dict[str, Any]]:
        """Classify and extract each page independently."""
        page_results: List[Dict[str, Any]] = []

        for i, page_text in enumerate(pages, 1):
            page_label = f"{filename} (page {i})"
            entry: Dict[str, Any] = {
                "page": i,
                "status": "ok",
                "document_type": None,
                "extracted_data": None,
                "validated": False,
                "error": None,
            }

            if not page_text.strip():
                entry["status"] = "skipped"
                entry["error"] = "Empty page"
                page_results.append(entry)
                continue

            try:
                doc_type = self._engine.classify(page_text)
                if doc_type is None:
                    entry["document_type"] = "unknown"
                    logger.debug("Could not classify %s.", page_label)
                elif doc_type not in self._engine.registered_types:
                    # Classified (e.g. "receipt") but no extractor registered
                    entry["document_type"] = doc_type
                    logger.debug(
                        "Classified %s as '%s' but no extractor registered.",
                        page_label, doc_type,
                    )
                else:
                    entry["document_type"] = doc_type
                    inference_result = self._engine.process_document(doc_type, page_text)
                    entry["extracted_data"] = inference_result.as_dict()
                    entry["validated"] = inference_result.validated_data is not None
            except Exception as exc:
                logger.error("Inference failed for %s: %s", page_label, exc)
                entry["status"] = "inference_error"
                entry["error"] = str(exc)

            page_results.append(entry)

        return page_results

    # ── Batch processing ──────────────────────────────────────────

    def process_batch(
        self,
        folder_path: str,
        *,
        max_workers: Optional[int] = None,
    ) -> EngineResult:
        """Process every supported document in a folder.

        Parameters
        ----------
        folder_path : str
            Path to the directory containing documents.
        max_workers : int, optional
            Cap on thread-pool size.

        Returns
        -------
        EngineResult
            Wrapper with convenience accessors over the list of dicts.
        """
        from src.pipeline.process_folder import process_folder

        results = process_folder(
            folder_path,
            run_inference=self._run_inference,
            engine=self._engine,
            max_workers=max_workers,
        )
        return EngineResult(results)

    # ── Export helpers ─────────────────────────────────────────────

    @staticmethod
    def export(
        results: EngineResult | List[Dict[str, Any]],
        *,
        output_dir: str,
        fmt: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """Persist results to disk.

        Delegates to :func:`src.pipeline.output_formatter.export_results`.

        Parameters
        ----------
        results : EngineResult or list[dict]
            The results to export.
        output_dir : str
            Directory to write output into (created if missing).
        fmt : str
            ``"json"`` | ``"csv"`` | ``"fhir"``.
        filename : str, optional
            Override the output filename / sub-directory name.

        Returns
        -------
        str
            Path to the output file or directory.
        """
        from src.pipeline.output_formatter import export_results

        items = list(results) if not isinstance(results, list) else results
        return export_results(items, output_dir=output_dir, fmt=fmt, filename=filename)
