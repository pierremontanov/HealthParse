"""DocIQ command-line interface.

Usage
-----
    # Process a folder with inference and export as JSON
    python -m src.cli --input data/generated --output-dir output --format json

    # Extraction-only (no classification/NER), export CSV
    python -m src.cli --input data/generated --output-dir output --format csv --no-inference

    # Process a single file
    python -m src.cli --input data/generated/prescription_1.pdf --output-dir output

    # Verbose logging
    python -m src.cli --input data/generated --output-dir output --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.pipeline.core_engine import DocIQEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dociq",
        description="DocIQ – AI-powered medical document processing engine.",
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a document file or a folder of documents.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for exported results (default: ./output).",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="Export format: 'json' (one file per doc) or 'csv' (single table). Default: json.",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip classification and NER; only extract text and metadata.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads for batch processing.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the DocIQ CLI. Returns 0 on success, 1 on error."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("dociq")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    run_inference = not args.no_inference
    engine = DocIQEngine(run_inference=run_inference)

    t0 = time.monotonic()

    if input_path.is_file():
        logger.info("Processing single file: %s", input_path)
        try:
            result = engine.process_file(str(input_path))
        except Exception as exc:
            logger.error("Failed to process %s: %s", input_path, exc)
            return 1
        results = [result]
    elif input_path.is_dir():
        logger.info("Processing folder: %s", input_path)
        batch = engine.process_batch(str(input_path), max_workers=args.max_workers)
        results = batch.all
    else:
        logger.error("Input is neither a file nor a directory: %s", input_path)
        return 1

    elapsed = time.monotonic() - t0

    # Export
    if results:
        out_path = DocIQEngine.export(
            results,
            output_dir=args.output_dir,
            fmt=args.format,
        )
        logger.info("Exported %d result(s) to %s", len(results), out_path)
    else:
        logger.warning("No documents processed.")

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = len(results) - ok
    logger.info(
        "Done in %.1fs — %d processed, %d ok, %d errors.",
        elapsed,
        len(results),
        ok,
        errors,
    )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
