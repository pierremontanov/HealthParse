"""DocIQ command-line interface.

Usage
-----
    # Process a folder with inference and export as JSON
    python -m src.cli --input data/generated --output-dir output --format json

    # Export as FHIR resources
    python -m src.cli --input data/generated --output-dir output --format fhir

    # Extraction-only (no classification/NER), export CSV
    python -m src.cli --input data/generated --output-dir output --format csv --no-inference

    # Process a single file
    python -m src.cli --input data/generated/prescription_1.pdf --output-dir output

    # Use a YAML config file
    python -m src.cli --config dociq.yaml

    # Verbose logging
    python -m src.cli --input data/generated --output-dir output --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.pipeline.core_engine import DocIQEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dociq",
        description="DocIQ – AI-powered medical document processing engine.",
    )

    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to a document file or a folder of documents.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for exported results (default: ./output).",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "fhir"],
        default=None,
        help="Export format: 'json' (one file per doc), 'csv' (single table), or 'fhir' (FHIR resources). Default: json.",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        default=None,
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
        default=None,
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to a YAML configuration file. CLI flags override config values.",
    )

    return parser


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file and return it as a dict.

    Supports keys: ``input``, ``output_dir``, ``format``, ``no_inference``,
    ``max_workers``, ``log_level``.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Try yaml first, fall back to a simple key: value parser
    try:
        import yaml  # type: ignore[import-untyped]
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        # Minimal YAML-like parser for simple key: value files
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    # Convert types
                    if value.lower() in ("true", "yes"):
                        value = True
                    elif value.lower() in ("false", "no"):
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.lower() == "null" or value == "~":
                        value = None
                    data[key] = value

    # Normalise key names (YAML uses underscores, CLI uses hyphens)
    normalised: Dict[str, Any] = {}
    for k, v in data.items():
        normalised[k.replace("-", "_")] = v
    return normalised


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

    # ── Load config file (if provided) and merge with CLI flags ───
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = _load_config(args.config)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    # CLI flags override config values; apply defaults last
    input_path_str = args.input or config.get("input") or config.get("input_dir")
    output_dir = args.output_dir or config.get("output_dir", "output")
    fmt = args.format or config.get("format", "json")
    log_level = args.log_level or config.get("log_level", "INFO")
    max_workers = args.max_workers or config.get("max_workers")
    no_inference = (
        args.no_inference
        if args.no_inference is not None and args.no_inference
        else config.get("no_inference", False)
    )

    _configure_logging(str(log_level))
    logger = logging.getLogger("dociq")

    if not input_path_str:
        logger.error("No input specified. Use --input or set 'input' in config file.")
        return 1

    input_path = Path(input_path_str)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    run_inference = not no_inference
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
        batch = engine.process_batch(str(input_path), max_workers=max_workers)
        results = batch.all
    else:
        logger.error("Input is neither a file nor a directory: %s", input_path)
        return 1

    elapsed = time.monotonic() - t0

    # Export
    if results:
        out_path = DocIQEngine.export(
            results,
            output_dir=output_dir,
            fmt=fmt,
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
