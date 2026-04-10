"""
DocIQ – AI-powered medical document processing engine.

Entry point for batch or single-file processing.  Delegates to the CLI
module which handles argument parsing, logging, and output export.

Usage
-----
    python -m src.main --input data/generated --output-dir output
    python -m src.main --input data/generated/prescription_1.pdf -o output -f csv
    python -m src.main --help

Author: Jean Pierre Montano (Amphibian Labs)
"""

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
