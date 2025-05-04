"""
Aivora - DocIQ: OCR Processing and Language Detection Runner

This script serves as the main entry point for processing a folder of medical documents.
It imports logic from modular components and exports the results to CSV.

Author: Jean Pierre Montano (Aivora Project)
"""

import pandas as pd
from src.pipeline.process_folder import process_folder

if __name__ == "__main__":
    folder = "C:/Users/PIERRE/Aivora/Projects/DocIQ/data/generated"
    results = process_folder(folder)

    df = pd.DataFrame(results)
    df.to_csv("C:/Users/PIERRE/Aivora/Projects/DocIQ/data/generated/ocr_results.csv", index=False, encoding="utf-8")

    print("✅ OCR complete. Results saved to ocr_results.csv")
