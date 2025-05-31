import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from pipeline.pdf_extractor import extract_text_from_pdf
from pipeline.utils.text_utils import clean_text, remove_numbers, lowercase

def preprocess_text(text: str) -> str:
    """Apply all text cleaning steps."""
    return lowercase(remove_numbers(clean_text(text)))

@pytest.mark.parametrize("file_path", [
    "C:/Users/PIERRE/Aivora/Projects/DocIQ/data/Test/Test_32_HC.pdf"
])
def test_pipeline_extraction_and_cleaning(file_path):
    assert os.path.exists(file_path), f"❌ File not found: {file_path}"

    raw_text = extract_text_from_pdf(file_path)
    assert isinstance(raw_text, str)
    assert len(raw_text) > 0, "Extracted text is empty."

    cleaned_text = preprocess_text(raw_text)
    assert isinstance(cleaned_text, str)
    assert len(cleaned_text) > 0, "Cleaned text is empty."
