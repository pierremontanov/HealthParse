import os
import pytest
from src.pipeline.pdf_extractor import is_pdf_text_based, extract_text_directly

@pytest.mark.skipif(not os.path.exists("data/generated/sample_text_based.pdf"), reason="Sample PDF not found")
def test_text_based_pdf():
    sample_pdf = "data/generated/sample_text_based.pdf"
    assert is_pdf_text_based(sample_pdf) is True

@pytest.mark.skipif(not os.path.exists("data/generated/sample_text_based.pdf"), reason="Sample PDF not found")
def test_extract_text_directly():
    sample_pdf = "data/generated/sample_text_based.pdf"
    text = extract_text_directly(sample_pdf)
    assert isinstance(text, str)
    assert len(text) > 10
