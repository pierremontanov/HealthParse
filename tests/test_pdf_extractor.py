from src.pipeline.pdf_extractor import is_pdf_text_based, extract_text_directly
import os

def test_text_based_pdf():
    sample_pdf = "data/generated/sample_text_based.pdf"
    if os.path.exists(sample_pdf):
        assert is_pdf_text_based(sample_pdf) is True

def test_extract_text_directly():
    sample_pdf = "data/generated/sample_text_based.pdf"
    if os.path.exists(sample_pdf):
        text = extract_text_directly(sample_pdf)
        assert isinstance(text, str)
        assert len(text) > 10
