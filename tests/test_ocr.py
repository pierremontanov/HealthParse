import os
import pytest
from src.pipeline.ocr import extract_text_from_image

# Path to sample image file for testing
SAMPLE_IMAGE_PATH = "tests/samples/sample_ocr_image.png"  # Make sure this exists

def test_extract_text_from_valid_image():
    """Test that OCR extracts non-empty text from a valid image."""
    text = extract_text_from_image(SAMPLE_IMAGE_PATH)
    assert isinstance(text, str)
    assert len(text.strip()) > 0, "OCR output should not be empty"

def test_extract_text_from_invalid_path():
    """Test that OCR handles an invalid file path gracefully."""
    text = extract_text_from_image("non_existent_image.png")
    assert "[ERROR]" in text or len(text.strip()) == 0
