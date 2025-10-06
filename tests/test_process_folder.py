import os
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from src.pipeline.process_folder import process_folder


@pytest.fixture
def fake_folder(tmp_path):
    # Create dummy files in a temporary directory
    files = ["doc1.pdf", "image1.png", "note.txt"]
    for f in files:
        (tmp_path / f).write_text("fake content")
    return tmp_path


@patch("src.pipeline.process_folder.detect_pdf_language")
@patch("src.pipeline.process_folder.extract_text_from_pdf_ocr")
@patch("src.pipeline.process_folder.extract_text_from_image")
@patch("src.pipeline.process_folder.detect_language")
@patch("src.pipeline.process_folder.is_pdf_text_based")
def test_process_folder_valid_files(
    mock_is_pdf_text_based,
    mock_detect_language,
    mock_extract_image,
    mock_extract_pdf,
    mock_detect_pdf_language,
    fake_folder,
):
    mock_is_pdf_text_based.return_value = True
    mock_extract_pdf.return_value = "PDF text"
    mock_extract_image.return_value = "Image text"
    mock_detect_language.return_value = "en"
    mock_detect_pdf_language.return_value = SimpleNamespace(
        language="en", text_sample="PDF text"
    )

    results = process_folder(str(fake_folder))

    # Should skip unsupported file (note.txt), process 2
    assert len(results) == 2
    filenames = [res["file"] for res in results]
    assert "doc1.pdf" in filenames
    assert "image1.png" in filenames
    assert all("text" in res for res in results)
    assert all(res["language"] == "en" for res in results)
    assert all(res["language_hint"] == "en" for res in results if res["file"] == "doc1.pdf")
