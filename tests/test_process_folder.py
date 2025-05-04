import os
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.process_folder import process_folder

@pytest.fixture
def fake_folder(tmp_path):
    # Create dummy files in a temporary directory
    files = ["doc1.pdf", "image1.png", "note.txt"]
    for f in files:
        (tmp_path / f).write_text("fake content")
    return tmp_path

@patch("src.pipeline.process_folder.extract_text_from_pdf_ocr")
@patch("src.pipeline.process_folder.extract_text_from_image")
@patch("src.pipeline.process_folder.detect_language")
def test_process_folder_valid_files(
    mock_detect_language,
    mock_extract_image,
    mock_extract_pdf,
    fake_folder
):
    mock_extract_pdf.return_value = "PDF text"
    mock_extract_image.return_value = "Image text"
    mock_detect_language.return_value = "en"

    results = process_folder(str(fake_folder))

    # Should skip unsupported file (note.txt), process 2
    assert len(results) == 2
    filenames = [res["file"] for res in results]
    assert "doc1.pdf" in filenames
    assert "image1.png" in filenames
    assert all("text" in res for res in results)
    assert all(res["language"] == "en" for res in results)
