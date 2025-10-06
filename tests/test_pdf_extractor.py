import os
import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is required for PDF threading tests")
from src.pipeline.pdf_extractor import (
    is_pdf_text_based,
    extract_text_directly,
)

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


def test_extract_text_directly_with_page_results(tmp_path):
    doc = fitz.open()
    for index in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Sample page {index + 1}")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()

    classifier_calls = []

    def classifier(text: str, page_index: int) -> str:
        classifier_calls.append(page_index)
        return f"label-{page_index + 1}"

    text, page_results = extract_text_directly(
        str(pdf_path),
        page_classifier=classifier,
        return_page_results=True,
    )

    assert "--- Page 1 ---" in text
    assert "--- Page 2 ---" in text
    assert len(page_results) == 2
    assert {entry["page"] for entry in page_results} == {1, 2}
    assert {entry["classification"] for entry in page_results} == {
        "label-1",
        "label-2",
    }
    assert set(classifier_calls) == {0, 1}
