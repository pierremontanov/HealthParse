from types import SimpleNamespace
from unittest.mock import patch

from src.pipeline.language import detect_language, detect_pdf_language


def test_detect_english():
    text = "Hello, this is a test."
    assert detect_language(text) == "en"


def test_detect_spanish():
    text = "Hola, esto es una prueba."
    assert detect_language(text) == "es"


def test_detect_empty():
    assert detect_language("") == "unknown"


@patch("src.pipeline.language.fitz.open")
def test_detect_pdf_language(mock_open):
    class DummyDoc(list):
        def close(self):
            pass

    page = SimpleNamespace(get_text=lambda mode="text": "Hello world" if mode == "text" else "")
    mock_open.return_value = DummyDoc([page])

    result = detect_pdf_language("dummy.pdf", max_pages=1)

    assert result.language == "en"
    assert "Hello" in result.text_sample
