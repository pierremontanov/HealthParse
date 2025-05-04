from src.pipeline.language import detect_language

def test_detect_english():
    text = "Hello, this is a test."
    assert detect_language(text) == "en"

def test_detect_spanish():
    text = "Hola, esto es una prueba."
    assert detect_language(text) == "es"

def test_detect_empty():
    assert detect_language("") == "unknown"
