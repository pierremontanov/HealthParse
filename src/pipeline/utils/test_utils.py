import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
print("Injected path:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from pipeline.utils.language import detect_language, is_english
from pipeline.utils.text_utils import clean_text, remove_numbers


def test_detect_language():
    assert detect_language("Hello, how are you?") == "en"
    assert detect_language("Hola, ¿cómo estás?") == "es"
    assert is_english("This is English.")

def test_clean_text():
    assert clean_text("  Hello!!\n\nThis— is ##a test###...") == "Hello This is a test..."
    assert remove_numbers("Order #4455 dated 2023") == "Order # dated "


