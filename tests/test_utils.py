import sys
import os

# Manually add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline.utils.language import detect_language, is_english
from pipeline.utils.text_utils import clean_text, remove_numbers

def test_detect_language_en():
    assert detect_language("Hello, how are you?") == "en"

def test_detect_language_es():
    assert detect_language("Hola, ¿cómo estás?") == "es"

def test_is_english():
    assert is_english("This is English.")

def test_clean_text():
    assert clean_text("  Hello!!\n\nThis— is ##a test###...") == "Hello This is a test..."

def test_remove_numbers():
    assert remove_numbers("Order #4455 dated 2023") == "Order # dated "
