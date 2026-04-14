"""Tests for src.pipeline.utils – language helpers and text utilities."""

import pytest

from src.pipeline.utils.language import detect_language, is_english, is_spanish
from src.pipeline.utils.text_utils import (
    clean_text,
    lowercase,
    normalize_whitespace,
    remove_numbers,
    strip_non_ascii,
    truncate,
)


# ═══════════════════════════════════════════════════════════════════
# Language helpers
# ═══════════════════════════════════════════════════════════════════

def test_detect_language_en():
    assert detect_language("Hello, how are you?") == "en"


def test_detect_language_es():
    assert detect_language("Hola, ¿cómo estás?") == "es"


def test_is_english():
    assert is_english("This is English.")


def test_is_spanish():
    assert is_spanish("Esta es una prueba de idioma en español.")


# ═══════════════════════════════════════════════════════════════════
# Text utilities – existing
# ═══════════════════════════════════════════════════════════════════

def test_clean_text():
    assert clean_text("  Hello!!\n\nThis— is ##a test###...") == "Hello This is a test..."


def test_remove_numbers():
    assert remove_numbers("Order #4455 dated 2023") == "Order # dated "


def test_lowercase():
    assert lowercase("Hello World") == "hello world"


# ═══════════════════════════════════════════════════════════════════
# Text utilities – new helpers
# ═══════════════════════════════════════════════════════════════════

class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("a   b   c") == "a b c"

    def test_collapses_newlines(self):
        assert normalize_whitespace("line1\n\n\nline2") == "line1 line2"

    def test_strips_edges(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_tabs(self):
        assert normalize_whitespace("a\t\tb") == "a b"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""


class TestStripNonAscii:
    def test_removes_emoji(self):
        result = strip_non_ascii("Hello 🌍 World")
        assert "🌍" not in result
        assert "Hello" in result

    def test_keeps_latin1(self):
        assert strip_non_ascii("café résumé") == "café résumé"

    def test_removes_cjk(self):
        result = strip_non_ascii("Hello 你好 World")
        assert "你好" not in result

    def test_empty_string(self):
        assert strip_non_ascii("") == ""


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("hello", max_length=100) == "hello"

    def test_truncates_at_word_boundary(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = truncate(text, max_length=20)
        assert result.endswith("...")
        assert len(result) <= 20

    def test_exact_length(self):
        text = "exact"
        assert truncate(text, max_length=5) == "exact"

    def test_custom_suffix(self):
        text = "a very long string that needs to be cut"
        result = truncate(text, max_length=15, suffix="…")
        assert result.endswith("…")

    def test_no_space_in_range(self):
        text = "abcdefghijklmnop"
        result = truncate(text, max_length=10, suffix="...")
        assert len(result) <= 13  # cut + suffix

    def test_empty_string(self):
        assert truncate("", max_length=10) == ""
