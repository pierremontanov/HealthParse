import re
import unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,:/-]", "", text)
    return text.strip()

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)

def lowercase(text: str) -> str:
    return text.lower()


