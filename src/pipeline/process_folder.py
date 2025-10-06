import os
from typing import Dict, List

from src.pipeline.language import detect_language, detect_pdf_language
from src.pipeline.ocr import extract_text_from_image
from src.pipeline.pdf_extractor import (
    extract_text_directly,
    extract_text_from_pdf_ocr,
)
from src.pipeline.pdf_type_detector import is_pdf_text_based


SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf"]


def _finalise_language(full_text: str, fallback: str) -> str:
    detected = detect_language(full_text)
    if detected == "unknown":
        return fallback
    return detected


def process_folder(folder_path: str) -> List[Dict[str, str]]:
    """Process documents in a folder and collect OCR/NER routing metadata."""
    results: List[Dict[str, str]] = []

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")

        language_hint = "unknown"
        text_sample = ""

        if ext == ".pdf":
            language_result = detect_pdf_language(file_path)
            language_hint = language_result.language
            text_sample = language_result.text_sample

            if is_pdf_text_based(file_path):
                text = extract_text_directly(file_path)
                method = "direct"
            else:
                text = extract_text_from_pdf_ocr(file_path)
                method = "ocr"

            language = _finalise_language(text, language_hint)
        else:
            text = extract_text_from_image(file_path)
            method = "image"
            language = detect_language(text)

        results.append(
            {
                "file": filename,
                "text": text.strip(),
                "language": language,
                "language_hint": language_hint,
                "language_sample": text_sample,
                "method": method,
            }
        )

    return results


print("✅ process_folder.py loaded")
