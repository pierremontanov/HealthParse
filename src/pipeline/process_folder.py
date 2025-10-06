import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Tuple

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


def _process_pdf(file_path: str, filename: str) -> Dict[str, str]:
    print(f"Processing: {filename}")

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

    return {
        "file": filename,
        "text": text.strip(),
        "language": language,
        "language_hint": language_hint,
        "language_sample": text_sample,
        "method": method,
    }


def _process_image(file_path: str, filename: str) -> Dict[str, str]:
    print(f"Processing: {filename}")

    text = extract_text_from_image(file_path)
    language = detect_language(text)

    return {
        "file": filename,
        "text": text.strip(),
        "language": language,
        "language_hint": "unknown",
        "language_sample": "",
        "method": "image",
    }


def process_folder(folder_path: str) -> List[Dict[str, str]]:
    """Process documents in a folder and collect OCR/NER routing metadata."""
    indexed_results: List[Tuple[int, Dict[str, str]]] = []
    pdf_futures: List[Tuple[int, Future[Dict[str, str]]]] = []

    with ThreadPoolExecutor() as executor:
        for index, filename in enumerate(os.listdir(folder_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(folder_path, filename)

            if ext == ".pdf":
                future = executor.submit(_process_pdf, file_path, filename)
                pdf_futures.append((index, future))
            else:
                indexed_results.append((index, _process_image(file_path, filename)))

        for index, future in pdf_futures:
            indexed_results.append((index, future.result()))

    indexed_results.sort(key=lambda item: item[0])
    return [result for _, result in indexed_results]


print("✅ process_folder.py loaded")
