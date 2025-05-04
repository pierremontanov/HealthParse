import os
from src.pipeline.ocr import extract_text_from_image
from src.pipeline.language import detect_language
from src.pipeline.pdf_extractor import extract_text_from_pdf_ocr
from src.pipeline.pdf_type_detector import is_pdf_text_based
from src.pipeline.pdf_extractor import extract_text_directly, is_pdf_text_based



SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf"]

def process_folder(folder_path):
    """
    Process documents in a folder:
    - Perform OCR (image or scanned PDF)
    - Detect language
    - Return list of structured results
    """
    results = []

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        file_path = os.path.join(folder_path, filename)

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        print(f"Processing: {filename}")

        if ext == ".pdf":
            if is_pdf_text_based(file_path):
                text = extract_text_directly(file_path)
                method = "direct"
            else:
                text = extract_text_from_pdf_ocr(file_path)
                method = "ocr"
        else:
            text = extract_text_from_image(file_path)
            method = "image"

        language = detect_language(text)

        results.append({
            "file": filename,
            "text": text.strip(),
            "language": detect_language(text),
            "method": method
        })

    return results

print("✅ process_folder.py loaded")
