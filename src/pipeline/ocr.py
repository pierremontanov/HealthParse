import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from src.pipeline.preprocess import preprocess_image


def extract_text_from_image(image_path):
    """
    Load an image file, preprocess it, and extract text using Tesseract OCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        return "[ERROR] Could not load image."
    
    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed)
    return text


def extract_text_from_pdf_ocr(pdf_path):
    """
    Convert each page of a scanned PDF to an image, then extract and combine text.
    """
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for i, page in enumerate(pages):
        img = page.convert("RGB")
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        preprocessed = preprocess_image(open_cv_image)
        text = pytesseract.image_to_string(preprocessed)
        full_text += f"\n--- Page {i+1} ---\n{text}"
    return full_text
