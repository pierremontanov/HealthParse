import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from pipeline.preprocess import preprocess_image
import fitz  # PyMuPDF

def extract_text_from_pdf_ocr(pdf_path):
    """
    Convert each page of a scanned PDF to an image, then extract and combine text using OCR.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Combined text extracted from all pages.
    """
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for i, page in enumerate(pages):
        img = page.convert("RGB")  # PIL to RGB
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL to OpenCV
        preprocessed = preprocess_image(open_cv_image)
        text = pytesseract.image_to_string(preprocessed)
        full_text += f"\n--- Page {i+1} ---\n{text}"
    return full_text

def extract_text_directly(pdf_path):
    """
    Extract text directly from a text-based PDF using PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Combined extracted text from all pages.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {i+1} ---\n{text}"
    return full_text

def is_pdf_text_based(pdf_path, min_char_threshold=10):
    """
    Check if a PDF is text-based by attempting to extract text from its pages.

    Args:
        pdf_path (str): Path to the PDF file.
        min_char_threshold (int): Minimum number of characters to consider the page 'text-based'.

    Returns:
        bool: True if the PDF has extractable text, False if it's likely scanned.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        if len(text.strip()) >= min_char_threshold:
            return True
    return False


def extract_text_from_pdf(pdf_path):
    """
    Automatically detect PDF type and extract text using the appropriate method.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    if is_pdf_text_based(pdf_path):
        return extract_text_directly(pdf_path)
    else:
        return extract_text_from_pdf_ocr(pdf_path)
