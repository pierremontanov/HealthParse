import cv2
from src.pipeline.utils.text_utils import clean_text, remove_numbers, lowercase
from src.pipeline.utils.language import detect_language, is_english, is_spanish

def preprocess_image(image):
    """
    Convert image to grayscale and apply thresholding for better OCR accuracy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return thresh

def preprocess_text(raw_text: str) -> str:
    """
    Clean and normalize extracted text for better NER/classification.
    """
    lang = detect_language(raw_text)
    print(f"[DEBUG] Language detected: {lang}")

    text = clean_text(raw_text)
    text = lowercase(text)
    return text
