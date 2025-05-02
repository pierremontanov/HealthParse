import os
import cv2
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# OPTIONAL: If Tesseract is not in your system PATH, set the path manually (uncomment and update the line below)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Supported file types for OCR
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf"]

def preprocess_image(image):
    """
    Convert image to grayscale and apply thresholding for better OCR accuracy.

    Args:
        image (numpy.ndarray): Original image in BGR format.

    Returns:
        numpy.ndarray: Preprocessed image ready for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text_from_image(image_path):
    """
    Load an image file, preprocess it, and extract text using Tesseract OCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return "[ERROR] Could not load image."
    
    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed)
    return text

def extract_text_from_pdf(pdf_path):
    """
    Convert each page of a PDF to an image, then extract and combine text from all pages.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Combined text extracted from all pages.
    """
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for i, page in enumerate(pages):
        img = page.convert("RGB")  # Convert PIL page to RGB image
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format
        preprocessed = preprocess_image(open_cv_image)
        text = pytesseract.image_to_string(preprocessed)
        full_text += f"\n--- Page {i+1} ---\n{text}"
    return full_text

def process_folder(folder_path):
    """
    Process all supported files in a folder, extract text, and collect results.

    Args:
        folder_path (str): Path to the folder containing documents.

    Returns:
        list[dict]: List of dictionaries with file name and extracted text.
    """
    results = []

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        file_path = os.path.join(folder_path, filename)

        if ext not in SUPPORTED_EXTENSIONS:
            continue  # Skip unsupported files

        print(f"Processing: {filename}")

        # Apply the appropriate OCR function
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_image(file_path)

        # Save results in structured format
        results.append({
            "file": filename,
            "text": text.strip()
        })

    return results

if __name__ == "__main__":
    # Define input folder with generated documents (PDFs or images)
    folder = "C:/Users/PIERRE/Aivora/Projects/DocIQ/data/generated"

    # Run OCR on the folder
    ocr_results = process_folder(folder)

    # Export results to a CSV file for later analysis or verification
    df = pd.DataFrame(ocr_results)
    df.to_csv("C:/Users/PIERRE/Aivora/Projects/DocIQ/data/generated/ocr_results.csv", index=False, encoding="utf-8")

    print("✅ OCR complete. Results saved to ocr_results.csv")
