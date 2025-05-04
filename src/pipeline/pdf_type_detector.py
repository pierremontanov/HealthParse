import fitz  # PyMuPDF

def is_pdf_text_based(pdf_path, min_char_threshold=10):
    """
    Check if a PDF is text-based by attempting to extract text from its pages.

    Args:
        pdf_path (str): Path to the PDF file.
        min_char_threshold (int): Minimum characters per page to consider it 'text-based'.

    Returns:
        bool: True if text-based, False if scanned/image-based.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        if len(text.strip()) >= min_char_threshold:
            return True
    return False
