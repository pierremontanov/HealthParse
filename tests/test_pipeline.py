import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline.pdf_extractor import extract_text_from_pdf
from pipeline.utils.text_utils import clean_text, remove_numbers, lowercase

def preprocess_text(text: str) -> str:
    """Apply all text cleaning steps."""
    return lowercase(remove_numbers(clean_text(text)))

def test_pipeline(file_path: str):
    print(f"\n🧪 Testing: {file_path}")
    
    # Step 1: Extract raw text
    raw_text = extract_text_from_pdf(file_path)
    print("\n📄 Extracted text preview:\n")
    print(raw_text[:500])

    # Step 2: Clean text
    cleaned_text = preprocess_text(raw_text)
    print("\n🧼 Cleaned text preview:\n")
    print(cleaned_text[:500])

if __name__ == "__main__":
    test_file = "C:/Users/PIERRE/Aivora/Projects/DocIQ/data/Test/Test_1.pdf"
    if os.path.exists(test_file):
        test_pipeline(test_file)
    else:
        print(f"❌ File not found: {test_file}")