"""Compare Tesseract vs EasyOCR vs PaddleOCR output on specific PDF pages.

Usage:
    python scripts/compare_ocr.py path/to/PRUEBAS.pdf --pages 2 23
    python scripts/compare_ocr.py path/to/PRUEBAS.pdf --pages 2 23 --output comparison.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Disable oneDNN as belt-and-suspenders for PaddleOCR
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
# Skip slow HuggingFace/ModelScope connectivity check on startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


# ── Engine availability checks ──────────────────────────────────

_TESSERACT_AVAILABLE: bool | None = None
_EASYOCR_AVAILABLE: bool | None = None
_PADDLEOCR_AVAILABLE: bool | None = None


def _check_tesseract() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
    except Exception:
        _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE


def _check_easyocr() -> bool:
    global _EASYOCR_AVAILABLE
    if _EASYOCR_AVAILABLE is not None:
        return _EASYOCR_AVAILABLE
    try:
        import easyocr  # noqa: F401
        _EASYOCR_AVAILABLE = True
    except Exception:
        _EASYOCR_AVAILABLE = False
    return _EASYOCR_AVAILABLE


def _check_paddleocr() -> bool:
    global _PADDLEOCR_AVAILABLE
    if _PADDLEOCR_AVAILABLE is not None:
        return _PADDLEOCR_AVAILABLE
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        _PADDLEOCR_AVAILABLE = True
    except Exception:
        _PADDLEOCR_AVAILABLE = False
    return _PADDLEOCR_AVAILABLE


# ── OCR runners ─────────────────────────────────────────────────

def ocr_tesseract(page_img: Image.Image) -> tuple[str, float]:
    """Run Tesseract OCR. Returns (text, elapsed_seconds)."""
    if not _check_tesseract():
        return "(Tesseract not available)", 0.0

    import pytesseract

    rgb = page_img.convert("RGB")
    cv_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    pil_pre = Image.fromarray(binary)

    t0 = time.time()
    text = pytesseract.image_to_string(pil_pre, lang="eng+spa")
    return text, time.time() - t0


def ocr_easyocr(page_img: Image.Image, reader) -> tuple[str, float]:
    """Run EasyOCR. Returns (text, elapsed_seconds)."""
    rgb = page_img.convert("RGB")
    img_array = np.array(rgb)

    t0 = time.time()
    result = reader.readtext(img_array)
    elapsed = time.time() - t0

    lines = [det[1] for det in result]
    return "\n".join(lines), elapsed


def ocr_paddle(page_img: Image.Image, ocr_instance) -> tuple[str, float]:
    """Run PaddleOCR 3.4. Returns (text, elapsed_seconds)."""
    rgb = page_img.convert("RGB")
    img_array = np.array(rgb)

    t0 = time.time()
    results = list(ocr_instance.predict(img_array))
    elapsed = time.time() - t0

    lines = []
    for result in results:
        if hasattr(result, "rec_texts"):
            lines.extend(result.rec_texts)
        elif isinstance(result, dict) and "rec_texts" in result:
            lines.extend(result["rec_texts"])
    return "\n".join(lines), elapsed


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare OCR engines on PDF pages.",
    )
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument(
        "--pages", nargs="+", type=int, default=[2, 23],
        help="1-indexed page numbers to test (default: 2 23).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for PDF-to-image conversion (default: 300).",
    )
    parser.add_argument(
        "--output",
        help="Save comparison to a file instead of stdout.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    # ── Auto-detect bundled Poppler ──
    poppler_path = None
    script_dir = Path(__file__).resolve().parent.parent
    for candidate in [
        script_dir / "poppler" / "Library" / "bin",
        script_dir / "poppler" / "bin",
        script_dir / "poppler",
    ]:
        if (candidate / "pdftoppm.exe").exists() or (candidate / "pdftoppm").exists():
            poppler_path = str(candidate)
            break

    # ── Convert PDF ──
    print("Converting PDF pages to images...", file=sys.stderr)
    convert_kwargs = {"dpi": args.dpi}
    if poppler_path:
        convert_kwargs["poppler_path"] = poppler_path
        print(f"  Using Poppler at: {poppler_path}", file=sys.stderr)

    t0 = time.time()
    all_pages = convert_from_path(str(pdf_path), **convert_kwargs)
    print(
        f"  {len(all_pages)} pages converted in {time.time() - t0:.1f}s\n",
        file=sys.stderr,
    )

    # ── Init engines ──
    easy_reader = None
    if _check_easyocr():
        print("Initialising EasyOCR...", file=sys.stderr)
        import easyocr
        easy_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        print("  EasyOCR ready.", file=sys.stderr)

    paddle_ocr = None
    if _check_paddleocr():
        print("Initialising PaddleOCR 3.4...", file=sys.stderr)
        from paddleocr import PaddleOCR
        paddle_ocr = PaddleOCR(
            ocr_version="PP-OCRv4",
            lang="en",              # English/Latin model handles Spanish text
            enable_mkldnn=False,
        )
        print("  PaddleOCR ready.", file=sys.stderr)

    print(file=sys.stderr)

    # ── Compare each page ──
    for page_num in args.pages:
        idx = page_num - 1
        if idx < 0 or idx >= len(all_pages):
            print(f"Page {page_num} out of range (1–{len(all_pages)}), skipping.", file=out)
            continue

        page_img = all_pages[idx]
        sep = "=" * 70
        results = {}

        # Tesseract
        try:
            text, elapsed = ocr_tesseract(page_img)
        except Exception as e:
            text, elapsed = f"ERROR: {e}", 0.0
        results["Tesseract"] = (text, elapsed)

        # EasyOCR
        if easy_reader:
            try:
                text, elapsed = ocr_easyocr(page_img, easy_reader)
            except Exception as e:
                text, elapsed = f"ERROR: {e}", 0.0
            results["EasyOCR"] = (text, elapsed)

        # PaddleOCR
        if paddle_ocr:
            try:
                text, elapsed = ocr_paddle(page_img, paddle_ocr)
            except Exception as e:
                text, elapsed = f"ERROR: {e}", 0.0
            results["PaddleOCR"] = (text, elapsed)

        # ── Output ──
        for engine, (text, elapsed) in results.items():
            print(f"\n{sep}", file=out)
            print(f"PAGE {page_num} — {engine.upper()}  [{elapsed:.2f}s, {len(text)} chars]", file=out)
            print(sep, file=out)
            print(text, file=out)

        print(f"\n{'─' * 70}", file=out)
        print(f"PAGE {page_num} SUMMARY:", file=out)
        for engine, (text, elapsed) in results.items():
            print(f"  {engine:>12}: {len(text):>5} chars in {elapsed:.2f}s", file=out)
        print(f"{'─' * 70}\n", file=out)

    if args.output:
        out.close()
        print(f"Comparison saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
