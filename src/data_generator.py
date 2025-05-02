import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

DATA_PATH = "C:/Users/PIERRE/Aivora/Projects/DocIQ/data/aivora_sample_documents.json"
OUTPUT_DIR = "C:/Users/PIERRE/Aivora/Projects/DocIQ/data/generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def render_text(doc, doc_type):
    lines = []

    if doc_type == "prescription":
        lines += [
            f"Patient Name: {doc.get('patient_name', '[missing]')}",
            f"Patient ID: {doc.get('patient_id', '[missing]')}",
            f"Date of Birth: {doc.get('patient_dob', doc.get('patient_birth_date', '[missing]'))}",
            f"Date of Prescription: {doc.get('prescription_date', '[missing]')}",
            f"Doctor: {doc.get('doctor_name', '[missing]')}",
            f"Clinic: {doc.get('clinic', '[missing]')}",
            "",
            "Prescription:",
            doc.get("prescription", "[missing]")
        ]

    elif doc_type == "lab_result":
        lines += [
            f"Patient Name: {doc.get('patient_name', '[missing]')}",
            f"Patient ID: {doc.get('patient_id', '[missing]')}",
            f"Date of Birth: {doc.get('patient_dob', doc.get('patient_birth_date', '[missing]'))}",
            f"Exam Date: {doc.get('exam_date', '[missing]')}",
            f"Clinic: {doc.get('clinic', '[missing]')}",
            "",
            "Test Results:"
        ]

        tests = doc.get("tests", [])
        if isinstance(tests, list):
            for t in tests:
                lines.append(
                    f"- {t.get('test_name', '[test name missing]')}: {t.get('patient_result', '[result missing]')} (Ref: {t.get('reference_range', '[range missing]')})"
                )
        else:
            lines.append("[WARNING] Tests data not structured as expected.")

        lines.append("")
        lines.append(f"Summary: {doc.get('summary', '[missing]')}")

    elif doc_type == "clinic_history":
        lines += [
            f"Patient Name: {doc.get('patient_name', '[missing]')}",
            f"Patient ID: {doc.get('patient_id', '[missing]')}",
            f"Date of Birth: {doc.get('patient_dob', doc.get('patient_birth_date', '[missing]'))}",
            f"Clinic: {doc.get('clinic', '[missing]')}",
            "",
            "Annotations:"
        ]
        for entry in doc.get("annotations", []):
            lines.append(f"- {entry.get('date', '[date missing]')}: {entry.get('note', '[note missing]')}")

    else:
        lines += [f"[WARNING] Unknown or missing type: {doc_type}", f"Document content: {doc}"]

    return lines

def create_pdf(text_lines, filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 40
    for line in text_lines:
        c.drawString(40, y, line)
        y -= 14
    c.save()

def create_image(text_lines, filename):
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 10
    for line in text_lines:
        draw.text((10, y), line, fill="black", font=font)
        y += 15
    img.save(filename)

# Load and generate
with open(DATA_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

# Normalize type and generate
doc_type_map = {
    "medical_prescription": "prescription",
    "lab_results": "lab_result",
    "clinic_history": "clinic_history",
    "prescription": "prescription",
    "lab_result": "lab_result"
}

for i, doc in enumerate(docs):
    raw_type = doc.get("type") or doc.get("document_type", "")
    normalized_type = raw_type.strip().lower().replace(" ", "_")
    doc_type = doc_type_map.get(normalized_type, "unknown")

    lines = render_text(doc, doc_type)
    base_name = f"{doc_type}_{i+1}"
    create_pdf(lines, os.path.join(OUTPUT_DIR, f"{base_name}.pdf"))
    create_image(lines, os.path.join(OUTPUT_DIR, f"{base_name}.png"))

print("✅ Done! PDFs and PNGs generated.")
