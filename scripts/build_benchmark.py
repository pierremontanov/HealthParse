"""Build the OCR benchmark dataset under ``data/benchmark/``.

Creates:

    data/benchmark/
    ├── manifest.json        # one entry per file
    ├── clean/               # synthetic PNGs (EN + ES) copied from data/generated
    ├── real/                # real Spanish PDFs copied from data/Test
    └── ground_truth/        # <id>.txt (exact rendered text) + <id>.fields.json

Ground truth for the clean tier is reconstructed from
``data/aivora_sample_documents.json`` using the exact same rendering logic as
``src/data_generator.py`` (file index ``{type}_{i+1}`` maps to JSON index ``i``).

Real-tier entries start with ``gt_text: null`` — draft transcriptions are
produced by ``benchmark_ocr.py --draft-gt`` and must be human-corrected before
the real tier is scored.

Usage:
    python scripts/build_benchmark.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_JSON = PROJECT_ROOT / "data" / "aivora_sample_documents.json"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
REAL_DIR = PROJECT_ROOT / "spanish_test_set"  # 44 PRUEBAS-*.pdf
BENCH_DIR = PROJECT_ROOT / "data" / "benchmark"

# JSON document_type → pipeline classifier type
TYPE_MAP = {
    "Medical Prescription": "prescription",
    "Lab Results": "result",
    "Clinic History": "clinical_history",
}
# JSON document_type → generated filename prefix (data_generator naming)
FILE_PREFIX = {
    "Medical Prescription": "prescription",
    "Lab Results": "lab_result",
    "Clinic History": "clinic_history",
}
# Real-tier doc types are unknown up front (PRUEBAS-*.pdf carry no label).
# They are labelled during the ground-truth correction session via
# ground_truth/real_labels.json: {"pruebas-1": "prescription", ...}
REAL_LABELS_FILE = "real_labels.json"


def render_text(doc: dict, doc_type: str) -> list[str]:
    """Replica of src.data_generator.render_text (kept in sync manually —
    the generator module executes generation at import time, so it cannot
    be imported safely)."""
    lines: list[str] = []
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
            doc.get("prescription", "[missing]"),
        ]
    elif doc_type == "lab_result":
        lines += [
            f"Patient Name: {doc.get('patient_name', '[missing]')}",
            f"Patient ID: {doc.get('patient_id', '[missing]')}",
            f"Date of Birth: {doc.get('patient_dob', doc.get('patient_birth_date', '[missing]'))}",
            f"Exam Date: {doc.get('exam_date', '[missing]')}",
            f"Clinic: {doc.get('clinic', '[missing]')}",
            "",
            "Test Results:",
        ]
        tests = doc.get("tests", [])
        if isinstance(tests, list):
            for t in tests:
                lines.append(
                    f"- {t.get('test_name', '[test name missing]')}: "
                    f"{t.get('patient_result', '[result missing]')} "
                    f"(Ref: {t.get('reference_range', '[range missing]')})"
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
            "Annotations:",
        ]
        for entry in doc.get("annotations", []):
            lines.append(f"- {entry.get('date', '[date missing]')}: {entry.get('note', '[note missing]')}")
    return lines


def gt_fields(doc: dict) -> dict:
    """Ground-truth field values (everything except metadata keys)."""
    return {k: v for k, v in doc.items() if k not in ("document_type", "language")}


# ── High-quality PNG rendering ───────────────────────────────────
# data_generator.create_image uses PIL's tiny bitmap font and crops long
# lines at the image edge, so GT text would not match the pixels. The
# benchmark renders its own images: TrueType font, wrapped lines, no crop.

_FONT_CANDIDATES = [
    "arial.ttf",                                   # Windows
    "DejaVuSans.ttf",                              # Linux (matplotlib/PIL)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int = 18):
    from PIL import ImageFont
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    raise RuntimeError(
        "No TrueType font found — install DejaVuSans or Arial. "
        "Rendering with the bitmap default would corrupt the benchmark."
    )


def render_png(lines: list[str], out_path: Path,
               width: int = 800, margin: int = 20, line_h: int = 26) -> None:
    from PIL import Image, ImageDraw
    font = _load_font()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_w = width - 2 * margin

    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        words, cur = line.split(" "), ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if probe.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                if cur:
                    wrapped.append(cur)
                cur = w
        wrapped.append(cur)

    height = 2 * margin + line_h * max(len(wrapped), 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for line in wrapped:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_h
    img.save(out_path, "PNG")


def main() -> int:
    if not SAMPLE_JSON.exists():
        print(f"ERROR: {SAMPLE_JSON} not found", file=sys.stderr)
        return 1

    (BENCH_DIR / "clean").mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "real").mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "ground_truth").mkdir(parents=True, exist_ok=True)

    docs = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    manifest: list[dict] = []
    skipped = 0

    # ── Tier: clean (synthetic PNGs, EN + ES, rendered here) ──
    for i, doc in enumerate(docs):
        jtype = doc["document_type"]
        prefix = FILE_PREFIX[jtype]
        lines = render_text(doc, prefix)

        entry_id = f"{prefix}_{i + 1}_clean"
        png_path = BENCH_DIR / "clean" / f"{prefix}_{i + 1}.png"
        if not png_path.exists() or "--force" in sys.argv:
            render_png(lines, png_path)

        gt_txt = BENCH_DIR / "ground_truth" / f"{prefix}_{i + 1}.txt"
        gt_txt.write_text("\n".join(lines), encoding="utf-8")

        gt_json = BENCH_DIR / "ground_truth" / f"{prefix}_{i + 1}.fields.json"
        gt_json.write_text(
            json.dumps(gt_fields(doc), ensure_ascii=False, indent=1), encoding="utf-8"
        )

        manifest.append({
            "id": entry_id,
            "file": f"clean/{png_path.name}",
            "tier": "clean",
            "doc_type": TYPE_MAP[jtype],
            "language": doc["language"],
            "gt_text": f"ground_truth/{prefix}_{i + 1}.txt",
            "gt_fields": f"ground_truth/{prefix}_{i + 1}.fields.json",
        })

    # ── Tier: real (44 Spanish PRUEBAS PDFs) ──
    labels_path = BENCH_DIR / "ground_truth" / REAL_LABELS_FILE
    labels: dict = (
        json.loads(labels_path.read_text(encoding="utf-8"))
        if labels_path.exists() else {}
    )
    if REAL_DIR.exists():
        def _num(p: Path) -> int:
            m = re.search(r"(\d+)", p.stem)
            return int(m.group(1)) if m else 0

        for pdf in sorted(REAL_DIR.glob("*.pdf"), key=_num):
            entry_id = pdf.stem.lower()
            shutil.copy2(pdf, BENCH_DIR / "real" / pdf.name)

            gt_txt_path = BENCH_DIR / "ground_truth" / f"{entry_id}.txt"
            manifest.append({
                "id": entry_id,
                "file": f"real/{pdf.name}",
                "tier": "real",
                "doc_type": labels.get(entry_id),
                "language": "es",
                "gt_text": (
                    f"ground_truth/{entry_id}.txt" if gt_txt_path.exists() else None
                ),
                "gt_fields": None,
            })
    else:
        print(f"WARNING: {REAL_DIR} not found — real tier skipped. "
              "Extract spanish_test_set.zip there first.", file=sys.stderr)

    (BENCH_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    tiers: dict[str, int] = {}
    for e in manifest:
        tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
    print(f"Manifest written: {BENCH_DIR / 'manifest.json'}")
    print(f"Entries per tier: {tiers}  (skipped: {skipped})")
    print("Note: run scripts/degrade_images.py next to add the degraded tier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
