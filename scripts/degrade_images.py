"""Generate the degraded tier of the OCR benchmark.

Applies five deterministic degradations to every clean-tier PNG:

    rot      — rotation (+2° or −2°, alternating by index, white fill)
    noise    — additive Gaussian noise (sigma 12)
    jpeg40   — JPEG re-encode at quality 40
    lowres   — downscale to 50% and back (simulates ~150 DPI scan)
    blur     — Gaussian blur radius 1.2

Each variant is a separate manifest entry sharing the clean entry's ground
truth. Deterministic: noise is seeded from the source filename, so re-running
produces byte-identical images.

Usage:
    python scripts/degrade_images.py          # requires build_benchmark.py first
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "data" / "benchmark"
DEGRADED_DIR = BENCH_DIR / "degraded"

VARIANTS = ("rot", "noise", "jpeg40", "lowres", "blur")


def _seed_for(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def degrade(img: Image.Image, variant: str, index: int, seed: int) -> Image.Image:
    img = img.convert("RGB")
    if variant == "rot":
        angle = 2.0 if index % 2 == 0 else -2.0
        return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    if variant == "noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(img).astype(np.int16)
        arr = arr + rng.normal(0, 12, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if variant == "jpeg40":
        return img  # handled at save time via quality=40
    if variant == "lowres":
        w, h = img.size
        return img.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if variant == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))
    raise ValueError(f"unknown variant: {variant}")


def main() -> int:
    manifest_path = BENCH_DIR / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: manifest.json missing — run scripts/build_benchmark.py first",
              file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_entries = [e for e in manifest if e["tier"] == "clean"]
    # Idempotency: drop previous degraded entries and regenerate
    manifest = [e for e in manifest if e["tier"] != "degraded"]

    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)
    created = 0

    for idx, entry in enumerate(clean_entries):
        src = BENCH_DIR / entry["file"]
        img = Image.open(src)
        stem = src.stem
        seed = _seed_for(src.name)

        for variant in VARIANTS:
            out_name = (f"{stem}__jpeg40.jpg" if variant == "jpeg40"
                        else f"{stem}__{variant}.png")
            out_path = DEGRADED_DIR / out_name
            # Regenerate when missing, forced, or stale (source newer than output).
            # Staleness check keeps the run resumable even after clean-tier updates.
            stale = (
                out_path.exists()
                and out_path.stat().st_mtime < src.stat().st_mtime
            )
            if "--force" in sys.argv or stale or not out_path.exists():
                out = degrade(img, variant, idx, seed)
                if variant == "jpeg40":
                    out.save(out_path, "JPEG", quality=40)
                else:
                    out.save(out_path, "PNG")
                created += 1

            manifest.append({
                "id": f"{stem}_{variant}",
                "file": f"degraded/{out_name}",
                "tier": "degraded",
                "variant": variant,
                "doc_type": entry["doc_type"],
                "language": entry["language"],
                "gt_text": entry["gt_text"],
                "gt_fields": entry["gt_fields"],
            })

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Created {created} new degraded images "
          f"({len([e for e in manifest if e['tier'] == 'degraded'])} total entries).")
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
