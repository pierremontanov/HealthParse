"""DocIQ OCR benchmark harness.

Runs the full pipeline (OCR → classification → extraction, LLM disabled) over
the benchmark manifest and scores it against ground truth.

Metrics
-------
- CER / WER            : character / word error rate vs ground-truth text
- accent_accuracy      : fraction of accented chars (á é í ó ú ñ ü, upper+lower)
                         in the GT that survive OCR (count-based proxy)
- cls_accuracy         : classified doc_type == expected doc_type
- field_recall         : fraction of GT field values found (fuzzy, accent- and
                         case-insensitive) anywhere in the extracted output
- validated_rate       : fraction of docs passing Pydantic validation
- latency p50 / p95    : per-file wall time (engine.process_file)
- peak_rss_mb          : peak resident memory during the run (psutil, optional)

Usage
-----
    # Score a run (writes benchmarks/results/<run-name>.json + .csv)
    python scripts/benchmark_ocr.py --run-name v4_baseline --tiers clean degraded real

    # Produce draft ground-truth transcriptions for the real tier
    python scripts/benchmark_ocr.py --draft-gt

Notes
-----
- Clean/degraded tiers are PNG/JPG images → always exercise OCR.
- Real-tier entries without gt_text are timed but not text-scored.
- Set DOCIQ_OCR_MODEL (on the v6 branch) to select the engine variant;
  it is recorded in the results metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import unicodedata
from pathlib import Path

# LLM must be off so the benchmark isolates the OCR engine.
os.environ.setdefault("DOCIQ_LLM_ENABLED", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCH_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

ACCENTED = set("áéíóúñüÁÉÍÓÚÑÜ")


# ── Text metrics ─────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Collapse whitespace and strip page markers for fair comparison."""
    import re
    text = re.sub(r"---\s*Page\s+\d+\s*---", " ", text)
    return " ".join(text.split())


def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(gt: str, hyp: str) -> float:
    gt, hyp = normalize(gt), normalize(hyp)
    return levenshtein(gt, hyp) / max(len(gt), 1)


def wer(gt: str, hyp: str) -> float:
    gt_w, hyp_w = normalize(gt).split(), normalize(hyp).split()
    # word-level Levenshtein via token → char mapping
    vocab = {w: chr(0xE000 + i) for i, w in enumerate(dict.fromkeys(gt_w + hyp_w))}
    return levenshtein(
        "".join(vocab[w] for w in gt_w), "".join(vocab[w] for w in hyp_w)
    ) / max(len(gt_w), 1)


def accent_accuracy(gt: str, hyp: str) -> float | None:
    """Count-based proxy: for each accented char, min(hyp_count, gt_count)/gt_count."""
    gt_counts = {c: gt.count(c) for c in ACCENTED if c in gt}
    if not gt_counts:
        return None  # no accented chars in GT → metric not applicable
    total = sum(gt_counts.values())
    kept = sum(min(hyp.count(c), n) for c, n in gt_counts.items())
    return kept / total


# ── Field metrics ────────────────────────────────────────────────

def _fold(s: str) -> str:
    """Lowercase + strip accents for lenient matching."""
    nfkd = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _leaf_values(obj) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_leaf_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_leaf_values(v))
    elif obj is not None:
        s = str(obj).strip()
        if s:
            out.append(s)
    return out


def field_recall(gt_fields: dict, extracted: dict | None) -> tuple[float | None, int, int]:
    """Fraction of GT leaf values found (folded substring) in extracted output.

    The extractor code is identical across benchmark runs — only the OCR text
    changes — so recall of GT values through the whole pipeline is a fair
    engine-vs-engine comparison even without per-key field mapping.
    """
    gt_vals = [v for v in _leaf_values(gt_fields) if len(_fold(v)) >= 3]
    if not gt_vals:
        return None, 0, 0
    if not extracted:
        return 0.0, 0, len(gt_vals)
    haystack = " | ".join(_fold(v) for v in _leaf_values(extracted))
    found = sum(1 for v in gt_vals if _fold(v) in haystack)
    return found / len(gt_vals), found, len(gt_vals)


# ── Memory sampler ───────────────────────────────────────────────

class PeakRss:
    def __init__(self) -> None:
        self.peak_mb = None
        self._stop = threading.Event()
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    def __enter__(self):
        if self._proc:
            self.peak_mb = 0.0
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / (1024 * 1024)
            self.peak_mb = max(self.peak_mb, rss)
            self._stop.wait(0.25)

    def __exit__(self, *exc):
        if self._proc:
            self._stop.set()
            self._thread.join(timeout=2)


# ── Main ─────────────────────────────────────────────────────────

def load_manifest(tiers: list[str]) -> list[dict]:
    manifest = json.loads((BENCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [e for e in manifest if e["tier"] in tiers]


def run(args: argparse.Namespace) -> int:
    entries = load_manifest(args.tiers)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("No manifest entries for tiers:", args.tiers, file=sys.stderr)
        return 1

    print(f"Benchmark: {len(entries)} files, tiers={args.tiers}, run={args.run_name}")

    t_import = time.monotonic()
    from src.pipeline.core_engine import DocIQEngine
    from src.pipeline.ocr_paddle import warmup

    engine = DocIQEngine(run_inference=True)
    warmup_ok = warmup()
    model_load_s = time.monotonic() - t_import
    print(f"Engine ready in {model_load_s:.1f}s (warmup ok: {warmup_ok})")

    rows: list[dict] = []
    with PeakRss() as mem:
        for n, entry in enumerate(entries, 1):
            path = BENCH_DIR / entry["file"]
            row = {k: entry.get(k) for k in
                   ("id", "tier", "variant", "doc_type", "language")}
            try:
                t0 = time.monotonic()
                result = engine.process_file(str(path))
                row["latency_s"] = round(time.monotonic() - t0, 3)
                row["status"] = result.get("status")

                hyp_text = result.get("text") or ""
                if not hyp_text and result.get("pages"):
                    hyp_text = " ".join(
                        json.dumps(p.get("extracted_data") or {}, ensure_ascii=False)
                        for p in result["pages"]
                    )

                # classification
                got_type = result.get("document_type")
                row["predicted_type"] = got_type
                row["cls_ok"] = (
                    got_type == entry["doc_type"] if entry["doc_type"] else None
                )
                row["validated"] = bool(result.get("validated"))

                # text metrics
                if entry.get("gt_text"):
                    gt = (BENCH_DIR / entry["gt_text"]).read_text(encoding="utf-8")
                    row["cer"] = round(cer(gt, hyp_text), 4)
                    row["wer"] = round(wer(gt, hyp_text), 4)
                    acc = accent_accuracy(gt, hyp_text)
                    row["accent_acc"] = None if acc is None else round(acc, 4)

                # field metrics
                if entry.get("gt_fields"):
                    gtf = json.loads(
                        (BENCH_DIR / entry["gt_fields"]).read_text(encoding="utf-8")
                    )
                    rec, found, total = field_recall(gtf, result.get("extracted_data"))
                    row["field_recall"] = None if rec is None else round(rec, 4)
                    row["fields_found"] = found
                    row["fields_total"] = total
            except Exception as exc:  # noqa: BLE001 — record and continue
                row["status"] = "harness_error"
                row["error"] = str(exc)[:300]

            rows.append(row)
            if n % 25 == 0 or n == len(entries):
                print(f"  {n}/{len(entries)} done")

    # ── Aggregate ──
    def _agg(key, tier=None, fn=statistics.median):
        vals = [r[key] for r in rows
                if r.get(key) is not None and (tier is None or r["tier"] == tier)]
        return round(fn(vals), 4) if vals else None

    summary: dict = {"overall": {}, "per_tier": {}}
    for tier in sorted({r["tier"] for r in rows}):
        tr = [r for r in rows if r["tier"] == tier]
        cls_vals = [r["cls_ok"] for r in tr if r.get("cls_ok") is not None]
        summary["per_tier"][tier] = {
            "files": len(tr),
            "cer_median": _agg("cer", tier),
            "wer_median": _agg("wer", tier),
            "accent_acc_median": _agg("accent_acc", tier),
            "field_recall_median": _agg("field_recall", tier),
            "cls_accuracy": round(sum(cls_vals) / len(cls_vals), 4) if cls_vals else None,
            "validated_rate": round(
                sum(1 for r in tr if r.get("validated")) / len(tr), 4
            ),
            "latency_p50_s": _agg("latency_s", tier),
            "latency_p95_s": _agg(
                "latency_s", tier,
                fn=lambda v: sorted(v)[min(len(v) - 1,
                                           max(0, math.ceil(len(v) * 0.95) - 1))],
            ),
            "errors": sum(1 for r in tr if r.get("status") == "harness_error"),
        }
    summary["overall"] = {
        "model_load_s": round(model_load_s, 1),
        "peak_rss_mb": None if mem.peak_mb is None else round(mem.peak_mb, 1),
    }

    payload = {
        "run_name": args.run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ocr_model": os.environ.get("DOCIQ_OCR_MODEL", "pp-ocrv4"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tiers": args.tiers,
        "summary": summary,
        "results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{args.run_name}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    out_csv = RESULTS_DIR / f"{args.run_name}.csv"
    keys = ["id", "tier", "variant", "language", "doc_type", "predicted_type",
            "status", "cls_ok", "validated", "cer", "wer", "accent_acc",
            "field_recall", "latency_s", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nResults: {out_json}")
    print(json.dumps(summary, indent=1))
    return 0


def draft_gt(args: argparse.Namespace) -> int:
    """OCR the real tier and write draft transcriptions for human correction."""
    entries = load_manifest(["real"])
    from src.pipeline.core_engine import DocIQEngine

    engine = DocIQEngine(run_inference=False)
    gt_dir = BENCH_DIR / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        out = gt_dir / f"{entry['id']}.draft.txt"
        if out.exists() and not args.force:
            print(f"skip (exists): {out.name}")
            continue
        try:
            result = engine.process_file(str(BENCH_DIR / entry["file"]))
            out.write_text(result.get("text") or "", encoding="utf-8")
            print(f"drafted: {out.name}  ({len(result.get('text') or '')} chars, "
                  f"method={result.get('method')})")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {entry['id']}: {exc}", file=sys.stderr)

    print("\nReview each .draft.txt, correct it, and save as <id>.txt "
          "(then re-run build_benchmark.py to link them in the manifest).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None, help="results filename stem")
    p.add_argument("--tiers", nargs="+", default=["clean", "degraded", "real"],
                   choices=["clean", "degraded", "real"])
    p.add_argument("--limit", type=int, default=0, help="only first N files (smoke test)")
    p.add_argument("--draft-gt", action="store_true",
                   help="produce draft transcriptions for the real tier and exit")
    p.add_argument("--force", action="store_true", help="overwrite existing drafts")
    args = p.parse_args()

    if args.draft_gt:
        return draft_gt(args)
    if not args.run_name:
        p.error("--run-name is required for a scoring run")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
