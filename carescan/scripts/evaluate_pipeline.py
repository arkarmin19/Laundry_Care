"""End-to-end evaluation on our own photos against ground_truth.csv.

mAP (from the training notebook) measures the detector alone. This script
measures what the USER gets: are the final symbols right after conflict
resolution, is the fabric read correctly, and how long does a scan take?

ground_truth.csv columns:
    image, symbols (space-separated class names), fabric_text, garment_colour, notes

Usage:
    python scripts/evaluate_pipeline.py --model models/best.pt \
        --images data/own_photos --truth data/ground_truth.csv [--no-ocr]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carescan.fabric import main_fabrics, parse_fabric  # noqa: E402
from carescan.pipeline import CareScanPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    pipe = CareScanPipeline(args.model, conf=args.conf, use_ocr=not args.no_ocr)
    tp, fp, fn = Counter(), Counter(), Counter()
    exact, fabric_ok, fabric_n, times, rows = 0, 0, 0, [], []

    with open(args.truth, newline="", encoding="utf-8") as f:
        truth = list(csv.DictReader(f))
    for r in truth:
        img = cv2.imread(str(args.images / r["image"]))
        if img is None:
            print(f"  missing image: {r['image']}")
            continue
        res = pipe.run(img, colour=r.get("garment_colour") or None)
        pred, gt = set(res.guide.symbols), set(r["symbols"].split())
        for s in pred & gt:
            tp[s] += 1
        for s in pred - gt:
            fp[s] += 1
        for s in gt - pred:
            fn[s] += 1
        exact += pred == gt
        times.append(res.info["timings_ms"]["total"])
        if not args.no_ocr and r.get("fabric_text"):
            fabric_n += 1
            fabric_ok += main_fabrics(res.fabrics) == main_fabrics(parse_fabric(r["fabric_text"]))
        rows.append({"image": r["image"], "truth": " ".join(sorted(gt)),
                     "predicted": " ".join(sorted(pred)), "missed": " ".join(sorted(gt - pred)),
                     "extra": " ".join(sorted(pred - gt)), "ocr": res.ocr_text,
                     "ms": res.info["timings_ms"]["total"]})

    n = len(rows)
    if not n:
        raise SystemExit("No images evaluated.")
    T, P, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    prec = T / (T + P) if T + P else 0
    rec = T / (T + N) if T + N else 0
    print(f"\nImages evaluated: {n}")
    print(f"Symbol precision: {prec:.1%}   recall: {rec:.1%}")
    print(f"Labels fully correct (all symbols right): {exact / n:.1%}")
    if fabric_n:
        print(f"Fabric read correctly: {fabric_ok / fabric_n:.1%} ({fabric_n} labels)")
    print(f"Time per scan: median {statistics.median(times):.0f} ms, max {max(times)} ms "
          f"(target < 5000 ms)")
    print("\nPer class (tp / fp / fn):")
    for s in sorted(set(tp) | set(fp) | set(fn)):
        print(f"  {s:20s} {tp[s]:3d} / {fp[s]:3d} / {fn[s]:3d}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-image results written to {args.out} (use it to find failure examples)")


if __name__ == "__main__":
    main()
