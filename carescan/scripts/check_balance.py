"""Count boxes per class in each split and flag classes that need more data.

Usage:
    python scripts/check_balance.py --data data/carescan --min 50
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carescan.symbols import ROOT, by_name, class_names  # noqa: E402


def count(label_dir: Path) -> Counter:
    c = Counter()
    if label_dir.exists():
        for f in label_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    c[int(line.split()[0])] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "carescan", type=Path)
    ap.add_argument("--min", default=50, type=int, help="minimum training boxes per class")
    args = ap.parse_args()

    names, info = class_names(), by_name()
    splits = ["train", "val", "test"]
    counts = {s: count(args.data / s / "labels") for s in splits}

    print(f"{'id':>3}  {'class':22s} {'category':15s} " + " ".join(f"{s:>6s}" for s in splits) + "  status")
    low = []
    for i, n in enumerate(names):
        row = [counts[s][i] for s in splits]
        status = "OK"
        if row[0] == 0:
            status = "MISSING"
        elif row[0] < args.min:
            status = f"LOW (<{args.min})"
        if status != "OK":
            low.append(n)
        print(f"{i:>3}  {n:22s} {info[n]['category']:15s} " + " ".join(f"{v:6d}" for v in row) + f"  {status}")
    totals = [sum(counts[s].values()) for s in splits]
    print(f"\nTotal boxes: " + ", ".join(f"{s}={t}" for s, t in zip(splits, totals)))
    if low:
        print(f"\nNeed more examples for {len(low)} classes: {', '.join(low)}")
        print("Options: photograph more labels with these symbols, or merge rare classes")
        print("(e.g. wash_30_gentle -> wash_30) in configs/class_aliases.yaml.")


if __name__ == "__main__":
    main()
