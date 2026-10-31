"""Split the merged pool into train / val / test and write data.yaml.

Our own phone photos are the most realistic test of the app, so a larger share
of them goes to the test set. Public images fill train/val.

Default split:
    own photos    : 40% train, 15% val, 45% test
    public images : 80% train, 15% val,  5% test
With ~100 own photos and ~1,000 public images this gives roughly 70/15/15
overall, and the test set is mostly real photos we took ourselves.

Usage:
    python scripts/split_dataset.py --src data/merged --out data/carescan
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carescan.symbols import ROOT, class_names  # noqa: E402


def place(src: Path, dst: Path) -> None:
    """Hard-link instead of copying (instant, no extra disk space). Falls back to
    a normal copy when src and dst are on different drives."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def group_key(path: Path) -> str:
    """Roboflow names files "<original>_jpg.rf.<hash>.jpg". Augmented copies of the
    same photo share <original> but have different hashes, and may sit in different
    export folders (train/valid/test). All copies must land in the SAME split,
    otherwise the test set contains near-copies of training images (data leakage)
    and the test mAP is too optimistic."""
    stem = path.stem.split(".rf.")[0]
    return re.sub(r"__(train|valid|test)__", "__", stem)


def split_list(items: list[Path], fracs: tuple[float, float, float], rng: random.Random):
    """Split by group, so augmented copies of one photo stay together."""
    groups: dict[str, list[Path]] = {}
    for p in items:
        groups.setdefault(group_key(p), []).append(p)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(n * fracs[0])
    n_val = round(n * fracs[1])
    parts = keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:]
    return tuple([p for k in part for p in groups[k]] for part in parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=ROOT / "data" / "merged", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "carescan", type=Path)
    ap.add_argument("--own", default="0.40,0.15,0.45", help="train,val,test fractions for own photos")
    ap.add_argument("--public", default="0.80,0.15,0.05", help="train,val,test fractions for public images")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    imgs = sorted(p for p in (args.src / "images").iterdir() if p.is_file())
    own = [p for p in imgs if p.name.startswith("own__")]
    pub = [p for p in imgs if not p.name.startswith("own__")]
    f_own = tuple(float(x) for x in args.own.split(","))
    f_pub = tuple(float(x) for x in args.public.split(","))
    n_groups = len({group_key(p) for p in imgs})
    print(f"{len(imgs)} images from {n_groups} original photos "
          f"({len(imgs) - n_groups} augmented copies, kept in the same split as their original)")

    splits = {"train": [], "val": [], "test": []}
    for group, fr in ((own, f_own), (pub, f_pub)):
        tr, va, te = split_list(group, fr, rng)
        splits["train"] += tr
        splits["val"] += va
        splits["test"] += te

    if args.out.exists():
        shutil.rmtree(args.out)
    total_files = sum(len(v) for v in splits.values())
    print(f"Writing {total_files} images to {args.out} ...", flush=True)
    done = 0
    for name, files in splits.items():
        (args.out / name / "images").mkdir(parents=True)
        (args.out / name / "labels").mkdir(parents=True)
        for img in files:
            place(img, args.out / name / "images" / img.name)
            lbl = args.src / "labels" / (img.stem + ".txt")
            if lbl.exists():
                place(lbl, args.out / name / "labels" / lbl.name)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total_files}", flush=True)

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(class_names())},
    }
    with open(args.out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    total = sum(len(v) for v in splits.values()) or 1
    print(f"Own photos: {len(own)}   Public images: {len(pub)}")
    for name, files in splits.items():
        n_own = sum(1 for p in files if p.name.startswith("own__"))
        print(f"  {name:5s}: {len(files):5d} images ({len(files) / total:5.1%})  — {n_own} own photos")
    print(f"Wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
