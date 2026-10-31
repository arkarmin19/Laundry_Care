"""Merge several YOLO-format datasets into one pool with CareScan class ids.

Each source is a folder exported from Roboflow in "YOLOv8" format:
    source/
      data.yaml            (has `names:`)
      train/images, train/labels
      valid/images, valid/labels   (optional)
      test/images,  test/labels    (optional)

All splits are pooled (split_dataset.py makes our own split later).

Usage:
    python scripts/merge_datasets.py \
        --public downloads/laundry_set_a downloads/laundry_set_b \
        --own    downloads/our_photos \
        --out    data/merged

Images from --own get the prefix "own__" so split_dataset.py can put most of
them into the test set.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carescan.symbols import ROOT, class_names  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that refuses duplicate keys (PyYAML silently keeps the last one)."""


def _unique_mapping(loader, node, deep=False):
    keys = [norm(loader.construct_object(k, deep=deep)) for k, _ in node.value]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        raise SystemExit(f"class_aliases.yaml has duplicate keys: {dups}")
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_alias_map(path: Path) -> dict[str, str]:
    ours = {n.lower(): n for n in class_names()}
    with open(path, encoding="utf-8") as f:
        aliases = yaml.load(f, Loader=_UniqueKeyLoader).get("aliases", {}) or {}
    custom = {norm(k): (v if v in ("drop", "skip") else ours.get(str(v).lower()))
              for k, v in aliases.items()}
    bad = [k for k, v in custom.items() if v is None]
    if bad:
        raise SystemExit(f"class_aliases.yaml maps to unknown CareScan names: {bad}")
    amap = dict(ours)   # our own names map to themselves...
    amap.update(custom)  # ...unless an alias says otherwise (e.g. merging rare classes)
    return amap


def source_names(src: Path) -> list[str]:
    with open(src / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):  # {0: name, 1: name}
        names = [names[k] for k in sorted(names)]
    return list(names)


def to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """YOLO box line -> (cx, cy, w, h). Polygon lines (segmentation export) are
    converted to their bounding box."""
    if len(values) == 4:
        return tuple(values)
    xs, ys = values[0::2], values[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def merge(public: list[Path], own: list[Path], out: Path, alias_file: Path) -> dict:
    amap = load_alias_map(alias_file)
    ids = {n: i for i, n in enumerate(class_names())}
    if out.exists():
        shutil.rmtree(out)  # start clean, so re-running after fixing aliases leaves no old files
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "skipped_unmapped": 0, "skipped_by_alias": 0, "boxes": Counter(),
             "dropped": Counter(), "skip_classes": Counter(), "unmapped": Counter()}
    for src, is_own in [(p, False) for p in public] + [(p, True) for p in own]:
        names = source_names(src)
        tag = ("own__" if is_own else "pub__") + norm(src.name)
        for img in sorted(src.rglob("*")):
            if img.suffix.lower() not in IMG_EXT or img.parent.name != "images":
                continue
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            lines_out, unmapped, skip = [], False, False
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    src_name = names[int(parts[0])]
                    target = amap.get(norm(src_name))
                    if target is None:
                        stats["unmapped"][f"{src.name}:{src_name}"] += 1
                        unmapped = True
                        continue
                    if target == "drop":
                        stats["dropped"][src_name] += 1
                        continue
                    if target == "skip":
                        stats["skip_classes"][src_name] += 1
                        skip = True
                        continue
                    cx, cy, w, h = to_bbox([float(v) for v in parts[1:]])
                    lines_out.append(f"{ids[target]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats["boxes"][target] += 1
            if unmapped:
                # Some symbols in this image would be unlabelled -> skip the image.
                stats["skipped_unmapped"] += 1
                continue
            if skip:
                # Image contains a class we can't map reliably -> leave the whole image out.
                stats["skipped_by_alias"] += 1
                continue
            stem = f"{tag}__{img.parent.parent.name}__{img.stem}"
            shutil.copy2(img, out / "images" / (stem + img.suffix.lower()))
            (out / "labels" / (stem + ".txt")).write_text("\n".join(lines_out))
            stats["images"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", nargs="*", default=[], type=Path)
    ap.add_argument("--own", nargs="*", default=[], type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "merged", type=Path)
    ap.add_argument("--aliases", default=ROOT / "configs" / "class_aliases.yaml", type=Path)
    args = ap.parse_args()
    if not args.public and not args.own:
        ap.error("give at least one --public or --own dataset folder")

    s = merge(args.public, args.own, args.out, args.aliases)
    print(f"\nMerged {s['images']} images into {args.out}")
    print(f"Boxes per class: {dict(sorted(s['boxes'].items()))}")
    if s["dropped"]:
        print(f"Dropped boxes (mapped to 'drop'): {dict(s['dropped'])}")
    if s["skip_classes"]:
        print(f"{s['skipped_by_alias']} images left out (contain a class mapped to 'skip'): "
              f"{dict(s['skip_classes'])}")
    if s["unmapped"]:
        print(f"\n!! {s['skipped_unmapped']} images skipped because of UNMAPPED classes.")
        print("   Add these to configs/class_aliases.yaml and run again:")
        for k, v in s["unmapped"].most_common():
            print(f"     {k}  ({v} boxes)")


if __name__ == "__main__":
    main()
