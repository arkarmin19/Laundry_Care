"""Load the symbol class list from configs/symbols.yaml (single source of truth)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_FILE = ROOT / "configs" / "symbols.yaml"

CATEGORY_ORDER = [
    "washing",
    "bleaching",
    "tumble_drying",
    "natural_drying",
    "ironing",
    "professional",
]

CATEGORY_LABELS = {
    "washing": "Washing",
    "bleaching": "Bleaching",
    "tumble_drying": "Tumble drying",
    "natural_drying": "Natural drying",
    "ironing": "Ironing",
    "professional": "Professional care",
}


@lru_cache(maxsize=None)
def load_symbols(path: str | None = None) -> tuple[dict, ...]:
    with open(path or SYMBOLS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    symbols = tuple(data["symbols"])
    names = [s["name"] for s in symbols]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate class names in symbols.yaml")
    return symbols


def class_names() -> list[str]:
    """Class names in YOLO id order."""
    return [s["name"] for s in load_symbols()]


def by_name() -> dict[str, dict]:
    return {s["name"]: s for s in load_symbols()}
