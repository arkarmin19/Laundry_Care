"""Rule-based reasoning: detected symbols + fabric + user context -> care guide.

This is the knowledge-representation part of the project:
  1. Clean up detections (confidence filter, one symbol per exclusive group).
  2. Turn each symbol into a plain-language step (from configs/symbols.yaml).
  3. Apply IF-THEN rules that combine symbols, fabric and garment colour into
     warnings and tips.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .fabric import DELICATE, SHRINK_PRONE, SYNTHETIC, main_fabrics
from .symbols import CATEGORY_LABELS, CATEGORY_ORDER, by_name

COLOURS = ("white", "light", "dark", "bright")


@dataclass
class Detection:
    name: str
    confidence: float = 1.0
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 in pixels


@dataclass
class CareWarning:
    level: str  # "danger" | "warning" | "info"
    title: str
    text: str
    rule: str  # id of the rule that fired (useful for the report / debugging)


@dataclass
class CareGuide:
    symbols: list[str]
    steps: list[dict]  # {"category", "symbol", "text"}
    warnings: list[CareWarning] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    missing_categories: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # removed by conflict resolution

    def to_dict(self) -> dict:
        return asdict(self)

    def rules_fired(self) -> list[str]:
        return [w.rule for w in self.warnings]


# A garment label can only have ONE symbol from each of these groups.
# Natural drying is not exclusive: "line dry" + "in shade" is a valid pair.
def _group(sym: dict) -> str | None:
    cat, name = sym["category"], sym["name"]
    if cat == "professional":
        return "wetclean" if "wetclean" in name else "dryclean"
    if cat == "natural_drying":
        return None
    return cat


def resolve_conflicts(
    detections: list[Detection], min_conf: float = 0.25
) -> tuple[list[Detection], list[str]]:
    """Drop low-confidence and unknown detections, then keep the most confident
    detection in each exclusive group. Returns (kept, dropped_names)."""
    known = by_name()
    best: dict[str, Detection] = {}
    free: dict[str, Detection] = {}
    dropped: list[str] = []
    for d in sorted(detections, key=lambda d: -d.confidence):
        sym = known.get(d.name)
        if sym is None or d.confidence < min_conf:
            dropped.append(d.name)
            continue
        g = _group(sym)
        bucket, key = (free, d.name) if g is None else (best, g)
        if key in bucket:
            if bucket[key].name != d.name:
                dropped.append(d.name)
            continue
        bucket[key] = d
    kept = list(best.values()) + list(free.values())
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    kept.sort(key=lambda d: order[known[d.name]["category"]])
    return kept, dropped


def build_care_guide(
    detections: list[Detection],
    fabrics: dict[str, int | None] | None = None,
    colour: str | None = None,
    is_new: bool = False,
    label_notes: list[str] | None = None,
    min_conf: float = 0.25,
) -> CareGuide:
    known = by_name()
    fabrics = fabrics or {}
    kept, dropped = resolve_conflicts(detections, min_conf)
    names = [d.name for d in kept]
    present = set(names)
    cats = {known[n]["category"] for n in names}
    fabs = main_fabrics(fabrics)

    steps = [
        {"category": known[n]["category"], "symbol": n, "text": known[n]["instruction"]}
        for n in names
    ]
    guide = CareGuide(symbols=names, steps=steps, dropped=dropped)
    W = guide.warnings

    def warn(level, title, text, rule):
        W.append(CareWarning(level, title, text, rule))

    wash = next((known[n] for n in names if known[n]["category"] == "washing"), None)
    wash_temp = wash["max_temp"] if wash else None

    # ---- No symbols at all ----
    if not names:
        warn("danger", "No symbols found",
             "We couldn't read any care symbols. Retake the photo closer, in good light, "
             "with the label flat.", "R00_no_symbols")
        return guide

    # ---- Cleaning method ----
    if "no_wash" in present:
        if present & {"dryclean_P", "dryclean_F", "dryclean_any", "wetclean_W"}:
            warn("warning", "Professional cleaning only",
                 "This item can't be washed at home — take it to a dry cleaner.",
                 "R01_dryclean_only")
        elif "no_dryclean" in present:
            warn("warning", "Spot clean only",
                 "No washing and no dry-cleaning: spot clean stains with a damp cloth.",
                 "R02_spot_clean")

    # ---- Shrinking ----
    shrinky = fabs & SHRINK_PRONE
    if shrinky and wash_temp and wash_temp >= 40:
        f = ", ".join(sorted(shrinky))
        warn("warning", "Shrink risk",
             f"{f.capitalize()} can shrink in warm water. Washing at 30°C is safest; "
             f"never go above {wash_temp}°C.", "R03_shrink_hot_wash")
    if shrinky and "tumble_normal" in present:
        warn("warning", "Shrink risk in the dryer",
             f"High dryer heat shrinks {', '.join(sorted(shrinky))}. Use low heat or hang to dry.",
             "R04_shrink_dryer")

    # ---- Delicate fibres ----
    delicate = fabs & DELICATE
    if delicate:
        f = " and ".join(sorted(delicate))
        warn("warning", f"Delicate fibre: {f}",
             "Use cool water and a wool/delicates detergent. Don't wring or rub, "
             "and lay flat to dry so it keeps its shape.", "R05_delicate_fibre")
        if "wool" in delicate and present & {"tumble_low", "tumble_normal"}:
            warn("danger", "Wool in the dryer",
                 "Even if the label allows it, tumble drying can felt wool. Dry flat if unsure.",
                 "R06_wool_dryer")

    # ---- Stretch fibres ----
    if "elastane" in fabrics and present & {"iron_high", "tumble_normal", "wash_60", "wash_95"}:
        warn("info", "Contains elastane",
             "Heat weakens stretch fibres over time. Lower temperatures keep the fit longer.",
             "R07_elastane_heat")

    # ---- Colour ----
    if colour in ("dark", "bright"):
        if is_new:
            warn("danger", "Wash separately",
                 "New dark or bright clothes often bleed dye. Wash alone for the first few washes.",
                 "R08_new_colour_bleed")
        else:
            guide.tips.append("Wash with similar colours, inside out, to reduce fading.")
        if present & {"bleach_any", "bleach_non_chlorine"}:
            warn("info", "Careful with bleach",
                 "Bleach is allowed, but it can still fade coloured fabric. "
                 "Use colour-safe (oxygen) bleach only if needed.", "R09_colour_bleach")
        if present & {"dry_line", "dry_drip", "dry_flat"} or "no_tumble" in present:
            guide.tips.append("Dry in the shade — direct sun fades dark and bright colours.")
    elif colour == "white":
        guide.tips.append("Wash with other whites to keep them bright.")
    elif colour == "light" and is_new:
        guide.tips.append("Keep away from new dark clothes in the first few washes.")

    # ---- Drying ----
    if "no_tumble" in present and not cats & {"natural_drying"}:
        how = "lay flat" if delicate else "hang"
        guide.tips.append(f"No dryer: {how} to dry instead.")

    # ---- Ironing ----
    if "ironing" not in cats and fabs & SYNTHETIC:
        guide.tips.append("Synthetic fabric: if you iron it, use low heat.")
    if "iron_high" in present and fabs & SYNTHETIC:
        warn("warning", "Check iron heat",
             "The label allows high heat, but the synthetic fibres in this blend can melt. "
             "Start on medium.", "R10_synthetic_iron")

    # ---- Sorting tip ----
    if wash_temp:
        cycle = "gentle" if wash.get("gentle") else "normal"
        group = {"white": "whites", "light": "light colours", "dark": "darks",
                 "bright": "bright colours"}.get(colour, "similar colours")
        guide.tips.append(f"Laundry basket: {group}, {wash_temp}°C {cycle} cycle.")

    # ---- Text printed on the label ----
    for note in label_notes or []:
        if note not in guide.tips:
            guide.tips.append(f"Label says: {note}")

    # ---- Missing categories ----
    for cat in ("washing", "bleaching", "ironing"):
        if cat not in cats:
            guide.missing_categories.append(CATEGORY_LABELS[cat])
    if not cats & {"tumble_drying", "natural_drying"}:
        guide.missing_categories.append("Drying")
    if guide.missing_categories:
        warn("info", "Some symbols not found",
             "No symbol detected for: " + ", ".join(guide.missing_categories)
             + ". If the label has one, retake the photo; otherwise use the gentlest option.",
             "R11_missing_category")
    return guide
