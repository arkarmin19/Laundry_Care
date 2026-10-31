"""Parse fabric composition and care notes from OCR text.

Care labels are often printed in several languages, so each fabric has a list of
keywords. Matching is case-insensitive and tolerant of OCR noise like "100 %".

Thai is supported too. Thai is written without spaces between words, so Thai
keywords are found by substring search instead of word splitting.
"""
from __future__ import annotations

import re
import unicodedata

FABRIC_KEYWORDS: dict[str, list[str]] = {
    "cotton": ["cotton", "coton", "algodon", "baumwolle", "cotone", "katoen",
               "คอตตอน", "คอทตอน", "ค็อตตอน", "ฝ้าย"],
    "polyester": ["polyester", "poliester", "polyestere",
                  "โพลีเอสเตอร์", "โพลิเอสเตอร์", "โพลีเอสเตอร", "โพลีเอสเทอร์"],
    "nylon": ["nylon", "polyamide", "poliamida", "polyamid",
              "ไนลอน", "ไนล่อน", "โพลีเอไมด์"],
    "wool": ["wool", "laine", "lana", "wolle", "merino", "cashmere", "cachemire",
             "ขนสัตว์", "ขนแกะ", "วูล", "แคชเมียร์"],
    "silk": ["silk", "soie", "seda", "seide", "seta", "ผ้าไหม", "ไหม"],
    "elastane": ["elastane", "spandex", "lycra", "elasthan", "elastan", "elastam",
                 "สแปนเด็กซ์", "สเปนเด็กซ์", "สแปนเดกซ์", "อีลาสเทน", "ไลครา"],
    "viscose": ["viscose", "rayon", "modal", "lyocell", "tencel", "viscosa", "cellulose",
                "วิสคอส", "วิสโคส", "เรยอน", "เรยอง", "โมดาล", "ไลโอเซลล์"],
    "linen": ["linen", "lin", "lino", "leinen", "flax", "ลินิน", "ลินิ่น"],
    "acrylic": ["acrylic", "acrylique", "acrilico", "acryl",
                "อะคริลิค", "อะคริลิก", "อะครีลิก"],
}

# Short keywords that are only trusted when next to a percentage (avoid false hits).
_NEEDS_PERCENT = {"lin", "lino", "lana", "seda", "acryl", "ไหม", "วูล"}

SHRINK_PRONE = {"cotton", "linen", "wool", "viscose"}
DELICATE = {"wool", "silk"}
SYNTHETIC = {"polyester", "nylon", "acrylic"}

CARE_NOTES = {
    # English (matched with normal spacing)
    "wash separately": "Wash separately",
    "wash inside out": "Wash inside out",
    "with similar colours": "Wash with similar colours",
    "with similar colors": "Wash with similar colours",
    "with like colours": "Wash with similar colours",
    "with like colors": "Wash with similar colours",
    "do not soak": "Do not soak",
    "do not wring": "Do not wring",
    "reshape while damp": "Reshape while damp",
    "iron on reverse": "Iron on the reverse side",
    "cool iron": "Use a cool iron",
    "do not iron print": "Do not iron on the print",
    # Thai (matched with all spaces removed)
    "ซักแยก": "Wash separately",
    "แยกซัก": "Wash separately",
    "กลับด้านซัก": "Wash inside out",
    "กลับด้านก่อนซัก": "Wash inside out",
    "ซักกับสีเดียวกัน": "Wash with similar colours",
    "ซักรวมกับสีเดียวกัน": "Wash with similar colours",
    "ซักกับผ้าสีใกล้เคียง": "Wash with similar colours",
    "ห้ามแช่": "Do not soak",
    "ห้ามบิด": "Do not wring",
    "กลับด้านรีด": "Iron on the reverse side",
    "รีดด้านใน": "Iron on the reverse side",
    "รีดด้านกลับ": "Iron on the reverse side",
    "ห้ามรีดทับลาย": "Do not iron on the print",
    "ห้ามรีดบนลาย": "Do not iron on the print",
    "ห้ามรีดลายสกรีน": "Do not iron on the print",
    "ตากในที่ร่ม": "Dry in the shade",
    "ตากในร่ม": "Dry in the shade",
}

_THAI = re.compile(r"[฀-๿]")


def _is_thai(s: str) -> bool:
    return bool(_THAI.search(s))


def _normalise(text: str) -> str:
    """Lowercase and strip accents from Latin letters (algodón -> algodon).
    Thai vowel and tone marks are kept, because they change the word."""
    text = unicodedata.normalize("NFD", text)
    out: list[str] = []
    for c in text:
        if unicodedata.combining(c) and out and ord(out[-1]) < 0x250:
            continue  # accent on a Latin letter
        out.append(c)
    text = unicodedata.normalize("NFC", "".join(out)).lower()
    # "100 เปอร์เซ็นต์" -> "100 %"
    return re.sub(r"เปอร์เซ็นต์|เปอร์เซ็นท์", "%", text)


def _keyword_to_fabric() -> dict[str, str]:
    return {_normalise(kw): fab for fab, kws in FABRIC_KEYWORDS.items() for kw in kws}


def _thai_matches(t: str, kw_map: dict[str, str]) -> list[tuple[int, int, str]]:
    """Find Thai keywords by substring search, longest first, without overlaps.
    Returns (start, end, keyword)."""
    taken = [False] * len(t)
    found = []
    for kw in sorted((k for k in kw_map if _is_thai(k)), key=len, reverse=True):
        start = t.find(kw)
        while start != -1:
            end = start + len(kw)
            if not any(taken[start:end]):
                found.append((start, end, kw))
                for i in range(start, end):
                    taken[i] = True
            start = t.find(kw, end)
    return found


def parse_fabric(text: str) -> dict[str, int | None]:
    """Return {fabric: percent or None}.
    "60% Cotton 40% Polyester" -> {"cotton": 60, "polyester": 40}
    "คอตตอน 100%"              -> {"cotton": 100}"""
    if not text:
        return {}
    t = _normalise(text)
    kw_map = _keyword_to_fabric()
    result: dict[str, int | None] = {}

    # Tokenise into percentages and fabric keywords, keeping their position.
    tokens: list[tuple[int, str, object]] = []
    for m in re.finditer(r"(\d{1,3})\s*%|([a-z]+)", t):
        if m.group(1):
            tokens.append((m.start(), "pct", int(m.group(1))))
        elif m.group(2) in kw_map:
            tokens.append((m.start(), "fab", (kw_map[m.group(2)], m.group(2))))
    for start, _, kw in _thai_matches(t, kw_map):
        tokens.append((start, "fab", (kw_map[kw], kw)))
    tokens.sort(key=lambda x: x[0])
    tokens = [(kind, val) for _, kind, val in tokens]

    # Labels write either "60% cotton" (percent first) or "cotton 60%" (fabric first).
    # Decide from whichever token type appears first, then pair neighbours.
    if tokens:
        first, second = ("pct", "fab") if tokens[0][0] == "pct" else ("fab", "pct")
        i = 0
        while i < len(tokens) - 1:
            a, b = tokens[i], tokens[i + 1]
            if a[0] == first and b[0] == second:
                pct = a[1] if a[0] == "pct" else b[1]
                fab = (b[1] if b[0] == "fab" else a[1])[0]
                if 0 < pct <= 100 and fab not in result:
                    result[fab] = pct
                i += 2
            else:
                i += 1

    # Keyword without a percentage, e.g. "COTTON" or "ผ้าฝ้าย" alone.
    for kind, val in tokens:
        if kind == "fab":
            fab, kw = val
            if fab not in result and kw not in _NEEDS_PERCENT:
                result[fab] = None
    return result


def main_fabrics(fabrics: dict[str, int | None], threshold: int = 20) -> set[str]:
    """Fabrics that matter for advice: >= threshold %, or no percentage given."""
    return {f for f, p in fabrics.items() if p is None or p >= threshold}


def parse_care_notes(text: str) -> list[str]:
    norm = _normalise(text or "")
    spaced = " ".join(norm.split())
    unspaced = "".join(norm.split())  # OCR often puts random spaces inside Thai text
    notes: list[str] = []
    for phrase, note in CARE_NOTES.items():
        hay = unspaced if _is_thai(phrase) else spaced
        if phrase in hay and note not in notes:
            notes.append(note)
    return notes
