"""CareScan web app.

Run:  streamlit run app/streamlit_app.py

Two modes:
  - Scan: uses the trained model at models/best.pt (+ EasyOCR)
  - Demo: pick symbols by hand to test the rules before the model is trained
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from carescan.fabric import parse_fabric  # noqa: E402
from carescan.pipeline import guide_from_symbols  # noqa: E402
from carescan.rules import Detection, build_care_guide  # noqa: E402
from carescan.symbols import CATEGORY_LABELS, by_name, class_names  # noqa: E402

MODEL = ROOT / "models" / "best.pt"
COLOURS = {"Not sure": None, "White": "white", "Light": "light", "Dark": "dark", "Bright": "bright"}
LEVEL = {"danger": st.error, "warning": st.warning, "info": st.info}

st.set_page_config(page_title="CareScan", page_icon="🧺", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def get_pipeline(conf: float, use_ocr: bool):
    from carescan.pipeline import CareScanPipeline
    return CareScanPipeline(str(MODEL), conf=conf, use_ocr=use_ocr)


def show_guide(guide, fabrics: dict):
    if fabrics:
        st.markdown("**Fabric:** " + ", ".join(
            f"{p}% {f}" if p else f for f, p in fabrics.items()))

    for w in guide.warnings:
        LEVEL[w.level](f"**{w.title}** — {w.text}")

    st.subheader("Care guide")
    for s in guide.steps:
        st.markdown(f"- **{CATEGORY_LABELS[s['category']]}:** {s['text']}  "
                    f"<span style='color:gray;font-size:0.8em'>`{s['symbol']}`</span>",
                    unsafe_allow_html=True)
    if guide.tips:
        st.subheader("Tips")
        for t in guide.tips:
            st.markdown(f"- {t}")
    if guide.dropped:
        with st.expander("Detections removed by conflict checks"):
            st.write(", ".join(guide.dropped))
    with st.expander("Rules that fired (for debugging)"):
        st.write(guide.rules_fired() or "none")


def save_to_wardrobe(item_name: str, symbols: list[str], fabrics: dict, colour):
    st.session_state.setdefault("wardrobe", []).append(
        {"name": item_name, "symbols": symbols, "fabrics": fabrics, "colour": colour})


# ---------------- Sidebar ----------------
st.sidebar.title("🧺 CareScan")
has_model = MODEL.exists()
mode = st.sidebar.radio("Mode", ["Scan label", "Demo (pick symbols)"],
                        index=0 if has_model else 1)
if mode == "Scan label" and not has_model:
    st.sidebar.warning("No trained model yet. Put best.pt in models/ (from the Colab notebook).")
conf = st.sidebar.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)
use_ocr = st.sidebar.checkbox("Read label text (OCR)", value=True)
ocr_size = st.sidebar.select_slider(
    "OCR image size (px)", options=[640, 800, 960, 1120, 1280], value=1280,
    help="Smaller = faster OCR, but small print (especially Thai) may be missed.")

st.sidebar.markdown("---")
st.sidebar.subheader("My wardrobe")
for i, item in enumerate(st.session_state.get("wardrobe", [])):
    if st.sidebar.button(f"👕 {item['name']}", key=f"w{i}"):
        st.session_state["reopen"] = item
if not st.session_state.get("wardrobe"):
    st.sidebar.caption("Saved items appear here.")

# ---------------- Main ----------------
st.title("CareScan")
st.caption("Scan a clothing care label to get clear washing advice.")

c1, c2 = st.columns(2)
colour = COLOURS[c1.selectbox("Garment colour", list(COLOURS))]
is_new = c2.checkbox("New item (first few washes)")

# Re-open a saved wardrobe item without rescanning
if "reopen" in st.session_state:
    item = st.session_state.pop("reopen")
    st.header(item["name"])
    g = build_care_guide([Detection(s) for s in item["symbols"]], item["fabrics"], item["colour"])
    show_guide(g, item["fabrics"])
    st.stop()

if mode == "Demo (pick symbols)":
    st.info("Demo mode: choose the symbols on a label to test the rule base without the model.")
    names = class_names()
    known = by_name()
    picked = st.multiselect("Symbols on the label", names,
                            default=["wash_40", "no_bleach", "iron_med", "dryclean_P", "no_tumble"],
                            format_func=lambda n: f"{n} — {known[n]['meaning']}")
    text = st.text_input("Text on the label (as OCR would read it)", "100% COTTON  Wash separately")
    if picked:
        guide = guide_from_symbols(picked, text, colour, is_new)
        show_guide(guide, parse_fabric(text))
    st.stop()

# Scan mode
src = st.radio("Photo source", ["Upload", "Camera"], horizontal=True)
file = st.camera_input("Hold the label inside the frame") if src == "Camera" \
    else st.file_uploader("Upload a label photo", type=["jpg", "jpeg", "png", "webp"])

if file and has_model:
    from carescan.detector import draw_detections

    pipe = get_pipeline(conf, use_ocr)

    # Cache per photo: Streamlit reruns this whole script on every click (colour,
    # Save button...), and the slow OCR must not run again each time.
    key = hashlib.md5(file.getvalue()).hexdigest() + f"_{conf}"
    scan = st.session_state.get("scan")
    cached = bool(scan and scan["key"] == key)
    c1, c2 = st.columns([3, 1])
    if c2.button("🔄 Scan again"):
        cached = False
    if not cached:
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        proc, dets, info = pipe.detect_stage(img)
        scan = {"key": key, "img": proc, "dets": dets, "info": info,
                "first_ms": round((time.perf_counter() - t0) * 1000), "ocr": {}}
        st.session_state["scan"] = scan
    c1.caption("♻️ Showing the saved result for this photo — only the advice was updated."
               if cached else "🆕 New scan of this photo.")

    def render(text: str | None):
        """text=None means OCR is still running."""
        guide, fabrics = pipe.build(scan["dets"], text or "", scan["info"], colour, is_new)
        st.image(cv2.cvtColor(draw_detections(scan["img"], scan["dets"]), cv2.COLOR_BGR2RGB),
                 caption=f"Symbols found: {len(guide.symbols)}  ·  care guide in {scan['first_ms']} ms")
        if use_ocr:
            if text is None:
                st.info("⏳ Reading the label text (fabric, notes)… the guide updates when done.")
            else:
                st.caption(f"OCR ({scan['ocr'][ocr_size][1]} ms): {text or '(no text read)'}")
        show_guide(guide, fabrics)
        return guide, fabrics

    area = st.empty()
    ocr_done = not use_ocr or ocr_size in scan["ocr"]
    if not ocr_done:
        with area.container():
            render(None)                      # symbols + care guide straight away
        scan["ocr"][ocr_size] = pipe.ocr_stage(scan["img"], ocr_size)
    text = scan["ocr"][ocr_size][0] if use_ocr else ""
    with area.container():
        guide, fabrics = render(text)          # final guide with fabric + label notes

    with st.form("save"):
        name = st.text_input("Save to My Wardrobe as", "T-shirt")
        if st.form_submit_button("Save"):
            save_to_wardrobe(name, guide.symbols, fabrics, colour)
            st.success(f"Saved {name}")
    with st.expander("Timings"):
        timings = dict(scan["info"]["timings_ms"])
        timings["care_guide_shown_after"] = scan["first_ms"]
        if use_ocr:
            timings["ocr"] = scan["ocr"][ocr_size][1]
            timings["full_guide_after"] = scan["first_ms"] + timings["ocr"]
        st.json(timings)