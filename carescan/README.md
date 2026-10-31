# CareScan

AI recognition of clothing care labels: one photo in, clear washing advice out.
CSC472 Sec 141 group project.

```
photo ─► OpenCV preprocess ─► YOLOv8 symbol detector ─► EasyOCR (fabric text) ─► rule base ─► care guide
          crop, deskew, CLAHE     28 ISO 3758 classes     "100% cotton" (TH+EN)     IF-THEN rules
```

## Project layout

| Path | What it is |
|---|---|
| `configs/symbols.yaml` | **The class list** (28 symbols), with instructions. Order = YOLO class ids. |
| `configs/class_aliases.yaml` | Maps class names from public datasets to our names |
| `docs/labelling_guide.md` | How to photograph and box symbols. **Everyone reads this first.** |
| `scripts/merge_datasets.py` | Merges Roboflow exports into one pool with our class ids |
| `scripts/split_dataset.py` | 70/15/15 split (own photos mostly go to test) + `data.yaml` |
| `scripts/check_balance.py` | Boxes per class, flags classes that need more data |
| `scripts/evaluate_pipeline.py` | End-to-end test on own photos vs `ground_truth.csv` |
| `notebooks/train_yolo_colab.ipynb` | Colab notebook: download → train → evaluate → export |
| `carescan/` | The pipeline: `preprocess`, `detector`, `ocr`, `fabric`, `rules`, `pipeline` |
| `app/streamlit_app.py` | Web app (scan mode + demo mode) |
| `tests/` | 43 unit tests for the rules, fabric parser, preprocessing and dataset scripts |

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover tests          # check everything works
streamlit run app/streamlit_app.py         # demo mode works before the model exists
```

**Demo mode** lets you pick symbols by hand and see the care guide, so the rules and the UI
can be built and tested while the dataset is still being labelled.

## Workflow

1. **Data:** follow `docs/labelling_guide.md`. Take 100+ own photos, label them in Roboflow, and find
   public laundry-symbol datasets on Roboflow Universe.
2. **Aliases:** check every class name in the public datasets and add them to `configs/class_aliases.yaml`.
3. **Train:** upload this folder to Google Drive and run `notebooks/train_yolo_colab.ipynb` on a T4 GPU.
   The notebook merges, splits and trains, then evaluates on the test set.
4. **Improve:** use the per-class AP and the confusion matrix to find weak classes. Add photos of those
   classes and retrain. Log each run in the notebook's experiment table.
5. **App:** copy `best.pt` to `models/` and run the app in scan mode.
6. **Evaluate end to end:** fill in `data/ground_truth.csv` (see the template) and run
   `scripts/evaluate_pipeline.py`.

## Targets (from the proposal)

| Metric | Target | Measured by |
|---|---|---|
| mAP@0.5 symbol detection | ≥ 85% | notebook section 5 |
| Photo → care guide time | < 5 s | `evaluate_pipeline.py` |
| Correct advice in user tests | ≥ 90% | user test with 5–10 people |

## Rule base

`carescan/rules.py` holds the knowledge base. Each rule has an id, shown in the app's
"Rules that fired" panel and checked in the tests:

| Rule | IF | THEN |
|---|---|---|
| R00 | no symbols detected | ask to retake photo |
| R01 | no_wash + dry-clean allowed | professional cleaning only |
| R02 | no_wash + no_dryclean | spot clean only |
| R03 | cotton/linen/wool/viscose + wash ≥ 40°C | shrink risk |
| R04 | shrink-prone fabric + tumble_normal | shrink risk in dryer |
| R05 | wool or silk | delicate handling advice |
| R06 | wool + tumble allowed | felting warning |
| R07 | elastane + high heat | stretch fibres weaken |
| R08 | dark/bright colour + new item | wash separately (dye bleed) |
| R09 | coloured + bleach allowed | fading warning |
| R10 | synthetic + iron_high | melting risk |
| R11 | category missing (wash/bleach/dry/iron) | tell user what wasn't found |

It also resolves conflicts: only one symbol is kept per exclusive group (e.g. two washing symbols →
keep the most confident one).

## Known limitations

- There is no `wash_50` class; label 50°C / 3-dot washtubs as `wash_40`.
- OCR reads Thai + English. Fabric words are also recognised in French, Spanish, German, Italian
  and Dutch when the OCR picks them up. Other scripts (Chinese, Korean) need a different EasyOCR
  model, set in `ocr.py`.
- Korean (KS) labels use different-looking symbols for bleach and ironing; tag those photos
  `korean` in Roboflow so you can check their accuracy separately.
- EasyOCR downloads its Thai and English models (~100 MB) the first time the app scans a label.
- Garment colour comes from the user, not from the image.
