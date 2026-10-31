# CareScan labelling guide

Everyone labels the same way, or the model learns inconsistent boxes.
Read this before you start, and keep `configs/symbols.yaml` open; it lists every
class with a description of what the symbol looks like.

## 1. Taking photos (own dataset)

- **Target:** 100+ labels in total, about 25 per person. More is better.
- Use different clothes: T-shirts, jeans, wool, sportswear, underwear, towels, bedding.
- Vary the conditions on purpose: different angles (straight, tilted, slightly
  rotated), indoor and daylight, some slightly blurry, curved or creased labels.
- Fill most of the frame with the label, the way a user would.
- **Hunt for rare symbols:** hand wash, wool items (dry flat), silk (dry-clean F),
  and anything with W, F or "dry in shade". These will be short in public datasets.
- File names: `<initials>_<number>.jpg`, e.g. `akm_001.jpg`, `kst_014.jpg`.
- Also write each label's **fabric text** (e.g. "100% cotton") in the shared
  spreadsheet `ground_truth.csv`; that sheet is used to test OCR and rules later.

## 2. Labelling in Roboflow

1. Create one Roboflow project, **Object Detection**, and invite all four members.
2. Add the classes **exactly** as written in `configs/symbols.yaml` (copy and paste
   the names; spelling and capitals matter, e.g. `dryclean_P`).
3. Draw one box per symbol:
   - The box is **tight** around the symbol outline, with a small margin of 1–2 px.
   - Include the dots, numbers, bars and cross inside the box; they are part of the symbol.
   - The bars under a washtub belong to the washtub box (`wash_30_gentle`).
   - Do **not** box text, brand logos or size labels.
   - A symbol that is cut off at the edge or unreadable: **don't box it**, and mark
     the image with the tag `partial`.
4. If you are unsure which class a symbol is, tag the image `check` and ask the group.
   Don't guess, because a wrong label is worse than a missing image.
5. Label every symbol you can see. An unboxed symbol teaches the model that it is
   background.

## 3. Common confusions

| Looks like | Check | Class |
|---|---|---|
| Washtub with dots instead of numbers | 1 dot = 30, 2 = 40, 4 = 60, 6 = 95 | `wash_30` / `wash_40` / `wash_60` / `wash_95` |
| Washtub with "50" or 3 dots | no 50 class; washing cooler is safe | `wash_40` |
| Washtub with 1 or 2 bars under it | gentle cycle | `wash_30_gentle` / `wash_40_gentle` |
| Square + circle + dots | dryer heat: 1 dot low, 2 dots normal | `tumble_low` / `tumble_normal` |
| Square + line at top (curved) | hang | `dry_line` |
| Square + 3 vertical lines | drip | `dry_drip` |
| Square + 1 horizontal line | flat | `dry_flat` |
| Diagonal lines in the top-left corner (alone or with another line) | shade: **one box around the whole square** | `dry_shade` |
| Empty square, or square with an X | not one of our classes | don't box |
| Circle with P / F / W | letter decides | `dryclean_P` / `dryclean_F` / `wetclean_W` |
| Bar under the circle | ignore the bar, use the letter | same as above |

Symbols outside our list (e.g. steam iron, "do not wring") are not boxed.

## 4. Exporting

- Export format: **YOLOv8**.
- In Roboflow, do **not** use horizontal or vertical flip augmentation, because
  flipped symbols are wrong. Do augmentation in the training notebook instead.
- Download the export zip and pass the folder to `scripts/merge_datasets.py --own`.

## 5. Splitting the work

| Member | Task |
|---|---|
| A | Download and check public datasets, fill `class_aliases.yaml` |
| B | Own photos + labelling |
| C | Own photos + labelling, `ground_truth.csv` |
| D | Review 10% of everyone's boxes (consistency check) |

Everyone takes photos; rotate the reviewer role each week.
