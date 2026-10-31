"""Unit tests for fabric parsing, rules and dataset scripts.

Run:  python -m unittest discover tests     (or: pytest)
No model, GPU or internet needed.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from carescan.fabric import main_fabrics, parse_care_notes, parse_fabric  # noqa: E402
from carescan.preprocess import preprocess  # noqa: E402
from carescan.rules import Detection, build_care_guide, resolve_conflicts  # noqa: E402
from carescan.symbols import class_names  # noqa: E402


def guide(symbols, text="", colour=None, is_new=False):
    return build_care_guide([Detection(s) for s in symbols], parse_fabric(text), colour, is_new,
                            parse_care_notes(text))


class TestSymbols(unittest.TestCase):
    def test_class_list(self):
        names = class_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 25)


class TestFabric(unittest.TestCase):
    def test_percent_first(self):
        self.assertEqual(parse_fabric("60% Cotton 40% Polyester"), {"cotton": 60, "polyester": 40})

    def test_fabric_first(self):
        self.assertEqual(parse_fabric("COTTON 95% ELASTANE 5%"), {"cotton": 95, "elastane": 5})

    def test_multilingual(self):
        self.assertEqual(parse_fabric("100% coton / algodón / cotton"), {"cotton": 100})
        self.assertEqual(parse_fabric("80% LAINE 20% POLYAMIDE"), {"wool": 80, "nylon": 20})

    def test_no_percent(self):
        self.assertEqual(parse_fabric("Pure silk  Made in Italy"), {"silk": None})

    def test_short_keyword_needs_percent(self):
        self.assertEqual(parse_fabric("Made in Berlin"), {})
        self.assertEqual(parse_fabric("LIN 100%"), {"linen": 100})

    def test_main_fabrics(self):
        self.assertEqual(main_fabrics({"cotton": 95, "elastane": 5}), {"cotton"})

    def test_care_notes(self):
        self.assertEqual(parse_care_notes("WASH SEPARATELY. Do not  soak"),
                         ["Wash separately", "Do not soak"])

    def test_empty(self):
        self.assertEqual(parse_fabric(""), {})

    # ---- Thai ----
    def test_thai_fabric_first(self):
        # From the AIIZ label: "คอตตอน 100%"
        self.assertEqual(parse_fabric("AIIZ คอตตอน 100% กลับด้านรีด WWW.AIIZ.COM"), {"cotton": 100})

    def test_thai_percent_first_no_spaces(self):
        self.assertEqual(parse_fabric("60%ฝ้าย40%โพลีเอสเตอร์"), {"cotton": 60, "polyester": 40})

    def test_thai_blend(self):
        self.assertEqual(parse_fabric("คอตตอน 95% สแปนเด็กซ์ 5%"), {"cotton": 95, "elastane": 5})

    def test_thai_word_percent(self):
        self.assertEqual(parse_fabric("ผ้าฝ้าย 100 เปอร์เซ็นต์"), {"cotton": 100})

    def test_thai_no_percent(self):
        self.assertEqual(parse_fabric("ผ้าฝ้าย"), {"cotton": None})

    def test_thai_silk_needs_percent(self):
        self.assertEqual(parse_fabric("สวยไหม"), {})  # question word, not silk
        self.assertEqual(parse_fabric("ไหม 100%"), {"silk": 100})

    def test_thai_and_english_together(self):
        self.assertEqual(parse_fabric("100% COTTON คอตตอน 100%"), {"cotton": 100})

    def test_thai_care_notes(self):
        self.assertEqual(parse_care_notes("กลับด้านรีด"), ["Iron on the reverse side"])
        self.assertEqual(parse_care_notes("ซัก แยก  ห้าม แช่"), ["Wash separately", "Do not soak"])

    def test_regenerated_cellulose(self):
        self.assertEqual(parse_fabric("95% regenerated cellulose fibers +5% spandex"),
                         {"viscose": 95, "elastane": 5})

    def test_latin_accents_still_stripped(self):
        self.assertEqual(parse_fabric("100% algodón"), {"cotton": 100})


class TestConflicts(unittest.TestCase):
    def test_keeps_most_confident_per_group(self):
        kept, dropped = resolve_conflicts([Detection("wash_40", 0.9), Detection("wash_30", 0.6)])
        self.assertEqual([d.name for d in kept], ["wash_40"])
        self.assertEqual(dropped, ["wash_30"])

    def test_natural_drying_not_exclusive(self):
        kept, _ = resolve_conflicts([Detection("dry_line", 0.8), Detection("dry_shade", 0.7)])
        self.assertEqual({d.name for d in kept}, {"dry_line", "dry_shade"})

    def test_dryclean_and_wetclean_together(self):
        kept, _ = resolve_conflicts([Detection("dryclean_P", 0.8), Detection("wetclean_W", 0.7)])
        self.assertEqual(len(kept), 2)

    def test_low_conf_and_unknown_dropped(self):
        kept, dropped = resolve_conflicts([Detection("wash_40", 0.1), Detection("banana", 0.9)])
        self.assertEqual(kept, [])
        self.assertEqual(set(dropped), {"wash_40", "banana"})

    def test_duplicate_same_class(self):
        kept, dropped = resolve_conflicts([Detection("no_bleach", 0.9), Detection("no_bleach", 0.5)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_sorted_by_category(self):
        kept, _ = resolve_conflicts([Detection("no_iron"), Detection("wash_30"), Detection("no_bleach")])
        self.assertEqual([d.name for d in kept], ["wash_30", "no_bleach", "no_iron"])


class TestRules(unittest.TestCase):
    def test_cover_slide_example(self):
        g = guide(["wash_40", "no_bleach", "iron_med", "dryclean_P", "no_tumble"],
                  "100% COTTON", "bright", True)
        rules = g.rules_fired()
        self.assertIn("R03_shrink_hot_wash", rules)
        self.assertIn("R08_new_colour_bleed", rules)
        self.assertEqual(len(g.steps), 5)
        self.assertNotIn("R11_missing_category", rules)

    def test_no_symbols(self):
        self.assertEqual(guide([]).rules_fired(), ["R00_no_symbols"])

    def test_dryclean_only(self):
        self.assertIn("R01_dryclean_only", guide(["no_wash", "dryclean_P"]).rules_fired())

    def test_spot_clean(self):
        self.assertIn("R02_spot_clean", guide(["no_wash", "no_dryclean"]).rules_fired())

    def test_no_shrink_warning_cold(self):
        self.assertNotIn("R03_shrink_hot_wash", guide(["wash_30"], "100% cotton").rules_fired())

    def test_no_shrink_warning_polyester(self):
        self.assertNotIn("R03_shrink_hot_wash", guide(["wash_60"], "100% polyester").rules_fired())

    def test_shrink_in_dryer(self):
        self.assertIn("R04_shrink_dryer", guide(["wash_30", "tumble_normal"], "100% cotton").rules_fired())

    def test_wool(self):
        r = guide(["wash_hand", "tumble_low"], "100% wool").rules_fired()
        self.assertIn("R05_delicate_fibre", r)
        self.assertIn("R06_wool_dryer", r)

    def test_elastane_minor_fibre_still_counts(self):
        self.assertIn("R07_elastane_heat", guide(["wash_60"], "95% cotton 5% elastane").rules_fired())

    def test_colour_bleach(self):
        self.assertIn("R09_colour_bleach", guide(["bleach_any"], colour="dark").rules_fired())
        self.assertNotIn("R09_colour_bleach", guide(["bleach_any"], colour="white").rules_fired())

    def test_synthetic_iron(self):
        self.assertIn("R10_synthetic_iron", guide(["iron_high"], "100% polyester").rules_fired())

    def test_missing_categories(self):
        g = guide(["wash_40"])
        self.assertIn("R11_missing_category", g.rules_fired())
        self.assertEqual(g.missing_categories, ["Bleaching", "Ironing", "Drying"])

    def test_no_dryer_tip(self):
        self.assertTrue(any("hang to dry" in t for t in guide(["no_tumble"]).tips))
        self.assertTrue(any("lay flat" in t for t in guide(["no_tumble"], "100% wool").tips))

    def test_sorting_tip(self):
        self.assertTrue(any("darks, 30°C gentle" in t for t in
                            guide(["wash_30_gentle"], colour="dark").tips))

    def test_label_notes(self):
        self.assertIn("Label says: Wash inside out", guide(["wash_30"], "wash inside out").tips)

    def test_every_symbol_has_instruction(self):
        for n in class_names():
            g = guide([n])
            self.assertEqual(len(g.steps), 1, n)
            self.assertTrue(g.steps[0]["text"], n)


class TestPreprocess(unittest.TestCase):
    def test_finds_and_crops_label(self):
        img = np.full((900, 1200, 3), 60, np.uint8)            # dark background
        cv2.rectangle(img, (300, 200), (900, 700), (235, 235, 235), -1)  # white label
        out, info = preprocess(img)
        self.assertTrue(info["cropped"])
        self.assertLess(abs(out.shape[1] - 600), 20)
        self.assertLess(abs(out.shape[0] - 500), 20)

    def test_resize_and_no_crop(self):
        img = np.random.randint(0, 255, (3000, 2000, 3), np.uint8)
        out, info = preprocess(img, crop=False)
        self.assertEqual(max(out.shape[:2]), 1280)
        self.assertFalse(info["cropped"])


class TestAliases(unittest.TestCase):
    def test_care_label_x5z3u_all_46_classes_mapped(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from merge_datasets import load_alias_map, norm
        amap = load_alias_map(ROOT / "configs" / "class_aliases.yaml")
        names = ['30C', '40C', '50C', '60C', '70C', '95C', 'Bleach', 'Chlorine_bleach', 'DN_Bleach',
                 'DN_dry', 'DN_dry_clean', 'DN_iron', 'DN_steam', 'DN_tumble_dry', 'DN_wash',
                 'DN_wiring', 'DN_wring', 'Drip_dry', 'Dry_clean', 'Dry_clean_any_solvent',
                 'Dry_clean_any_solvent_execpt_tricholoroethylene', 'Dry_clean_low_heat',
                 'Dry_clean_no_steam', 'Dry_clean_petrol_only', 'Dry_clean_reduced_moisture',
                 'Dry_clean_short_cycle', 'Dry_flat', 'Wet_clean', 'Wring', 'hand_wash', 'iron',
                 'iron_high', 'iron_low', 'iron_medium', 'line_dry', 'line_dry_in_shade',
                 'machine_wash_delicate', 'machine_wash_normal', 'machine_wash_permanent_press',
                 'natural_dry', 'non_chlorine_bleach', 'shade_dry', 'tumble_dry_low',
                 'tumble_dry_medium', 'tumble_dry_no_heat', 'tumble_dry_normal']
        self.assertEqual([n for n in names if norm(n) not in amap], [])
        self.assertEqual(amap[norm("Dry_clean_petrol_only")], "dryclean_F")
        self.assertEqual(amap[norm("iron_medium")], "iron_med")


class TestNoLeakage(unittest.TestCase):
    def test_augmented_copies_stay_in_one_split(self):
        import random
        sys.path.insert(0, str(ROOT / "scripts"))
        from split_dataset import group_key, split_list
        files = []
        for i in range(60):  # 60 originals, 3 Roboflow copies each, spread over export folders
            for j, folder in enumerate(["train", "train", "valid"]):
                files.append(Path(f"pub__ds__{folder}__img{i}_jpg.rf.{i}abc{j}.jpg"))
        self.assertEqual(group_key(files[0]), group_key(files[2]))
        tr, va, te = split_list(files, (0.8, 0.15, 0.05), random.Random(0))
        keys = [{group_key(p) for p in s} for s in (tr, va, te)]
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        self.assertEqual(len(tr) + len(va) + len(te), 180)


class TestDatasetScripts(unittest.TestCase):
    """Build two fake Roboflow exports, merge them, split them."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _fake_export(self, name, names, n, lines):
        d = self.tmp / name
        for split in ("train", "valid"):
            (d / split / "images").mkdir(parents=True)
            (d / split / "labels").mkdir(parents=True)
            for i in range(n):
                cv2.imwrite(str(d / split / "images" / f"img{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
                (d / split / "labels" / f"img{i}.txt").write_text(lines)
        (d / "data.yaml").write_text(yaml.safe_dump({"names": names}))
        return d

    def _run(self, *args):
        return subprocess.run([sys.executable, *args], capture_output=True, text=True, check=True).stdout

    def test_merge_and_split(self):
        # public set: "40C" -> wash_40, "Do not bleach" -> no_bleach, "steam" -> drop
        pub = self._fake_export("pubA", ["40C", "Do not bleach", "steam"], 20,
                                "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.7 0.7 0.1 0.1")
        # set with an unknown class -> images skipped and reported
        bad = self._fake_export("pubB", ["mystery_symbol"], 3, "0 0.5 0.5 0.2 0.2")
        # own set, polygon (segmentation) labels using our own names
        own = self._fake_export("own", ["wash_hand", "dry_flat"], 10,
                                "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n1 0.5 0.5 0.6 0.6")
        # set with a class mapped to "skip" -> those images left out entirely
        skp = self._fake_export("pubC", ["machine_wash_normal", "40C"], 4, "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1")
        merged, final = self.tmp / "merged", self.tmp / "final"
        out = self._run(str(ROOT / "scripts/merge_datasets.py"), "--public", str(pub), str(bad), str(skp),
                        "--own", str(own), "--out", str(merged))
        self.assertIn("Merged 60 images", out)
        self.assertIn("pubB:mystery_symbol", out)
        self.assertIn("8 images left out", out)
        self.assertFalse(list((merged / "images").glob("pub__pubc__*")))

        ids = {n: i for i, n in enumerate(class_names())}
        lbl = next((merged / "labels").glob("pub__puba__*.txt")).read_text().splitlines()
        self.assertEqual([int(l.split()[0]) for l in lbl], [ids["wash_40"], ids["no_bleach"]])
        own_lbl = next((merged / "labels").glob("own__*.txt")).read_text().splitlines()
        cx, cy, w, h = map(float, own_lbl[0].split()[1:])
        self.assertAlmostEqual(cx, 0.2)
        self.assertAlmostEqual(w, 0.2)

        out = self._run(str(ROOT / "scripts/split_dataset.py"), "--src", str(merged), "--out", str(final))
        counts = {s: len(list((final / s / "images").iterdir())) for s in ("train", "val", "test")}
        self.assertEqual(sum(counts.values()), 60)
        own_test = len(list((final / "test" / "images").glob("own__*")))
        # 20 own files = 10 originals x 2 copies (same name in train/ and valid/);
        # 10 groups -> 4 train / 2 val / 4 test groups -> 8 test images, copies together
        self.assertEqual(own_test, 8)
        data = yaml.safe_load((final / "data.yaml").read_text())
        self.assertEqual(list(data["names"].values()), class_names())

        out = self._run(str(ROOT / "scripts/check_balance.py"), "--data", str(final), "--min", "5")
        self.assertIn("wash_40", out)
        self.assertIn("MISSING", out)


if __name__ == "__main__":
    unittest.main()
