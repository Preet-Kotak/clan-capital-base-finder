"""
Step 1 of pipeline: Process all local screenshots → layouts.json

Loops through all images in data/screenshots/by_district/
Runs YOLO extractor on each, saves results to data/layouts/layouts.json

Format:
{
    "469": {
        "district": 0,
        "district_name": "capital_peak",
        "buildings": [{"type": "cannon", "x": 0.17, "y": -0.35, "conf": 0.90}],
        "anchor_found": true,
        "total_detections": 49
    },
    ...
}

Usage: python scripts/extract_all.py
Resume safe — skips already processed IDs.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extractor import Extractor

# ─────────────────────────────────────────────
SCREENSHOTS_DIR = Path("data/screenshots/by_district")
OUTPUT_FILE     = Path("data/layouts/layouts.json")

DISTRICT_MAP = {
    "capital_peak":     0,
    "barbarian_camp":   1,
    "wizard_valley":    2,
    "balloon_lagoon":   3,
    "builders_workshop":4,
    "dragon_cliffs":    5,
    "golem_quarry":     6,
    "skeleton_park":    7,
    "goblin_mines":     8,
}
# ─────────────────────────────────────────────


def get_base_id(filename: str, district: str) -> str | None:
    """Extract base ID from filename. e.g. capital_peak_469.png → 469"""
    stem   = Path(filename).stem
    prefix = district + "_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return None


def main():
    print("=" * 55)
    print("  Extract All — YOLO → layouts.json")
    print("=" * 55)

    # load existing results to support resume
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            layouts = json.load(f)
        print(f"Loaded {len(layouts)} existing layouts (resume mode)")
    else:
        layouts = {}

    # collect all images
    all_images = []
    for district, district_num in DISTRICT_MAP.items():
        district_dir = SCREENSHOTS_DIR / district
        if not district_dir.exists():
            print(f"  ⚠ Folder not found: {district_dir}")
            continue
        for img in sorted(district_dir.glob("*.*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                base_id = get_base_id(img.name, district)
                if base_id:
                    all_images.append((base_id, district, district_num, img))

    total     = len(all_images)
    already   = sum(1 for base_id, _, _, _ in all_images if base_id in layouts)
    remaining = total - already

    print(f"\nTotal images:     {total}")
    print(f"Already done:     {already}")
    print(f"To process:       {remaining}")

    if remaining == 0:
        print("\nAll images already processed!")
        return

    # load model once
    print("\nLoading YOLO model...")
    extractor = Extractor()
    print("Model loaded. Starting extraction...\n")

    success = 0
    failed  = 0
    start   = time.time()

    for i, (base_id, district, district_num, img_path) in enumerate(all_images, 1):
        # skip already processed
        if base_id in layouts:
            continue

        try:
            result = extractor.extract(img_path, district=district_num)

            layouts[base_id] = {
                "district":        district_num,
                "district_name":   district,
                "buildings":       result["buildings"],
                "anchor_found":    result["anchor_found"],
                "total_detections":result["total_detections"],
                "conf_used":       result["conf_used"],
                "runs":            result["runs"],
            }

            success += 1

            # progress every 25 images
            if success % 25 == 0:
                elapsed  = time.time() - start
                per_img  = elapsed / success
                eta_secs = per_img * (remaining - success)
                print(f"  [{success}/{remaining}] {district}/{img_path.name}"
                      f"  | {result['total_detections']} buildings"
                      f"  | anchor={'✓' if result['anchor_found'] else '✗'}"
                      f"  | ETA: {eta_secs/60:.1f}m")

                # save checkpoint every 25 images
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(layouts, f, indent=2)

        except Exception as e:
            print(f"  ✗ Failed {base_id} ({img_path.name}): {e}")
            failed += 1

    # final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(layouts, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'=' * 55}")
    print(f"Done!")
    print(f"  Processed:  {success}")
    print(f"  Failed:     {failed}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Saved to:   {OUTPUT_FILE.resolve()}")
    print(f"{'=' * 55}")

    # summary stats
    no_anchor = sum(1 for v in layouts.values() if not v["anchor_found"])
    print(f"\nNo anchor found in: {no_anchor}/{len(layouts)} bases")
    if no_anchor > 0:
        print("These bases used image center as reference.")


if __name__ == "__main__":
    main()
