"""
Creates a test set by pulling 2 images per district from train
and 2 images per district from valid (8+8=16 total).
Run this BEFORE training so test images are never seen by the model.

Usage: python scripts/make_test_split.py
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path("dataset")
PER_DISTRICT_FROM_TRAIN = 2
PER_DISTRICT_FROM_VAL   = 2
RANDOM_SEED = 42

# ─────────────────────────────────────────────

def get_district(filename: str) -> str:
    """Extract district name from Roboflow filename e.g. capital_peak_469_png.rf... → capital_peak"""
    # filenames are like: capital_peak_469_png.rf.xxxxx.jpg
    # district names all use underscores, and are followed by a number
    parts = filename.split("_")
    district_parts = []
    for part in parts:
        if part.isdigit():
            break
        district_parts.append(part)
    return "_".join(district_parts)


def collect_images_by_split(dataset_root: Path) -> tuple[dict, dict]:
    """Collect images from train and valid separately, grouped by district."""
    train_by_district: dict[str, list[Path]] = defaultdict(list)
    val_by_district:   dict[str, list[Path]] = defaultdict(list)

    for img in sorted((dataset_root / "train" / "images").glob("*.*")):
        d = get_district(img.name)
        if d:
            train_by_district[d].append(img)

    for img in sorted((dataset_root / "valid" / "images").glob("*.*")):
        d = get_district(img.name)
        if d:
            val_by_district[d].append(img)

    return dict(train_by_district), dict(val_by_district)


def move_to_test(img_path: Path, dataset_root: Path):
    """Move image and its label to the test folder."""
    # figure out which split it came from
    split = img_path.parent.parent.name  # "train" or "valid"

    test_img_dir = dataset_root / "test" / "images"
    test_lbl_dir = dataset_root / "test" / "labels"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    # move image
    dest_img = test_img_dir / img_path.name
    shutil.move(str(img_path), str(dest_img))

    # move corresponding label
    lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(test_lbl_dir / lbl_path.name))
    else:
        print(f"  ⚠ No label found for {img_path.name}")


def main():
    random.seed(RANDOM_SEED)

    print("Collecting images by district...")
    train_by_district, val_by_district = collect_images_by_split(DATASET_ROOT)

    all_districts = sorted(set(list(train_by_district.keys()) + list(val_by_district.keys())))
    print(f"\nFound {len(all_districts)} districts:")
    for d in all_districts:
        print(f"  {d}: {len(train_by_district.get(d, []))} train, {len(val_by_district.get(d, []))} val")

    # check test folder — skip if already populated
    test_img_dir = DATASET_ROOT / "test" / "images"
    if test_img_dir.exists() and any(test_img_dir.iterdir()):
        existing = len(list(test_img_dir.iterdir()))
        print(f"\nTest folder already has {existing} images. Delete it first to re-split.")
        return

    print(f"\nMoving {PER_DISTRICT_FROM_TRAIN} from train + {PER_DISTRICT_FROM_VAL} from val per district...")
    total_moved = 0

    for district in all_districts:
        train_imgs = train_by_district.get(district, [])
        val_imgs   = val_by_district.get(district, [])

        # pick from train
        n_train = min(PER_DISTRICT_FROM_TRAIN, len(train_imgs))
        selected_train = random.sample(train_imgs, n_train)

        # pick from val
        n_val = min(PER_DISTRICT_FROM_VAL, len(val_imgs))
        selected_val = random.sample(val_imgs, n_val)

        for img in selected_train + selected_val:
            print(f"  → {img.name}")
            move_to_test(img, DATASET_ROOT)
            total_moved += 1

    # recount
    train_count = len(list((DATASET_ROOT / "train" / "images").glob("*.*")))
    valid_count = len(list((DATASET_ROOT / "valid" / "images").glob("*.*")))
    test_count  = len(list(test_img_dir.glob("*.*")))

    print(f"\nDone!")
    print(f"  Train: {train_count} images")
    print(f"  Valid: {valid_count} images")
    print(f"  Test:  {test_count} images  ← {total_moved} moved")


if __name__ == "__main__":
    main()
