"""
Evaluate model on the held-out test set.
Run after training to get honest accuracy on unseen images.

Usage: python scripts/evaluate_test.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ─────────────────────────────────────────────
WEIGHTS   = os.getenv("WEIGHTS", "runs/train/coc_capital_v9_final/weights/best.pt")
TEST_IMGS = os.getenv("TEST_IMGS", "dataset/test/images")
DATA_YAML = os.getenv("DATA_YAML", "dataset/data_fixed.yaml")
IMG_SIZE  = 640
CONF      = 0.25   # confidence threshold
IOU       = 0.6    # NMS IoU threshold
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Test Set Evaluation")
    print(f"  Weights: {WEIGHTS}")
    print("=" * 50)

    test_path = Path(TEST_IMGS)
    if not test_path.exists() or not any(test_path.iterdir()):
        print(f"No test images found at {TEST_IMGS}")
        print("Run scripts/make_test_split.py first.")
        return

    test_count = len(list(test_path.glob("*.*")))
    print(f"Test images: {test_count}")

    model = YOLO(WEIGHTS)

    results = model.val(
        data    = DATA_YAML,
        split   = "test",
        imgsz   = IMG_SIZE,
        conf    = CONF,
        iou     = IOU,
        device  = 0,
        plots   = True,
        save_json = False,
    )

    print("\n" + "=" * 50)
    print("  TEST RESULTS (unseen images)")
    print("=" * 50)
    print(f"  mAP50:    {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    print(f"  Precision: {results.box.mp:.3f}")
    print(f"  Recall:    {results.box.mr:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
