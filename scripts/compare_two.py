"""
Script B: Compare 2 base screenshots and show match % with visual diff.

Matched buildings   → grayed out (blends in)
Unmatched buildings → full brightness + red dot + label
Overall image       → slightly grayed

Usage:
    python scripts/compare_two.py --image1 path/a.jpg --image2 path/b.jpg
    python scripts/compare_two.py --image1 path/a.jpg --image2 path/b.jpg --district 3 --fuzzy
"""

import sys
import cv2
import argparse
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extractor import Extractor, DISTRICT_MAP, ANCHOR_CLASSES
from src.matcher import match_layouts

OUTPUT_DIR   = Path("runs/compare")
WEIGHTS      = os.getenv("WEIGHTS", "runs/train/coc_capital_v9_final/weights/best.pt")
GRAY_FACTOR  = 0.50
DOT_RADIUS   = 10
DOT_COLOR    = (0, 0, 255)
DOT_OUTLINE  = (255, 255, 255)


def get_pixel_boxes(img_path: Path, conf: float = 0.25) -> list[dict]:
    """
    Run YOLO and return pixel bounding boxes with normalized center coords.
    Used to map normalized layout coords back to pixel positions.
    """
    model   = YOLO(WEIGHTS)
    preds   = model.predict(source=str(img_path), conf=conf, iou=0.5, verbose=False)
    result  = preds[0]
    names   = result.names
    boxes   = result.boxes

    # find anchor
    anchor_x, anchor_y = 0.5, 0.5
    for box in boxes:
        if names[int(box.cls[0])] in ANCHOR_CLASSES:
            anchor_x = float(box.xywhn[0][0])
            anchor_y = float(box.xywhn[0][1])
            break

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = names[int(box.cls[0])]
        x_center = float(box.xywhn[0][0])
        y_center = float(box.xywhn[0][1])

        detections.append({
            "type":  cls_name,
            "x":     round(x_center - anchor_x, 4),
            "y":     round(y_center - anchor_y, 4),
            "x1":    x1, "y1": y1,
            "x2":    x2, "y2": y2,
            "cx":    (x1 + x2) // 2,
            "cy":    (y1 + y2) // 2,
        })
    return detections


def find_unmatched_pixels(
    pixel_boxes: list[dict],
    unmatched_layout: list[dict],
    tolerance: float = 0.05,
) -> list[dict]:
    """
    Match unmatched layout entries (x, y, type) to pixel boxes.
    Returns list of pixel boxes that correspond to unmatched buildings.
    """
    unmatched_pixels = []
    used = set()

    for um in unmatched_layout:
        best_idx  = None
        best_dist = float("inf")

        for i, pb in enumerate(pixel_boxes):
            if i in used:
                continue
            if pb["type"] != um["type"]:
                continue
            dist = ((pb["x"] - um["x"])**2 + (pb["y"] - um["y"])**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        if best_idx is not None and best_dist < tolerance:
            unmatched_pixels.append(pixel_boxes[best_idx])
            used.add(best_idx)

    return unmatched_pixels


def draw_comparison(
    img_path: Path,
    pixel_boxes: list[dict],
    unmatched_pixels: list[dict],
    label: str,
    out_path: Path,
):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not read {img_path}")
        return

    # build set of unmatched pixel coords for fast lookup
    unmatched_coords = {(pb["cx"], pb["cy"]) for pb in unmatched_pixels}

    # gray out entire image
    result = (img.astype(np.float32) * GRAY_FACTOR).astype(np.uint8)

    for pb in pixel_boxes:
        x1, y1, x2, y2 = pb["x1"], pb["y1"], pb["x2"], pb["y2"]
        cx, cy          = pb["cx"], pb["cy"]
        cls_name        = pb["type"]

        if (cx, cy) in unmatched_coords:
            # unmatched → restore brightness + red dot + label
            result[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            cv2.circle(result, (cx, cy), DOT_RADIUS + 2, DOT_OUTLINE, -1)
            cv2.circle(result, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)
            text = cls_name
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.putText(result, text, (cx - tw//2, cy + DOT_RADIUS + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # matched → stays grayed, no marking

    # title
    cv2.putText(result, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(result, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1",   required=True)
    parser.add_argument("--image2",   required=True)
    parser.add_argument("--district", required=False, type=int, default=None,
                        help="District 0-8 (optional, enables calibration)")
    parser.add_argument("--fuzzy",    action="store_true",
                        help="Cannon/spear count as same type")
    args = parser.parse_args()

    district_name = DISTRICT_MAP.get(args.district, "unknown") if args.district is not None else "auto"
    mode          = "fuzzy" if args.fuzzy else "strict"

    print(f"\nComparing 2 bases")
    print(f"District: {district_name}")
    print(f"Mode:     {mode}")

    extractor = Extractor()

    # extract layouts (normalized coords for matching)
    print(f"\nExtracting base 1: {Path(args.image1).name}")
    r1 = extractor.extract(args.image1, district=args.district)
    print(f"  Detections: {r1['total_detections']}  runs: {r1['runs']}")

    print(f"Extracting base 2: {Path(args.image2).name}")
    r2 = extractor.extract(args.image2, district=args.district)
    print(f"  Detections: {r2['total_detections']}  runs: {r2['runs']}")

    # match
    print(f"\nRunning Hungarian matching...")
    result = match_layouts(r1["buildings"], r2["buildings"], fuzzy=args.fuzzy)

    print(f"\n{'=' * 50}")
    print(f"  Match:     {result['match_pct']}%")
    print(f"  Matched:   {result['matched']} / {result['total']} buildings")
    print(f"  Unmatched in base 1: {len(result['unmatched_a'])}")
    print(f"  Unmatched in base 2: {len(result['unmatched_b'])}")
    print(f"{'=' * 50}")

    # get pixel boxes for visualization
    print(f"\nGetting pixel positions for visualization...")
    px1 = get_pixel_boxes(Path(args.image1), conf=r1["conf_used"])
    px2 = get_pixel_boxes(Path(args.image2), conf=r2["conf_used"])

    # find exact unmatched pixel boxes
    unmatched_px1 = find_unmatched_pixels(px1, result["unmatched_a"])
    unmatched_px2 = find_unmatched_pixels(px2, result["unmatched_b"])

    print(f"  Pinpointed {len(unmatched_px1)} unmatched in base 1")
    print(f"  Pinpointed {len(unmatched_px2)} unmatched in base 2")

    # draw
    stem1 = Path(args.image1).stem
    stem2 = Path(args.image2).stem

    print(f"\nGenerating visualizations...")
    draw_comparison(
        Path(args.image1), px1, unmatched_px1,
        f"Base 1 — {result['match_pct']}% match",
        OUTPUT_DIR / f"{stem1}_vs_{stem2}_base1.jpg",
    )
    draw_comparison(
        Path(args.image2), px2, unmatched_px2,
        f"Base 2 — {result['match_pct']}% match",
        OUTPUT_DIR / f"{stem1}_vs_{stem2}_base2.jpg",
    )
    print(f"\nDone! Open runs/compare/ to see results.")


if __name__ == "__main__":
    main()
