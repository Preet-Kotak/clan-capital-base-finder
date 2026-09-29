"""
Picks 1 random image per district from data/screenshots/by_district,
runs the model, draws bounding boxes + labels, and saves one result per district.

Usage: python scripts/visualize_predictions.py
"""

import random
import cv2
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
WEIGHTS      = "runs/train/coc_capital_v9_final/weights/best.pt"
SCREENSHOTS  = Path("data/screenshots/by_district")
OUTPUT_DIR   = Path("runs/visualize")
CONF         = 0.25
RANDOM_SEED  = 42

# only run on labeled districts
DISTRICTS = [
    "capital_peak",
    "barbarian_camp",
    "wizard_valley",
    "balloon_lagoon",
    "builders_workshop",
    "dragon_cliffs",
    "golem_quarry",
    "skeleton_park",
    "goblin_mines",
]

CLASS_COLORS = {
    "cannon":          (0,   200, 255),
    "air_defence":     (0,   255, 100),
    "motor":           (255, 100,   0),
    "inferno_tower":   (0,   80,  255),
    "wizard_tower":    (180,   0, 255),
    "air_bomb":        (0,   255, 220),
    "bomb_tower":      (50,   50, 255),
    "tesla":           (255, 220,   0),
    "gaint_cannon":    (0,   160, 255),
    "multi_cannon":    (0,   255, 160),
    "rapid_rocket":    (255, 160,   0),
    "rocket_artilery": (100, 255,   0),
    "spear":           (255,   0, 160),
    "blast_bow":       (0,   100, 255),
    "crusher":         (160, 255,   0),
    "district_hall":   (255, 255,   0),
    "capital_peak":    (0,   255, 255),
    "post_cannon":     (200, 200, 200),
    "post_dragon":     (150, 100, 255),
    "post_gaint":      (255, 150, 100),
}
# ─────────────────────────────────────────────


def pick_random_image(district_dir: Path) -> Path | None:
    images = list(district_dir.glob("*.jpg")) + list(district_dir.glob("*.png"))
    if not images:
        return None
    return random.choice(images)


def draw_predictions(img_path: Path, results, output_path: Path, district: str):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Could not read {img_path}")
        return

    names = results[0].names
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = names[cls_id]
        color  = CLASS_COLORS.get(label, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # district label top-left
    title = district.replace("_", " ").title()
    cv2.putText(img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # detection count bottom-left
    count_text = f"{len(boxes)} detections"
    cv2.putText(img, count_text, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"  Saved → {output_path}  ({len(boxes)} detections)")


def main():
    random.seed(RANDOM_SEED)

    print("=" * 50)
    print("  Visual Prediction Check")
    print(f"  Source: {SCREENSHOTS}")
    print("=" * 50)

    model = YOLO(WEIGHTS)

    print()
    for district in DISTRICTS:
        district_dir = SCREENSHOTS / district
        if not district_dir.exists():
            print(f"  ✗ {district} — folder not found, skipping")
            continue

        img_path = pick_random_image(district_dir)
        if img_path is None:
            print(f"  ✗ {district} — no images found, skipping")
            continue

        print(f"  [{district}] using: {img_path.name}")

        results = model.predict(
            source=str(img_path),
            conf=CONF,
            iou=0.5,
            verbose=False,
        )

        out_path = OUTPUT_DIR / f"{district}_prediction.jpg"
        draw_predictions(img_path, results, out_path, district)

    print(f"\nDone! Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
