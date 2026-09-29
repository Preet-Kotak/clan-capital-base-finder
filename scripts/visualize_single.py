"""
Visualize a single image with YOLO predictions.
Usage: python scripts/visualize_single.py
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

WEIGHTS   = "runs/train/coc_capital_v9_final/weights/best.pt"
IMAGE     = "data/screenshots/by_district/capital_peak/capital_peak_469.png"
OUTPUT    = "runs/visualize/capital_peak_469_debug.jpg"
CONF      = 0.15  # low conf to see everything

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
    "hive":            (0,   200, 200),
    "reflector":       (200,   0, 200),
    "post_goblin":     (255, 200,   0),
}

model   = YOLO(WEIGHTS)
results = model.predict(source=IMAGE, conf=CONF, iou=0.5, verbose=False)
result  = results[0]
names   = result.names
boxes   = result.boxes

img = cv2.imread(IMAGE)

# count per class
class_counts: dict[str, int] = {}
for box in boxes:
    label = names[int(box.cls[0])]
    class_counts[label] = class_counts.get(label, 0) + 1

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf_val = float(box.conf[0])
    color    = CLASS_COLORS.get(label, (200, 200, 200))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf_val:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, text, (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

# draw total count top-left
total_text = f"Total: {len(boxes)} detections (conf>={CONF})"
cv2.putText(img, total_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img, total_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(OUTPUT, img)

print(f"Total detections: {len(boxes)}")
print(f"Saved to: {OUTPUT}")
print("\nPer class:")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls:20s}: {count}")
