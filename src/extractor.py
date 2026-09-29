"""
Layout extractor — runs YOLO on a screenshot and returns normalized building positions.
Includes confidence auto-calibration to hit the expected building count per district.

Usage:
    from src.extractor import Extractor
    ext = Extractor()
    layout = ext.extract("path/to/image.jpg", district=0)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ─────────────────────────────────────────────
WEIGHTS      = os.getenv("WEIGHTS", "runs/train/coc_capital_v9_final/weights/best.pt")
DEFAULT_CONF = 0.25
IOU          = 0.5
MAX_RUNS     = 3
CONF_STEP    = 0.05
CONF_MIN     = 0.10
CONF_MAX     = 0.50

DISTRICT_MAP = {
    0: "capital_peak",
    1: "barbarian_camp",
    2: "wizard_valley",
    3: "balloon_lagoon",
    4: "builders_workshop",
    5: "dragon_cliffs",
    6: "golem_quarry",
    7: "skeleton_park",
    8: "goblin_mines",
}

# valid building counts per district — either count is acceptable
# None means no calibration for that district
DISTRICT_COUNTS: dict[int, list[int]] = {
    0: [50, 48],  # capital_peak
    1: [58, 54],  # barbarian_camp
    2: [49, 47],  # wizard_valley
    3: [56, 54],  # balloon_lagoon
    4: [58, 53],  # builders_workshop
    5: [47],      # dragon_cliffs
    6: [58, 57],  # golem_quarry
    7: [59, 54],  # skeleton_park
    8: [55, 52],  # goblin_mines
}

ANCHOR_CLASSES = {"district_hall", "capital_peak"}
# ─────────────────────────────────────────────


def _is_valid_count(count: int, valid_counts: list[int]) -> bool:
    """Check if detection count exactly matches any valid count."""
    return count in valid_counts


def _closest_count(count: int, valid_counts: list[int]) -> int:
    """Return which valid count is closest to detected count."""
    return min(valid_counts, key=lambda v: abs(count - v))


class Extractor:
    def __init__(self, weights: str = WEIGHTS):
        self.model = YOLO(weights)

    def _run_inference(self, image_path: Path, conf: float) -> dict:
        """Run YOLO once and return raw detection results."""
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=IOU,
            verbose=False,
        )

        result = results[0]
        names  = result.names
        boxes  = result.boxes

        buildings    = []
        anchor_x     = 0.5
        anchor_y     = 0.5
        anchor_found = False

        # first pass — find anchor
        for box in boxes:
            cls_name = names[int(box.cls[0])]
            if cls_name in ANCHOR_CLASSES:
                anchor_x = float(box.xywhn[0][0])
                anchor_y = float(box.xywhn[0][1])
                anchor_found = True
                break

        # second pass — extract all buildings relative to anchor
        for box in boxes:
            cls_name = names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            x_center = float(box.xywhn[0][0])
            y_center = float(box.xywhn[0][1])

            buildings.append({
                "type": cls_name,
                "x":    round(x_center - anchor_x, 4),
                "y":    round(y_center - anchor_y, 4),
                "conf": round(conf_val, 3),
            })

        return {
            "buildings":        buildings,
            "anchor_found":     anchor_found,
            "total_detections": len(buildings),
        }

    def extract(self, image_path: str | Path, district: int | None = None) -> dict:
        """
        Run YOLO on image with auto-calibration if district is provided.

        Args:
            image_path: path to screenshot
            district:   district number 0-8 (enables calibration)

        Returns:
            {
                "buildings":        [{type, x, y, conf}],
                "anchor_found":     bool,
                "total_detections": int,
                "conf_used":        float,
                "calibrated":       bool,
                "runs":             int,
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        valid_counts = DISTRICT_COUNTS.get(district) if district is not None else None

        # no calibration needed
        if valid_counts is None:
            result = self._run_inference(image_path, DEFAULT_CONF)
            result["conf_used"]  = DEFAULT_CONF
            result["calibrated"] = False
            result["runs"]       = 1
            return result

        conf         = DEFAULT_CONF
        best_result  = None
        best_diff    = float("inf")

        for run in range(1, MAX_RUNS + 1):
            result = self._run_inference(image_path, conf)
            count  = result["total_detections"]

            # track best result so far
            diff = min(abs(count - v) for v in valid_counts)
            if diff < best_diff:
                best_diff   = diff
                best_result = result
                best_result["conf_used"]  = conf
                best_result["runs"]       = run
                best_result["calibrated"] = True

            # check if we hit a valid count
            if _is_valid_count(count, valid_counts):
                best_result["calibrated"] = True
                return best_result

            # last run — return best we found
            if run == MAX_RUNS:
                break

            # adjust conf for next run
            closest = _closest_count(count, valid_counts)
            if count < closest:
                # too few buildings → lower conf to detect more
                conf = max(CONF_MIN, conf - CONF_STEP)
            else:
                # too many buildings → raise conf to be stricter
                conf = min(CONF_MAX, conf + CONF_STEP)

        return best_result
