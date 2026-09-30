"""
Layout extractor — runs YOLO on a screenshot and returns normalized building positions.
Uses two SAHI tiles, overlap cleanup, and one highest-confidence hall.

Usage:
    from src.extractor import Extractor
    ext = Extractor()
    layout = ext.extract("path/to/image.jpg", district=0)
"""

import os
from math import ceil
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from dotenv import load_dotenv
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.postprocess.combine import NMSPostprocess
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

load_dotenv()

# ─────────────────────────────────────────────
WEIGHTS      = os.getenv("WEIGHTS", "runs/train/coc_capital_v9_final/weights/best.pt")
DEFAULT_CONF = 0.25
IOU          = 0.5
IMAGE_SIZE   = 448
OVERLAP      = 0.2
CLEANUP_IOU  = 0.8

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


class _SlicedModel(UltralyticsDetectionModel):
    def perform_batch_inference(self, images: list[NDArray[np.uint8]]) -> None:
        predictions = self.model.predict(
            source=[image[:, :, ::-1] for image in images], device=self.device,
            conf=self.confidence_threshold, iou=IOU, imgsz=self.image_size, batch=len(images),
            rect=False, half=False, augment=False, max_det=300,
            save=False, save_txt=False, save_crop=False, verbose=False,
        )
        self._original_predictions = self._extract_predictions(predictions)
        self._original_shapes = [image.shape for image in images]


class Extractor:
    def __init__(self, weights: str = WEIGHTS):
        self.model = YOLO(weights)
        self.detector = _SlicedModel(model=self.model, confidence_threshold=DEFAULT_CONF, image_size=IMAGE_SIZE)

    def _run_inference(self, image_path: Path, conf: float) -> dict:
        """Return cleaned layout and pixel boxes from the same sliced predictions."""
        self.detector.confidence_threshold = conf
        with Image.open(image_path) as source, source.convert("RGB") as image:
            width, height = image.size
            result = get_sliced_prediction(
                image, self.detector, slice_height=height, slice_width=ceil(width / (2 - OVERLAP)),
                overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
                perform_standard_pred=False, postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS", postprocess_match_threshold=0.5,
                postprocess_class_agnostic=False, auto_slice_resolution=False,
                batch_size=1, verbose=0, progress_bar=False,
            )
        boxes = result.object_prediction_list
        if boxes:
            boxes = NMSPostprocess(match_threshold=CLEANUP_IOU, match_metric="IOU", class_agnostic=True)(boxes)
        hall = min((box for box in boxes if box.category.name in ANCHOR_CLASSES), default=None,
                   key=lambda box: (-box.score.value, box.category.name, tuple(box.bbox.to_xyxy())))

        buildings    = []
        pixel_boxes  = []
        anchor_x     = 0.5
        anchor_y     = 0.5
        anchor_found = False

        if hall is not None:
            x1, y1, x2, y2 = hall.bbox.to_xyxy()
            anchor_x = (x1 + x2) / (2 * width)
            anchor_y = (y1 + y2) / (2 * height)
            anchor_found = True

        # second pass — extract all buildings relative to anchor
        for box in boxes:
            cls_name = box.category.name
            if cls_name in ANCHOR_CLASSES and box is not hall:
                continue
            x1, y1, x2, y2 = box.bbox.to_xyxy()
            conf_val = float(box.score.value)
            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)

            buildings.append({
                "type": cls_name,
                "x":    round(x_center - anchor_x, 4),
                "y":    round(y_center - anchor_y, 4),
                "conf": round(conf_val, 3),
            })
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            pixel_boxes.append({
                **buildings[-1], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
            })

        return {
            "buildings":        buildings,
            "anchor_found":     anchor_found,
            "total_detections": len(buildings),
            "pixel_boxes":      pixel_boxes,
        }

    def extract(self, image_path: str | Path, district: int | None = None) -> dict:
        """
        Run sliced inference at fixed confidence; district enables the count check.

        Args:
            image_path: path to screenshot
            district:   district number 0-8 (checks the expected count)

        Returns:
            {
                "buildings":        [{type, x, y, conf}],
                "anchor_found":     bool,
                "total_detections": int,
                "conf_used":        float,
                "calibrated":       bool,
                "runs":             int,
                "pixel_boxes":      [{type, x, y, conf, x1, y1, x2, y2, cx, cy}],
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        valid_counts = DISTRICT_COUNTS.get(district) if district is not None else None

        result = self._run_inference(image_path, DEFAULT_CONF)
        result["conf_used"]  = DEFAULT_CONF
        result["calibrated"] = valid_counts is not None and _is_valid_count(result["total_detections"], valid_counts)
        result["runs"]       = 1
        return result
