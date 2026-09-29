"""
YOLOv8 Training Script for Clan Capital Base Detection
Trains on your Roboflow-exported dataset.

Run: python scripts/train.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
import yaml
import shutil

load_dotenv()

# ─────────────────────────────────────────────
DATASET_YAML  = Path(os.getenv("DATASET_YAML", "dataset/data.yaml"))
MODEL_SIZE    = os.getenv("MODEL_SIZE", "models/base/yolo11m.pt")
EPOCHS        = int(os.getenv("TRAIN_EPOCHS", 150))
BATCH_SIZE    = int(os.getenv("TRAIN_BATCH", 4))
IMG_SIZE      = 640
PROJECT_NAME  = os.getenv("TRAIN_PROJECT", "runs/train")
RUN_NAME      = os.getenv("TRAIN_RUN_NAME", "coc_capital_v9_final")
DEVICE        = int(os.getenv("TRAIN_DEVICE", 0))
# ─────────────────────────────────────────────


def fix_yaml_paths(yaml_path: Path) -> Path:
    """
    Roboflow exports relative paths like ../train/images
    We override them with correct absolute paths based on actual folder structure.
    Returns path to the fixed yaml file.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_root = yaml_path.parent.resolve()

    # Roboflow puts train/val/test inside the dataset folder
    # test folder may not exist — create it empty if missing
    config["train"] = str(dataset_root / "train" / "images")
    config["val"]   = str(dataset_root / "valid" / "images")
    config["test"]  = str(dataset_root / "test"  / "images")

    # ensure folders exist (test may be absent in some exports)
    for split in ["train", "val", "test"]:
        Path(config[split]).mkdir(parents=True, exist_ok=True)

    # write fixed yaml
    fixed_path = yaml_path.parent / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Fixed yaml written to: {fixed_path}")
    print(f"  train: {config['train']}")
    print(f"  val:   {config['val']}")
    print(f"  test:  {config['test']}")
    print(f"  classes ({config.get('nc')}): {config.get('names')}")

    return fixed_path


def split_train_val(dataset_root: Path, val_ratio: float = 0.2):
    """
    If Roboflow already generated a val set, skip splitting.
    Only splits manually if val is empty.
    """
    val_images = dataset_root / "valid" / "images"

    if val_images.exists() and any(val_images.iterdir()):
        count = len(list(val_images.iterdir()))
        print(f"Val set already exists with {count} images, skipping split.")
        return

    # manual split fallback
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_labels   = dataset_root / "valid" / "labels"

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(train_images.glob("*.*"))
    n_val  = max(1, int(len(images) * val_ratio))
    val_set = images[-n_val:]

    print(f"Splitting: {len(images)} train images → moving {n_val} to val")
    for img_path in val_set:
        shutil.move(str(img_path), str(val_images / img_path.name))
        lbl_path = train_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(val_labels / lbl_path.name))

    remaining = len(list(train_images.glob("*.*")))
    print(f"  Train: {remaining} images, Val: {n_val} images")


def main():
    print("=" * 50)
    print("  CoC Clan Capital — YOLO Training")
    print("=" * 50)

    dataset_root = DATASET_YAML.parent.resolve()

    # split train/val if needed
    split_train_val(dataset_root)

    # fix yaml paths
    fixed_yaml = fix_yaml_paths(DATASET_YAML)

    # load pretrained YOLOv8 model
    print(f"\nLoading model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    # train
    print(f"\nStarting training for {EPOCHS} epochs on device {DEVICE}...")
    results = model.train(
        data    = str(fixed_yaml),
        epochs  = EPOCHS,
        batch   = BATCH_SIZE,
        imgsz   = IMG_SIZE,
        device  = DEVICE,
        project = PROJECT_NAME,
        name    = RUN_NAME,
        # augmentation — extra on top of Roboflow augmentations
        fliplr  = 0.5,       # horizontal flip
        flipud  = 0.0,       # no vertical flip (bases have a fixed orientation)
        degrees = 10,        # rotation ±10°
        hsv_v   = 0.4,       # brightness variation
        mosaic  = 1.0,       # mosaic augmentation (great for small datasets)
        # training settings
        patience      = 30,  # stop early if no improvement for 30 epochs
        save          = True,
        save_period   = 10,  # save checkpoint every 10 epochs
        plots         = True,
        workers       = 2,
        optimizer     = "AdamW",
        lr0           = 0.001,
        weight_decay  = 0.0005,
    )

    print("\n" + "=" * 50)
    print("Training complete!")
    best_weights = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    print(f"Best weights saved to: {best_weights.resolve()}")
    print(f"Results and plots in:  {(Path(PROJECT_NAME) / RUN_NAME).resolve()}")
    print("=" * 50)

    # quick validation on val set
    print("\nRunning validation...")
    val_results = model.val()
    print(f"  mAP50:    {val_results.box.map50:.3f}")
    print(f"  mAP50-95: {val_results.box.map:.3f}")


if __name__ == "__main__":
    main()
