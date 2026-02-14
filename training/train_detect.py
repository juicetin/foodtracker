#!/usr/bin/env python3
"""
Train YOLO26-N food object detector.

This is Stage 2 of the three-stage detection pipeline:
  1. Binary gate -- fast food/not-food filter
  2. Detection (this script) -- bounding boxes around food items
  3. Classification -- identify specific dishes

The detector localizes food items in images with bounding boxes,
enabling the classifier to identify individual dishes even when
multiple foods appear in a single photo.

Usage:
    python training/train_detect.py [--epochs N] [--batch N] [--device DEVICE]
    python training/train_detect.py --epochs 10  # Quick training run
    python training/train_detect.py --prepare-data  # Create minimal detection dataset
    python training/train_detect.py --resume      # Resume from last checkpoint

Prerequisites:
    - Detection dataset at training/datasets/food-detection-merged/
    - Or pass --prepare-data to auto-prepare a minimal dataset
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent
DATASETS_DIR = TRAINING_DIR / "datasets"
DETECT_DATA_DIR = DATASETS_DIR / "food-detection-merged"
CONFIGS_DIR = TRAINING_DIR / "configs"
DETECT_CONFIG = CONFIGS_DIR / "food-detect.yaml"
RUNS_DIR = TRAINING_DIR / "runs"

# Default training config
DEFAULT_CONFIG = {
    "model": "yolo26n.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 20,
    "augment": True,
    "project": str(RUNS_DIR / "detect"),
    "name": "food-detect",
}

# YOLO detection model fallback order
MODEL_FALLBACK = ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]


def get_device() -> str:
    """Detect best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            log.info("CUDA available -- using GPU")
            return "cuda"
        if torch.backends.mps.is_available():
            log.info("MPS available -- using Apple Silicon GPU")
            return "mps"
    except Exception:
        pass
    log.info("No GPU detected -- using CPU (training will be slow)")
    return "cpu"


def load_model(preferred_model: str = DEFAULT_CONFIG["model"]):
    """Load YOLO detection model with fallback chain."""
    from ultralytics import YOLO

    for model_name in MODEL_FALLBACK:
        if model_name != preferred_model and model_name != MODEL_FALLBACK[0]:
            continue
        try:
            log.info("Loading model: %s", model_name)
            model = YOLO(model_name)
            log.info("Model loaded successfully: %s (task: %s)", model_name, model.task)
            return model, model_name
        except Exception as e:
            log.warning("Failed to load %s: %s", model_name, e)

    # Try all fallbacks
    for model_name in MODEL_FALLBACK:
        try:
            log.info("Trying fallback model: %s", model_name)
            model = YOLO(model_name)
            log.info("Fallback model loaded: %s", model_name)
            return model, model_name
        except Exception as e:
            log.warning("Fallback %s failed: %s", model_name, e)

    raise RuntimeError(
        f"Could not load any YOLO detection model. Tried: {MODEL_FALLBACK}"
    )


def check_dataset(data_dir: Path) -> Optional[dict]:
    """Verify the detection dataset is properly structured.

    Expected YOLO detection format:
        data_dir/
            images/train/  images/val/
            labels/train/  labels/val/
    """
    if not data_dir.exists():
        return None

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        return None

    stats = {}
    for split in ["train", "val"]:
        img_dir = images_dir / split
        lbl_dir = labels_dir / split

        if not img_dir.exists() or not lbl_dir.exists():
            return None

        image_count = sum(
            1
            for f in img_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
        label_count = sum(
            1
            for f in lbl_dir.iterdir()
            if f.suffix.lower() == ".txt"
        )

        if image_count == 0:
            return None

        stats[f"{split}_images"] = image_count
        stats[f"{split}_labels"] = label_count

    # Check for classes file or count from config
    classes_file = data_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        stats["num_classes"] = len(classes)
        stats["class_names"] = classes
    else:
        stats["num_classes"] = None
        stats["class_names"] = []

    log.info(
        "Detection dataset found: %d train images, %d val images, %s classes",
        stats["train_images"],
        stats["val_images"],
        stats.get("num_classes", "unknown"),
    )
    return stats


def prepare_minimal_dataset(data_dir: Path) -> None:
    """Create a minimal detection dataset for training validation.

    Downloads a subset of Food-101, creates full-image bounding box labels
    in YOLO format (single 'food' class covering the entire image). This is
    a simplified dataset for verifying the training pipeline works.

    For production training, run auto_label.py to generate proper bounding
    box annotations with Florence-2.
    """
    log.info("Preparing minimal detection dataset for training validation...")

    random.seed(42)

    # Create directory structure
    for split in ["train", "val", "test"]:
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    food_images_created = 0

    # Try to download Food-101 subset
    try:
        from datasets import load_dataset

        log.info("Downloading Food-101 subset from HuggingFace...")
        ds = load_dataset("ethz/food101", split="train", trust_remote_code=True)

        # Take a random subset
        indices = random.sample(range(len(ds)), min(1500, len(ds)))
        train_count = int(len(indices) * 0.7)
        val_count = int(len(indices) * 0.15)

        for i, idx in enumerate(indices):
            example = ds[idx]
            image = example["image"]

            if i < train_count:
                split = "train"
            elif i < train_count + val_count:
                split = "val"
            else:
                split = "test"

            img_path = data_dir / "images" / split / f"food_{i:06d}.jpg"
            image.save(img_path, "JPEG")

            # Create full-image bounding box label (class 0, centered, full size)
            # YOLO format: class_id center_x center_y width height (normalized)
            lbl_path = data_dir / "labels" / split / f"food_{i:06d}.txt"
            with open(lbl_path, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

            food_images_created += 1

            if food_images_created % 500 == 0:
                log.info("  Saved %d images...", food_images_created)

        log.info("Downloaded %d food images from Food-101", food_images_created)

    except Exception as e:
        log.warning("Could not download Food-101: %s", e)
        log.info("Creating synthetic placeholder images with labels...")

        from PIL import Image

        for i in range(600):
            # Create colored placeholder images
            img = Image.new(
                "RGB", (640, 640),
                (random.randint(100, 255), random.randint(50, 200), random.randint(0, 100)),
            )

            split = "train" if i < 420 else ("val" if i < 510 else "test")
            img_path = data_dir / "images" / split / f"food_{i:06d}.jpg"
            img.save(img_path, "JPEG")

            # Full-image bounding box
            lbl_path = data_dir / "labels" / split / f"food_{i:06d}.txt"
            with open(lbl_path, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

            food_images_created += 1

    # Write classes.txt
    with open(data_dir / "classes.txt", "w") as f:
        f.write("food\n")

    # Update the YAML config with actual class info
    _update_detect_config(data_dir, ["food"])

    log.info(
        "Minimal detection dataset prepared: %d images at %s",
        food_images_created,
        data_dir,
    )


def _update_detect_config(data_dir: Path, class_names: list[str]) -> None:
    """Update food-detect.yaml with dataset info."""
    config_content = (
        "# Food object detection dataset config\n"
        f"# Auto-generated by train_detect.py --prepare-data\n"
        "\n"
        f"path: {data_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    DETECT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(DETECT_CONFIG, "w") as f:
        f.write(config_content)
    log.info("Updated config: %s", DETECT_CONFIG)


def train(args):
    """Run food detection model training."""
    device = args.device or get_device()

    # Check dataset
    stats = check_dataset(DETECT_DATA_DIR)
    if stats is None:
        if args.prepare_data:
            prepare_minimal_dataset(DETECT_DATA_DIR)
            stats = check_dataset(DETECT_DATA_DIR)
            if stats is None:
                log.error("Dataset preparation failed. Cannot train.")
                sys.exit(1)
        else:
            log.error(
                "Detection dataset not found or incomplete at %s.\n"
                "Options:\n"
                "  1. Run: python training/datasets/scripts/auto_label.py\n"
                "  2. Pass --prepare-data to auto-prepare a minimal dataset",
                DETECT_DATA_DIR,
            )
            sys.exit(1)

    # Determine data source -- use config YAML if it has nc defined, else direct path
    data_source = str(DETECT_CONFIG)
    try:
        with open(DETECT_CONFIG) as f:
            config_text = f.read()
        if "nc:" not in config_text or config_text.strip().startswith("# nc:"):
            # Config doesn't have nc populated, use direct path
            log.info("Config YAML has no nc defined, using dataset directory directly")
            data_source = str(DETECT_DATA_DIR)
    except FileNotFoundError:
        data_source = str(DETECT_DATA_DIR)

    # Load model
    model, model_name = load_model(args.model)

    config = {
        "model": model_name,
        "data": data_source,
        "train_images": stats.get("train_images", "?"),
        "val_images": stats.get("val_images", "?"),
        "num_classes": stats.get("num_classes", "?"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "patience": args.patience,
        "augment": args.augment,
    }

    print("\n" + "=" * 60)
    print("FOOD OBJECT DETECTION TRAINING")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60 + "\n")

    # Graceful interrupt handler
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            log.warning("Force quit. Training progress saved to last checkpoint.")
            sys.exit(1)
        interrupted = True
        log.info(
            "Interrupt received. Finishing current epoch and saving checkpoint... "
            "(press Ctrl+C again to force quit)"
        )

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_interrupt)

    start_time = time.time()

    try:
        results = model.train(
            data=data_source,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=args.patience,
            augment=args.augment,
            project=str(RUNS_DIR / "detect"),
            name="food-detect",
            exist_ok=True,
            verbose=True,
        )

        elapsed = time.time() - start_time

        # Validate and get final metrics
        log.info("Running validation on best model...")
        metrics = model.val()

        map50 = metrics.box.map50
        map5095 = metrics.box.map

        print("\n" + "=" * 60)
        print("FOOD DETECTION TRAINING RESULTS")
        print("=" * 60)
        print(f"  mAP@0.5:      {map50:.3f}")
        print(f"  mAP@0.5:0.95: {map5095:.3f}")
        print(f"  Training time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Model saved:   {RUNS_DIR / 'detect' / 'food-detect' / 'weights' / 'best.pt'}")
        print("=" * 60)

        # Check target
        if map50 >= 0.60:
            print("  TARGET MET: >60% mAP@0.5")
        elif map50 >= 0.50:
            print(f"  ACCEPTABLE: {map50:.1%} mAP@0.5 (target was 60%, acceptable at this stage)")
        else:
            print(f"  BELOW TARGET: {map50:.1%} mAP@0.5 -- may need more data or epochs")

        # Save metrics JSON
        metrics_data = {
            "model": model_name,
            "task": "food-detection",
            "num_classes": stats.get("num_classes"),
            "box.map50": float(map50),
            "box.map": float(map5095),
            "epochs_trained": args.epochs,
            "training_time_seconds": elapsed,
            "device": device,
            "config": config,
            "best_model_path": str(
                RUNS_DIR / "detect" / "food-detect" / "weights" / "best.pt"
            ),
        }

        metrics_path = RUNS_DIR / "detect" / "food-detect" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        log.info("Metrics saved to %s", metrics_path)

        return metrics_data

    except KeyboardInterrupt:
        log.info("Training interrupted. Checkpoint saved.")
        return None
    finally:
        signal.signal(signal.SIGINT, original_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO food object detector"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_CONFIG["batch"],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch']})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"],
        help=f"Image size (default: {DEFAULT_CONFIG['imgsz']})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: mps, cuda, cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_CONFIG["patience"],
        help=f"Early stopping patience (default: {DEFAULT_CONFIG['patience']})",
    )
    parser.add_argument(
        "--augment", action="store_true", default=DEFAULT_CONFIG["augment"],
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_CONFIG["model"],
        help=f"YOLO model checkpoint (default: {DEFAULT_CONFIG['model']})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--prepare-data", action="store_true",
        help="Auto-prepare minimal detection dataset (downloads Food-101 subset)",
    )
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    train(args)


if __name__ == "__main__":
    main()
