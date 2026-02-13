#!/usr/bin/env python3
"""
Train dish classification model using YOLO26-N-cls.

This is Stage 3 of the three-stage detection pipeline:
  1. Binary gate -- fast food/not-food filter
  2. Detection -- bounding boxes around food items
  3. Classification (this script) -- identify specific dishes

The classifier identifies specific dish types (e.g., "pad-thai", "sushi",
"hamburger") from cropped food regions. Used for ingredient inference
via the knowledge graph.

Usage:
    python training/train_classify.py [--epochs N] [--batch N] [--device DEVICE]
    python training/train_classify.py --epochs 10  # Quick training run
    python training/train_classify.py --prepare-data  # Auto-download Food-101

Prerequisites:
    - Datasets prepared at training/datasets/food-classification/
    - Run download_datasets.py + merge_datasets.py first, or pass --prepare-data
"""

from __future__ import annotations

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
CLASSIFY_DATA_DIR = DATASETS_DIR / "food-classification"
RUNS_DIR = TRAINING_DIR / "runs"

# Default training config
DEFAULT_CONFIG = {
    "model": "yolo26n-cls.pt",
    "epochs": 50,
    "imgsz": 224,
    "batch": 64,
    "patience": 20,
    "augment": True,
    "project": str(RUNS_DIR / "classify"),
    "name": "food-dish",
}

# YOLO model fallback order
MODEL_FALLBACK = ["yolo26n-cls.pt", "yolo11n-cls.pt", "yolov8n-cls.pt"]


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
    """Load YOLO classification model with fallback chain."""
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
        f"Could not load any YOLO classification model. Tried: {MODEL_FALLBACK}"
    )


def check_dataset(data_dir: Path) -> Optional[dict]:
    """Verify the classification dataset and return stats."""
    if not data_dir.exists():
        return None

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        return None

    train_classes = [
        d for d in train_dir.iterdir() if d.is_dir()
    ]
    val_classes = [
        d for d in val_dir.iterdir() if d.is_dir()
    ]

    if not train_classes:
        return None

    # Count images
    train_images = 0
    for cls_dir in train_classes:
        train_images += sum(
            1
            for f in cls_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )

    val_images = 0
    for cls_dir in val_classes:
        val_images += sum(
            1
            for f in cls_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )

    if train_images == 0:
        return None

    stats = {
        "train_classes": len(train_classes),
        "val_classes": len(val_classes),
        "train_images": train_images,
        "val_images": val_images,
        "class_names": sorted([d.name for d in train_classes]),
    }

    log.info(
        "Dataset found: %d classes, %d train images, %d val images",
        stats["train_classes"],
        stats["train_images"],
        stats["val_images"],
    )

    return stats


def prepare_dataset(data_dir: Path) -> None:
    """Download Food-101 and prepare classification dataset.

    Uses the HuggingFace datasets library for reliable download.
    """
    log.info("Preparing classification dataset from Food-101...")

    random.seed(42)

    try:
        from datasets import load_dataset

        log.info("Downloading Food-101 from HuggingFace (101 classes, ~101K images)...")
        log.info("This may take several minutes on first run...")
        ds = load_dataset("ethz/food101", trust_remote_code=True)

    except Exception as e:
        log.error("Failed to download Food-101: %s", e)
        log.error(
            "Please install the datasets library: pip install datasets\n"
            "Or prepare the dataset manually."
        )
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Process train and validation splits
    splits_map = {"train": "train", "validation": "val"}
    class_names = set()

    for hf_split, local_split in splits_map.items():
        if hf_split not in ds:
            continue

        split_data = ds[hf_split]
        log.info("Processing %s split (%d images)...", hf_split, len(split_data))

        for i, example in enumerate(split_data):
            label = example["label"]
            label_name = split_data.features["label"].int2str(label)

            # Normalize class name
            cls_name = label_name.lower().replace("_", "-").replace(" ", "-")
            class_names.add(cls_name)

            cls_dir = data_dir / local_split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            image = example["image"]
            img_path = cls_dir / f"{cls_name}_{i:06d}.jpg"
            image.save(img_path, "JPEG")

            if (i + 1) % 10000 == 0:
                log.info("  Processed %d/%d images...", i + 1, len(split_data))

    # Create test split from validation (take 50% of val for test)
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    if val_dir.exists():
        for cls_dir in val_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            test_cls_dir = test_dir / cls_dir.name
            test_cls_dir.mkdir(parents=True, exist_ok=True)

            images = sorted(cls_dir.glob("*.jpg"))
            # Move second half to test
            split_point = len(images) // 2
            for img in images[split_point:]:
                img.rename(test_cls_dir / img.name)

    # Write classes.txt
    sorted_classes = sorted(class_names)
    with open(data_dir / "classes.txt", "w") as f:
        for cls in sorted_classes:
            f.write(f"{cls}\n")

    log.info(
        "Dataset prepared: %d classes at %s",
        len(sorted_classes),
        data_dir,
    )


def train(args):
    """Run dish classifier training."""
    device = args.device or get_device()

    # Check dataset
    stats = check_dataset(CLASSIFY_DATA_DIR)
    if stats is None:
        if args.prepare_data:
            prepare_dataset(CLASSIFY_DATA_DIR)
            stats = check_dataset(CLASSIFY_DATA_DIR)
            if stats is None:
                log.error("Dataset preparation failed.")
                sys.exit(1)
        else:
            log.error(
                "Classification dataset not found at %s.\n"
                "Options:\n"
                "  1. Run: python training/datasets/scripts/download_datasets.py && "
                "python training/datasets/scripts/merge_datasets.py\n"
                "  2. Pass --prepare-data to auto-download Food-101",
                CLASSIFY_DATA_DIR,
            )
            sys.exit(1)

    # Load model
    model, model_name = load_model(args.model)

    config = {
        "model": model_name,
        "data": str(CLASSIFY_DATA_DIR),
        "classes": stats["train_classes"],
        "train_images": stats["train_images"],
        "val_images": stats["val_images"],
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "patience": args.patience,
        "augment": args.augment,
    }

    print("\n" + "=" * 60)
    print("DISH CLASSIFICATION TRAINING")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60 + "\n")

    # Graceful interrupt handler
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            log.warning("Force quit.")
            sys.exit(1)
        interrupted = True
        log.info("Interrupt received. Finishing current epoch...")

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_interrupt)

    start_time = time.time()

    try:
        results = model.train(
            data=str(CLASSIFY_DATA_DIR),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=args.patience,
            augment=args.augment,
            project=str(RUNS_DIR / "classify"),
            name="food-dish",
            exist_ok=True,
            verbose=True,
        )

        elapsed = time.time() - start_time

        # Validate
        log.info("Running validation on best model...")
        metrics = model.val()

        top1 = metrics.top1
        top5 = metrics.top5

        print("\n" + "=" * 60)
        print("DISH CLASSIFICATION TRAINING RESULTS")
        print("=" * 60)
        print(f"  Top-1 Accuracy: {top1:.3f} ({top1:.1%})")
        print(f"  Top-5 Accuracy: {top5:.3f} ({top5:.1%})")
        print(f"  Classes:        {stats['train_classes']}")
        print(f"  Training time:  {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Model saved to: {RUNS_DIR / 'classify' / 'food-dish' / 'weights' / 'best.pt'}")
        print("=" * 60)

        if top1 >= 0.70:
            print("  TARGET MET: >70% Top-1 accuracy")
        elif top1 >= 0.50:
            print(f"  ACCEPTABLE: {top1:.1%} Top-1 (target was 70%, but acceptable at this stage)")
        else:
            print(f"  BELOW TARGET: {top1:.1%} < 50% -- may need more data or epochs")

        # Save metrics
        metrics_data = {
            "model": model_name,
            "task": "dish-classification",
            "num_classes": stats["train_classes"],
            "class_names": stats["class_names"][:20],  # Save first 20 for reference
            "total_class_count": len(stats["class_names"]),
            "top1_accuracy": float(top1),
            "top5_accuracy": float(top5),
            "epochs_trained": args.epochs,
            "training_time_seconds": elapsed,
            "device": device,
            "config": config,
            "best_model_path": str(
                RUNS_DIR / "classify" / "food-dish" / "weights" / "best.pt"
            ),
        }

        metrics_path = RUNS_DIR / "classify" / "food-dish" / "metrics.json"
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
        description="Train dish classification model (Food-101 + ISIA-500)"
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
        help="Auto-download Food-101 and prepare dataset",
    )
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    train(args)


if __name__ == "__main__":
    main()
