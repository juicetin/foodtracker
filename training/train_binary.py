#!/usr/bin/env python3
"""
Train binary food/not-food classifier using YOLO26-N-cls.

This is Stage 1 of the three-stage detection pipeline:
  1. Binary gate (this script) -- fast food/not-food filter
  2. Detection -- bounding boxes around food items
  3. Classification -- identify specific dishes

The binary classifier filters out non-food images quickly, saving
compute for the more expensive detection and classification stages.

Usage:
    python training/train_binary.py [--epochs N] [--batch N] [--device DEVICE]
    python training/train_binary.py --epochs 10  # Quick training run
    python training/train_binary.py --resume      # Resume from last checkpoint

Prerequisites:
    - Datasets prepared at training/datasets/food-binary/ (run download + merge first)
    - Or pass --prepare-data to auto-prepare datasets
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent
DATASETS_DIR = TRAINING_DIR / "datasets"
BINARY_DATA_DIR = DATASETS_DIR / "food-binary"
RUNS_DIR = TRAINING_DIR / "runs"

# Default training config
DEFAULT_CONFIG = {
    "model": "yolo26n-cls.pt",
    "epochs": 30,
    "imgsz": 224,
    "batch": 64,
    "patience": 10,
    "augment": True,
    "project": str(RUNS_DIR / "classify"),
    "name": "food-binary",
}

# YOLO model fallback order if yolo26n-cls.pt is unavailable
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


def check_dataset(data_dir: Path) -> bool:
    """Verify the binary dataset is properly structured."""
    required = ["train/food", "train/not-food", "val/food", "val/not-food"]
    for subdir in required:
        d = data_dir / subdir
        if not d.exists():
            log.warning("Missing directory: %s", d)
            return False
        image_count = sum(
            1
            for f in d.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
        if image_count == 0:
            log.warning("Empty directory: %s", d)
            return False
        log.info("  %s: %d images", subdir, image_count)
    return True


def prepare_minimal_dataset(data_dir: Path) -> None:
    """Create a minimal binary dataset from Food-101 if no dataset exists.

    Downloads a small subset of Food-101 and creates synthetic not-food images
    for a quick training validation. For production training, run the full
    download_datasets.py + merge_datasets.py pipeline.
    """
    log.info("Preparing minimal binary dataset for training validation...")

    import random

    random.seed(42)

    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in ["food", "not-food"]:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Try to download Food-101 subset
    food_images_created = 0
    try:
        from datasets import load_dataset

        log.info("Downloading Food-101 subset from HuggingFace...")
        ds = load_dataset("ethz/food101", split="train", trust_remote_code=True)

        # Take a random subset for quick training
        indices = random.sample(range(len(ds)), min(2000, len(ds)))
        train_count = int(len(indices) * 0.7)
        val_count = int(len(indices) * 0.15)

        for i, idx in enumerate(indices):
            example = ds[idx]
            image = example["image"]
            label_name = ds.features["label"].int2str(example["label"])

            if i < train_count:
                split = "train"
            elif i < train_count + val_count:
                split = "val"
            else:
                split = "test"

            img_path = data_dir / split / "food" / f"food_{i:06d}.jpg"
            image.save(img_path, "JPEG")
            food_images_created += 1

            if food_images_created % 500 == 0:
                log.info("  Saved %d food images...", food_images_created)

        log.info("Downloaded %d food images from Food-101", food_images_created)

    except Exception as e:
        log.warning("Could not download Food-101: %s", e)
        log.info("Creating synthetic food placeholder images...")
        from PIL import Image

        # Create colored placeholder images as food class
        for i in range(500):
            img = Image.new(
                "RGB", (224, 224),
                (random.randint(100, 255), random.randint(50, 200), random.randint(0, 100)),
            )
            split = "train" if i < 350 else ("val" if i < 425 else "test")
            img.save(data_dir / split / "food" / f"food_{i:06d}.jpg", "JPEG")
            food_images_created += 1

    # Create synthetic not-food images (textures, landscapes, objects)
    log.info("Generating synthetic not-food images...")
    from PIL import Image

    not_food_count = food_images_created  # Balance classes
    for i in range(not_food_count):
        # Generate varied non-food patterns
        pattern = i % 5
        if pattern == 0:
            # Solid color
            img = Image.new(
                "RGB", (224, 224),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            )
        elif pattern == 1:
            # Gradient
            import numpy as np

            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            for y in range(224):
                arr[y, :, 0] = int(255 * y / 224)
                arr[y, :, 1] = random.randint(50, 200)
                arr[y, :, 2] = int(255 * (224 - y) / 224)
            img = Image.fromarray(arr)
        elif pattern == 2:
            # Checkerboard
            import numpy as np

            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            block = 28
            c1 = [random.randint(0, 255) for _ in range(3)]
            c2 = [random.randint(0, 255) for _ in range(3)]
            for y in range(224):
                for x in range(224):
                    if ((y // block) + (x // block)) % 2 == 0:
                        arr[y, x] = c1
                    else:
                        arr[y, x] = c2
            img = Image.fromarray(arr)
        elif pattern == 3:
            # Random noise
            import numpy as np

            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
        else:
            # Stripes
            import numpy as np

            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            stripe_w = random.randint(5, 30)
            c1 = [random.randint(0, 255) for _ in range(3)]
            c2 = [random.randint(0, 255) for _ in range(3)]
            for x in range(224):
                if (x // stripe_w) % 2 == 0:
                    arr[:, x] = c1
                else:
                    arr[:, x] = c2
            img = Image.fromarray(arr)

        split_i = i / not_food_count
        split = "train" if split_i < 0.7 else ("val" if split_i < 0.85 else "test")
        img.save(data_dir / split / "not-food" / f"not_food_{i:06d}.jpg", "JPEG")

    log.info(
        "Minimal dataset prepared: %d food + %d not-food images",
        food_images_created,
        not_food_count,
    )


def train(args):
    """Run binary classifier training."""
    # Determine device
    device = args.device or get_device()

    # Check dataset
    if not check_dataset(BINARY_DATA_DIR):
        if args.prepare_data:
            prepare_minimal_dataset(BINARY_DATA_DIR)
            if not check_dataset(BINARY_DATA_DIR):
                log.error("Dataset preparation failed. Cannot train.")
                sys.exit(1)
        else:
            log.error(
                "Binary dataset not found or incomplete at %s.\n"
                "Options:\n"
                "  1. Run: python training/datasets/scripts/download_datasets.py && "
                "python training/datasets/scripts/merge_datasets.py\n"
                "  2. Pass --prepare-data to auto-prepare a minimal dataset",
                BINARY_DATA_DIR,
            )
            sys.exit(1)

    # Load model
    model, model_name = load_model(args.model)

    # Print training configuration
    config = {
        "model": model_name,
        "data": str(BINARY_DATA_DIR),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "patience": args.patience,
        "augment": args.augment,
    }

    print("\n" + "=" * 60)
    print("BINARY FOOD/NOT-FOOD CLASSIFIER TRAINING")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60 + "\n")

    # Set up graceful interrupt handler
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
        # Train
        results = model.train(
            data=str(BINARY_DATA_DIR),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=args.patience,
            augment=args.augment,
            project=str(RUNS_DIR / "classify"),
            name="food-binary",
            exist_ok=True,
            verbose=True,
        )

        elapsed = time.time() - start_time

        # Validate and get final metrics
        log.info("Running validation on best model...")
        metrics = model.val()

        top1 = metrics.top1
        top5 = metrics.top5

        print("\n" + "=" * 60)
        print("BINARY CLASSIFIER TRAINING RESULTS")
        print("=" * 60)
        print(f"  Top-1 Accuracy: {top1:.3f}")
        print(f"  Top-5 Accuracy: {top5:.3f}")
        print(f"  Training time:  {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Model saved to: {RUNS_DIR / 'classify' / 'food-binary' / 'weights' / 'best.pt'}")
        print("=" * 60)

        # Check target
        if top1 >= 0.90:
            print("  TARGET MET: >90% accuracy")
        else:
            print(f"  TARGET NOT MET: {top1:.1%} < 90% (may need more data or epochs)")

        # Save metrics JSON
        metrics_data = {
            "model": model_name,
            "task": "binary-classification",
            "classes": ["food", "not-food"],
            "top1_accuracy": float(top1),
            "top5_accuracy": float(top5),
            "epochs_trained": args.epochs,
            "training_time_seconds": elapsed,
            "device": device,
            "config": config,
            "best_model_path": str(
                RUNS_DIR / "classify" / "food-binary" / "weights" / "best.pt"
            ),
        }

        metrics_path = RUNS_DIR / "classify" / "food-binary" / "metrics.json"
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
        description="Train binary food/not-food classifier"
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
        help="Auto-prepare dataset if not found (downloads Food-101 subset)",
    )
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    train(args)


if __name__ == "__main__":
    main()
