#!/usr/bin/env python3
"""
Merge downloaded food datasets into unified directory structures for YOLO training.

Produces three output datasets:
  1. food-classification/ -- ImageNet-style folders with train/val/test splits
  2. food-detection/      -- YOLO format (images/ + labels/) from Roboflow sources
  3. food-binary/         -- food/ and not-food/ folders with train/val/test splits

Usage:
    python training/datasets/scripts/merge_datasets.py [--raw-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = TRAINING_DIR / "datasets" / "raw"
DATASETS_DIR = TRAINING_DIR / "datasets"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed for reproducible splits
RANDOM_SEED = 42


def normalize_class_name(name: str) -> str:
    """Convert class name to consistent lowercase-hyphenated format.

    Examples:
        "Pad Thai" -> "pad-thai"
        "fried_rice" -> "fried-rice"
        "French Fries" -> "french-fries"
        "baby_back_ribs" -> "baby-back-ribs"
    """
    name = name.lower().strip()
    # Replace underscores and multiple spaces with single hyphen
    name = re.sub(r"[_\s]+", "-", name)
    # Remove non-alphanumeric characters except hyphens
    name = re.sub(r"[^a-z0-9\-]", "", name)
    # Collapse multiple hyphens
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def split_files(files: list, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO) -> dict:
    """Split a list of files into train/val/test sets."""
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }


def merge_classification(raw_dir: Path, output_dir: Path) -> dict:
    """Merge Food-101 (+ ISIA-500 if available) into ImageNet-style classification structure.

    Output structure:
        food-classification/
            train/
                pad-thai/
                    img001.jpg
                sushi/
                    img001.jpg
            val/
            test/
            classes.txt
    """
    cls_dir = output_dir / "food-classification"
    if cls_dir.exists():
        log.info("Classification directory already exists at %s. Removing for fresh merge.", cls_dir)
        shutil.rmtree(cls_dir)

    cls_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (cls_dir / split).mkdir(exist_ok=True)

    random.seed(RANDOM_SEED)

    # Collect all class images from all sources
    class_images: dict[str, list[Path]] = {}

    # --- Food-101 ---
    food101_dir = raw_dir / "food-101"
    if food101_dir.exists():
        log.info("Processing Food-101 from %s", food101_dir)
        for split_dir in food101_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                cls_name = normalize_class_name(class_dir.name)
                if cls_name not in class_images:
                    class_images[cls_name] = []
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                class_images[cls_name].extend(images)
        log.info("  Food-101: %d classes found", len(class_images))
    else:
        log.warning("Food-101 not found at %s", food101_dir)

    # --- ISIA-500 (if available) ---
    isia_dir = raw_dir / "isia-500"
    if isia_dir.exists() and any(isia_dir.iterdir()):
        log.info("Processing ISIA-500 from %s", isia_dir)
        isia_classes_before = len(class_images)
        for class_dir in isia_dir.rglob("*"):
            if not class_dir.is_dir():
                continue
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if not images:
                continue
            cls_name = normalize_class_name(class_dir.name)
            if cls_name not in class_images:
                class_images[cls_name] = []
            class_images[cls_name].extend(images)
        log.info(
            "  ISIA-500: %d new classes added",
            len(class_images) - isia_classes_before,
        )
    else:
        log.info("ISIA-500 not available, continuing with Food-101 only.")

    # Sort classes for deterministic ordering
    all_classes = sorted(class_images.keys())
    log.info("Total classification classes: %d", len(all_classes))

    # Write classes.txt
    classes_file = cls_dir / "classes.txt"
    with open(classes_file, "w") as f:
        for cls_name in all_classes:
            f.write(f"{cls_name}\n")
    log.info("Class list written to %s", classes_file)

    # Split and copy images
    total_images = 0
    stats = {"train": 0, "val": 0, "test": 0}

    for cls_name in tqdm(all_classes, desc="Merging classification classes"):
        images = class_images[cls_name]
        if len(images) < 3:
            # Too few images for a meaningful split, put all in train
            splits = {"train": images, "val": [], "test": []}
        else:
            splits = split_files(images)

        for split_name, split_images in splits.items():
            target_dir = cls_dir / split_name / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for i, img_path in enumerate(split_images):
                suffix = img_path.suffix or ".jpg"
                target_path = target_dir / f"{cls_name}_{i:06d}{suffix}"
                shutil.copy2(img_path, target_path)
                stats[split_name] += 1
                total_images += 1

    log.info(
        "Classification merge complete: %d images (%d train, %d val, %d test) across %d classes",
        total_images,
        stats["train"],
        stats["val"],
        stats["test"],
        len(all_classes),
    )

    return {
        "classes": len(all_classes),
        "total_images": total_images,
        "splits": stats,
        "classes_file": str(classes_file),
    }


def merge_detection(raw_dir: Path, output_dir: Path) -> dict:
    """Merge Roboflow detection datasets into unified YOLO-format directory.

    Output structure:
        food-detection/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/
    """
    det_dir = output_dir / "food-detection"
    if det_dir.exists():
        log.info("Detection directory already exists at %s. Removing for fresh merge.", det_dir)
        shutil.rmtree(det_dir)

    # Create directory structure
    for subdir in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (det_dir / subdir).mkdir(parents=True, exist_ok=True)

    rf_det_dir = raw_dir / "roboflow-detection"
    if not rf_det_dir.exists() or not any(rf_det_dir.iterdir()):
        log.warning(
            "No Roboflow detection data found at %s. "
            "Detection dataset will be populated by auto_label.py.",
            rf_det_dir,
        )
        return {"classes": 0, "total_images": 0, "splits": {"train": 0, "val": 0, "test": 0}}

    # Collect all class names across Roboflow detection datasets
    unified_classes = []
    class_remap = {}  # (dataset_name, old_id) -> new_id

    log.info("Scanning Roboflow detection datasets for class lists...")
    dataset_dirs = [d for d in rf_det_dir.iterdir() if d.is_dir()]

    for ds_dir in dataset_dirs:
        # Look for data.yaml or similar config
        yaml_files = list(ds_dir.rglob("*.yaml")) + list(ds_dir.rglob("*.yml"))
        for yaml_file in yaml_files:
            try:
                import yaml

                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                if config and "names" in config:
                    names = config["names"]
                    if isinstance(names, dict):
                        names = [names[k] for k in sorted(names.keys())]
                    for old_id, name in enumerate(names):
                        norm_name = normalize_class_name(name)
                        if norm_name not in unified_classes:
                            unified_classes.append(norm_name)
                        new_id = unified_classes.index(norm_name)
                        class_remap[(ds_dir.name, old_id)] = new_id
                    log.info("  %s: %d classes", ds_dir.name, len(names))
                    break
            except Exception as e:
                log.warning("  Could not parse %s: %s", yaml_file, e)

    log.info("Unified detection classes: %d", len(unified_classes))

    # Copy images and remap labels
    total_images = 0
    stats = {"train": 0, "val": 0, "test": 0}

    for ds_dir in dataset_dirs:
        for split in ["train", "valid", "val", "test"]:
            # Roboflow uses "valid" sometimes
            target_split = "val" if split == "valid" else split

            img_dir = ds_dir / split / "images"
            lbl_dir = ds_dir / split / "labels"

            if not img_dir.exists():
                # Try alternative structure
                img_dir = ds_dir / "images" / split
                lbl_dir = ds_dir / "labels" / split

            if not img_dir.exists():
                continue

            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue

                # Copy image
                dest_img = det_dir / "images" / target_split / f"{ds_dir.name}_{img_file.name}"
                shutil.copy2(img_file, dest_img)

                # Copy and remap label
                label_file = lbl_dir / f"{img_file.stem}.txt"
                dest_label = det_dir / "labels" / target_split / f"{ds_dir.name}_{img_file.stem}.txt"

                if label_file.exists():
                    with open(label_file) as f:
                        lines = f.readlines()
                    remapped_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_id = int(parts[0])
                            new_id = class_remap.get((ds_dir.name, old_id), old_id)
                            remapped_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                    with open(dest_label, "w") as f:
                        f.writelines(remapped_lines)
                else:
                    # Create empty label file
                    dest_label.touch()

                stats[target_split] += 1
                total_images += 1

    # Write unified class list
    classes_file = det_dir / "classes.txt"
    with open(classes_file, "w") as f:
        for cls_name in unified_classes:
            f.write(f"{cls_name}\n")

    log.info(
        "Detection merge complete: %d images (%d train, %d val, %d test) across %d classes",
        total_images,
        stats["train"],
        stats["val"],
        stats["test"],
        len(unified_classes),
    )

    return {
        "classes": len(unified_classes),
        "total_images": total_images,
        "splits": stats,
        "class_names": unified_classes,
    }


def merge_binary(raw_dir: Path, output_dir: Path) -> dict:
    """Create binary food/not-food dataset with train/val/test splits.

    If Roboflow binary dataset available, use it.
    Otherwise, synthesize from Food-101 (food) and create a not-food set
    from a small curated set or placeholder.

    Output structure:
        food-binary/
            train/
                food/
                not-food/
            val/
                food/
                not-food/
            test/
                food/
                not-food/
    """
    binary_dir = output_dir / "food-binary"
    if binary_dir.exists():
        log.info("Binary directory already exists at %s. Removing for fresh merge.", binary_dir)
        shutil.rmtree(binary_dir)

    # Create structure
    for split in ["train", "val", "test"]:
        for cls in ["food", "not-food"]:
            (binary_dir / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    # Try Roboflow binary dataset first
    rf_binary_dir = raw_dir / "roboflow-binary"
    has_roboflow_binary = rf_binary_dir.exists() and any(rf_binary_dir.iterdir())

    food_images: list[Path] = []
    not_food_images: list[Path] = []

    if has_roboflow_binary:
        log.info("Using Roboflow binary dataset from %s", rf_binary_dir)
        for img_file in rf_binary_dir.rglob("*"):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            # Try to infer class from directory name
            parent_name = img_file.parent.name.lower()
            if "not" in parent_name or "non" in parent_name:
                not_food_images.append(img_file)
            elif "food" in parent_name:
                food_images.append(img_file)

    # Supplement with Food-101 for food class if needed
    food101_dir = raw_dir / "food-101"
    if food101_dir.exists() and len(food_images) < 1000:
        log.info("Supplementing food class from Food-101...")
        food101_images = list(food101_dir.rglob("*.jpg")) + list(food101_dir.rglob("*.png"))
        random.shuffle(food101_images)
        # Take a balanced subset -- use up to 5000 images from Food-101 for binary
        needed = max(5000 - len(food_images), 0)
        food_images.extend(food101_images[:needed])

    # If no not-food images, create a placeholder set
    if len(not_food_images) < 100:
        log.warning(
            "Insufficient not-food images (%d). "
            "For production training, add not-food images to %s/not-food/. "
            "Creating placeholder structure for now.",
            len(not_food_images),
            binary_dir,
        )
        # Create a small synthetic not-food set with solid color images
        try:
            from PIL import Image

            log.info("Generating %d synthetic not-food placeholder images...", 1000)
            for i in range(1000):
                # Generate random colored images as not-food placeholders
                img = Image.new(
                    "RGB",
                    (224, 224),
                    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                )
                placeholder_path = binary_dir / "synthetic_not_food" / f"synthetic_{i:04d}.jpg"
                placeholder_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(placeholder_path, "JPEG")
                not_food_images.append(placeholder_path)
        except ImportError:
            log.error("Pillow not installed, cannot create synthetic not-food images.")

    log.info("Binary dataset: %d food images, %d not-food images", len(food_images), len(not_food_images))

    # Split and copy
    food_splits = split_files(food_images)
    not_food_splits = split_files(not_food_images)

    stats = {"train": 0, "val": 0, "test": 0}
    for split_name in ["train", "val", "test"]:
        for i, img_path in enumerate(food_splits[split_name]):
            suffix = img_path.suffix or ".jpg"
            dest = binary_dir / split_name / "food" / f"food_{i:06d}{suffix}"
            shutil.copy2(img_path, dest)
            stats[split_name] += 1

        for i, img_path in enumerate(not_food_splits[split_name]):
            suffix = img_path.suffix or ".jpg"
            dest = binary_dir / split_name / "not-food" / f"not_food_{i:06d}{suffix}"
            shutil.copy2(img_path, dest)
            stats[split_name] += 1

    total_images = sum(stats.values())
    log.info(
        "Binary merge complete: %d images (%d train, %d val, %d test)",
        total_images,
        stats["train"],
        stats["val"],
        stats["test"],
    )

    # Clean up synthetic directory if created
    synthetic_dir = binary_dir / "synthetic_not_food"
    if synthetic_dir.exists():
        shutil.rmtree(synthetic_dir)

    return {
        "food_images": len(food_images),
        "not_food_images": len(not_food_images),
        "total_images": total_images,
        "splits": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge food datasets into unified structures")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DIR),
        help=f"Raw dataset directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATASETS_DIR),
        help=f"Output directory (default: {DATASETS_DIR})",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        log.error(
            "Raw dataset directory %s does not exist. "
            "Run download_datasets.py first.",
            raw_dir,
        )
        sys.exit(1)

    log.info("Merging datasets from %s into %s", raw_dir, output_dir)

    # 1. Classification dataset
    log.info("\n--- Merging Classification Dataset ---")
    cls_result = merge_classification(raw_dir, output_dir)

    # 2. Detection dataset (from Roboflow sources)
    log.info("\n--- Merging Detection Dataset ---")
    det_result = merge_detection(raw_dir, output_dir)

    # 3. Binary food/not-food dataset
    log.info("\n--- Merging Binary Dataset ---")
    binary_result = merge_binary(raw_dir, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("DATASET MERGE SUMMARY")
    print("=" * 60)
    print(f"  Classification: {cls_result.get('classes', 0)} classes, {cls_result.get('total_images', 0)} images")
    print(f"    Splits: {cls_result.get('splits', {})}")
    print(f"  Detection: {det_result.get('classes', 0)} classes, {det_result.get('total_images', 0)} images")
    print(f"    Splits: {det_result.get('splits', {})}")
    print(f"  Binary: {binary_result.get('total_images', 0)} images")
    print(f"    Food: {binary_result.get('food_images', 0)}, Not-food: {binary_result.get('not_food_images', 0)}")
    print(f"    Splits: {binary_result.get('splits', {})}")
    print("=" * 60)

    # Save merge report
    report = {
        "classification": cls_result,
        "detection": det_result,
        "binary": binary_result,
    }
    report_path = output_dir / "merge_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nMerge report saved to {report_path}")


if __name__ == "__main__":
    import sys

    main()
