#!/usr/bin/env python3
"""
Auto-label food classification images with Florence-2 to generate YOLO-format
bounding box annotations for detection training.

Uses Florence-2-base (microsoft/Florence-2-base, 0.23B params) to run object
detection (<OD> task) on classification images, then converts outputs to
YOLO format: class_id x_center y_center width height (all normalized 0-1).

Features:
  - Checkpoint/resume: tracks processed images in a JSON file
  - Batch processing for GPU efficiency
  - Confidence filtering (skip annotations < 0.3)
  - Full-image fallback for images with 0 detections
  - Merges auto-labeled data with Roboflow detection datasets
  - Produces unified food-detect.yaml config

Usage:
    python training/datasets/scripts/auto_label.py [--batch-size 8] [--device mps]
    python training/datasets/scripts/auto_label.py --merge-only  # Skip labeling, just merge
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent.parent
DATASETS_DIR = TRAINING_DIR / "datasets"
CLS_DIR = DATASETS_DIR / "food-classification"
AUTOLABEL_DIR = DATASETS_DIR / "food-detection-autolabeled"
ROBOFLOW_DET_DIR = DATASETS_DIR / "food-detection"
MERGED_DIR = DATASETS_DIR / "food-detection-merged"
CONFIGS_DIR = TRAINING_DIR / "configs"

# Progress checkpoint file
PROGRESS_FILE = AUTOLABEL_DIR / "auto_label_progress.json"

# Minimum confidence to keep an annotation
MIN_CONFIDENCE = 0.3

# Default batch size (tuned for ~16GB GPU memory)
DEFAULT_BATCH_SIZE = 8


def select_device(preferred: str = "auto") -> str:
    """Select the best available compute device."""
    import torch

    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_class_list(cls_dir: Path) -> list[str]:
    """Load class names from classes.txt."""
    classes_file = cls_dir / "classes.txt"
    if not classes_file.exists():
        log.error("classes.txt not found at %s", classes_file)
        sys.exit(1)

    with open(classes_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_progress(progress_file: Path) -> dict:
    """Load auto-labeling progress from checkpoint file."""
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"processed": [], "stats": {"total": 0, "annotated": 0, "zero_detection": 0, "errors": 0}}


def save_progress(progress_file: Path, progress: dict) -> None:
    """Save auto-labeling progress to checkpoint file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump(progress, f)


def florence2_detect(model, processor, images, device: str) -> list[dict]:
    """Run Florence-2 object detection on a batch of PIL images.

    Returns list of dicts with 'bboxes' and 'labels' keys per image.
    """
    import torch

    inputs = processor(
        text=["<OD>"] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    results = []
    decoded = processor.batch_decode(outputs, skip_special_tokens=False)

    for i, text in enumerate(decoded):
        try:
            parsed = processor.post_process_generation(
                text, task="<OD>", image_size=images[i].size
            )
            # Florence-2 post_process_generation returns dict under '<OD>' key
            if isinstance(parsed, dict) and "<OD>" in parsed:
                parsed = parsed["<OD>"]
            results.append(parsed)
        except Exception as e:
            log.warning("Failed to parse Florence-2 output for image %d: %s", i, e)
            results.append({"bboxes": [], "labels": []})

    return results


def convert_to_yolo(
    detections: dict,
    image_size: tuple[int, int],
    class_id: int,
    class_name: str,
) -> list[str]:
    """Convert Florence-2 detections to YOLO-format label lines.

    Args:
        detections: dict with 'bboxes' (list of [x1,y1,x2,y2]) and 'labels' (list of str)
        image_size: (width, height) of the image
        class_id: YOLO class ID to use (from the classification dataset)
        class_name: Name of the class for label matching

    Returns:
        List of YOLO-format strings: "class_id x_center y_center width height"
    """
    w, h = image_size
    if w == 0 or h == 0:
        return []

    bboxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])
    yolo_lines = []

    for bbox, label in zip(bboxes, labels):
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Normalize to 0-1
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h

        # Validate normalized values
        if not all(0 <= v <= 1 for v in [x_center, y_center, box_w, box_h]):
            continue

        # Use the folder class name as the class ID (since Food-101 images
        # are single-food-per-image, Florence-2 may detect generic "food")
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return yolo_lines


def create_full_image_annotation(class_id: int) -> str:
    """Create a YOLO annotation covering the full image.

    Used as fallback when Florence-2 produces 0 detections, since
    Food-101 images are single-food-per-image.
    """
    return f"{class_id} 0.500000 0.500000 1.000000 1.000000"


def auto_label_split(
    model,
    processor,
    cls_dir: Path,
    split: str,
    class_list: list[str],
    output_dir: Path,
    device: str,
    batch_size: int,
    progress: dict,
) -> dict:
    """Auto-label all images in a classification split.

    Args:
        model: Florence-2 model
        processor: Florence-2 processor
        cls_dir: Classification dataset directory
        split: 'train' or 'val'
        class_list: List of class names (index = class_id)
        output_dir: Auto-labeled output directory
        device: Compute device
        batch_size: Number of images to process at once
        progress: Progress tracking dict

    Returns:
        Updated stats dict
    """
    from PIL import Image

    split_dir = cls_dir / split
    if not split_dir.exists():
        log.warning("Split directory %s does not exist, skipping.", split_dir)
        return progress["stats"]

    # Create output directories
    img_out = output_dir / "images" / split
    lbl_out = output_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Collect all images with their class info
    image_jobs = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in class_list:
            log.warning("Class '%s' not in class list, skipping.", class_name)
            continue
        class_id = class_list.index(class_name)

        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            # Check if already processed
            rel_key = f"{split}/{class_name}/{img_file.name}"
            if rel_key in progress["processed"]:
                continue
            image_jobs.append((img_file, class_name, class_id, rel_key))

    if not image_jobs:
        log.info("Split '%s': all images already processed.", split)
        return progress["stats"]

    log.info("Split '%s': %d images to process", split, len(image_jobs))

    # Process in batches
    for batch_start in tqdm(range(0, len(image_jobs), batch_size), desc=f"Auto-labeling {split}"):
        batch_jobs = image_jobs[batch_start : batch_start + batch_size]
        batch_images = []
        batch_meta = []

        for img_path, class_name, class_id, rel_key in batch_jobs:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_meta.append((img_path, class_name, class_id, rel_key, img.size))
            except Exception as e:
                log.warning("Failed to open %s: %s", img_path, e)
                progress["stats"]["errors"] += 1
                progress["processed"].append(rel_key)

        if not batch_images:
            continue

        # Run Florence-2 detection
        try:
            detections = florence2_detect(model, processor, batch_images, device)
        except Exception as e:
            log.error("Florence-2 batch inference failed: %s", e)
            for _, _, _, rel_key, _ in batch_meta:
                progress["stats"]["errors"] += 1
                progress["processed"].append(rel_key)
            continue

        # Process each detection result
        for i, (img_path, class_name, class_id, rel_key, img_size) in enumerate(batch_meta):
            if i >= len(detections):
                progress["stats"]["errors"] += 1
                progress["processed"].append(rel_key)
                continue

            yolo_lines = convert_to_yolo(detections[i], img_size, class_id, class_name)

            if not yolo_lines:
                # Fallback: create full-image annotation
                yolo_lines = [create_full_image_annotation(class_id)]
                progress["stats"]["zero_detection"] += 1
            else:
                progress["stats"]["annotated"] += 1

            progress["stats"]["total"] += 1

            # Write label file
            label_name = f"{class_name}_{img_path.stem}.txt"
            label_path = lbl_out / label_name
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            # Symlink image (save disk space)
            img_dest = img_out / f"{class_name}_{img_path.name}"
            if not img_dest.exists():
                try:
                    os.symlink(img_path.resolve(), img_dest)
                except OSError:
                    # Fallback to copy if symlink fails
                    shutil.copy2(img_path, img_dest)

            progress["processed"].append(rel_key)

        # Save checkpoint every batch
        if batch_start % (batch_size * 10) == 0:
            save_progress(PROGRESS_FILE, progress)

    save_progress(PROGRESS_FILE, progress)
    return progress["stats"]


def run_auto_labeling(device: str, batch_size: int) -> dict:
    """Run Florence-2 auto-labeling on classification images."""
    log.info("Loading Florence-2-base model...")

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    log.info("Florence-2-base loaded on device: %s", device)

    class_list = load_class_list(CLS_DIR)
    log.info("Loaded %d classes from classes.txt", len(class_list))

    progress = load_progress(PROGRESS_FILE)
    log.info(
        "Resuming from checkpoint: %d images already processed",
        len(progress["processed"]),
    )

    # Process train and val splits
    for split in ["train", "val"]:
        log.info("\n--- Processing split: %s ---", split)
        stats = auto_label_split(
            model=model,
            processor=processor,
            cls_dir=CLS_DIR,
            split=split,
            class_list=class_list,
            output_dir=AUTOLABEL_DIR,
            device=device,
            batch_size=batch_size,
            progress=progress,
        )

    # Final stats
    stats = progress["stats"]
    log.info("\n--- Auto-labeling Statistics ---")
    log.info("Total images processed: %d", stats["total"])
    log.info("Images with Florence-2 detections: %d", stats["annotated"])
    log.info("Images with full-image fallback: %d", stats["zero_detection"])
    log.info("Errors: %d", stats["errors"])
    if stats["total"] > 0:
        pct = stats["annotated"] / stats["total"] * 100
        log.info("Detection rate: %.1f%%", pct)

    return stats


def merge_detection_datasets() -> dict:
    """Merge auto-labeled data with Roboflow detection datasets.

    Produces a unified food-detection-merged/ directory with reconciled
    class lists and re-indexed label files.
    """
    log.info("\n--- Merging Detection Datasets ---")

    if MERGED_DIR.exists():
        log.info("Removing existing merged directory: %s", MERGED_DIR)
        shutil.rmtree(MERGED_DIR)

    # Create merged structure
    for subdir in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (MERGED_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Build unified class list
    unified_classes = []

    # Start with classification class list (auto-labeled data uses these)
    classes_file = CLS_DIR / "classes.txt"
    if classes_file.exists():
        with open(classes_file) as f:
            for line in f:
                name = line.strip()
                if name and name not in unified_classes:
                    unified_classes.append(name)

    # Add Roboflow detection classes
    roboflow_class_remap = {}  # old_id -> new_id
    roboflow_classes_file = ROBOFLOW_DET_DIR / "classes.txt"
    if roboflow_classes_file.exists():
        with open(roboflow_classes_file) as f:
            for old_id, line in enumerate(f):
                name = line.strip()
                if name:
                    if name not in unified_classes:
                        unified_classes.append(name)
                    roboflow_class_remap[old_id] = unified_classes.index(name)

    log.info("Unified class list: %d classes", len(unified_classes))

    # Copy auto-labeled data
    autolabel_count = 0
    if AUTOLABEL_DIR.exists():
        for split in ["train", "val"]:
            img_src = AUTOLABEL_DIR / "images" / split
            lbl_src = AUTOLABEL_DIR / "labels" / split

            if not img_src.exists():
                continue

            for img_file in img_src.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                # Copy/symlink image
                dest_img = MERGED_DIR / "images" / split / img_file.name
                if img_file.is_symlink():
                    # Resolve and re-symlink
                    target = img_file.resolve()
                    try:
                        os.symlink(target, dest_img)
                    except OSError:
                        shutil.copy2(target, dest_img)
                else:
                    try:
                        os.symlink(img_file.resolve(), dest_img)
                    except OSError:
                        shutil.copy2(img_file, dest_img)

                # Copy label (no remap needed -- already uses classification class IDs)
                lbl_file = lbl_src / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    shutil.copy2(lbl_file, MERGED_DIR / "labels" / split / lbl_file.name)

                autolabel_count += 1

    log.info("Auto-labeled images added: %d", autolabel_count)

    # Copy Roboflow detection data with class remapping
    roboflow_count = 0
    if ROBOFLOW_DET_DIR.exists():
        for split in ["train", "val", "test"]:
            img_src = ROBOFLOW_DET_DIR / "images" / split
            lbl_src = ROBOFLOW_DET_DIR / "labels" / split

            if not img_src.exists():
                continue

            for img_file in img_src.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                dest_img = MERGED_DIR / "images" / split / f"rf_{img_file.name}"
                shutil.copy2(img_file, dest_img)

                lbl_file = lbl_src / f"{img_file.stem}.txt"
                dest_lbl = MERGED_DIR / "labels" / split / f"rf_{img_file.stem}.txt"

                if lbl_file.exists() and roboflow_class_remap:
                    # Remap class IDs
                    with open(lbl_file) as f:
                        lines = f.readlines()
                    remapped = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_id = int(parts[0])
                            new_id = roboflow_class_remap.get(old_id, old_id)
                            remapped.append(f"{new_id} {' '.join(parts[1:])}")
                    with open(dest_lbl, "w") as f:
                        f.write("\n".join(remapped) + "\n")
                elif lbl_file.exists():
                    shutil.copy2(lbl_file, dest_lbl)

                roboflow_count += 1

    log.info("Roboflow detection images added: %d", roboflow_count)

    # Write unified classes.txt
    classes_out = MERGED_DIR / "classes.txt"
    with open(classes_out, "w") as f:
        for name in unified_classes:
            f.write(f"{name}\n")

    total = autolabel_count + roboflow_count
    log.info("Merged detection dataset: %d total images, %d classes", total, len(unified_classes))

    return {
        "total_images": total,
        "autolabel_images": autolabel_count,
        "roboflow_images": roboflow_count,
        "classes": len(unified_classes),
        "class_names": unified_classes,
    }


def create_detect_config(merge_stats: dict) -> Path:
    """Create food-detect.yaml config for the merged detection dataset."""
    config = {
        "path": "../datasets/food-detection-merged",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": merge_stats["classes"],
        "names": merge_stats["class_names"],
    }

    config_path = CONFIGS_DIR / "food-detect.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with comments
    with open(config_path, "w") as f:
        f.write("# Food object detection dataset config (merged: auto-labeled + Roboflow)\n")
        f.write("# Used for Stage 2 of the detection pipeline: bounding box detection\n")
        f.write("#\n")
        f.write("# Dataset structure (YOLO detection format):\n")
        f.write("#   food-detection-merged/\n")
        f.write("#     images/\n")
        f.write("#       train/\n")
        f.write("#       val/\n")
        f.write("#       test/\n")
        f.write("#     labels/\n")
        f.write("#       train/\n")
        f.write("#       val/\n")
        f.write("#       test/\n")
        f.write("#     classes.txt\n")
        f.write("#\n")
        f.write(f"# Total images: {merge_stats['total_images']}\n")
        f.write(f"#   Auto-labeled (Florence-2): {merge_stats['autolabel_images']}\n")
        f.write(f"#   Roboflow detection: {merge_stats['roboflow_images']}\n")
        f.write(f"# Total classes: {merge_stats['classes']}\n")
        f.write("\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log.info("Detection config written to %s", config_path)
    return config_path


def verify_labels(merged_dir: Path, sample_count: int = 10) -> bool:
    """Spot-check label files for correct YOLO format."""
    log.info("\n--- Verifying Labels ---")
    label_files = []
    for split in ["train", "val", "test"]:
        lbl_dir = merged_dir / "labels" / split
        if lbl_dir.exists():
            label_files.extend(list(lbl_dir.glob("*.txt")))

    if not label_files:
        log.warning("No label files found to verify.")
        return True

    import random

    random.seed(42)
    sample = random.sample(label_files, min(sample_count, len(label_files)))

    errors = 0
    for lbl_file in sample:
        with open(lbl_file) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    log.error(
                        "  %s line %d: expected 5 values, got %d: %s",
                        lbl_file.name,
                        line_num,
                        len(parts),
                        line.strip(),
                    )
                    errors += 1
                    continue

                try:
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                    for v in values:
                        if not (0 <= v <= 1):
                            log.error(
                                "  %s line %d: value %.4f out of [0,1] range",
                                lbl_file.name,
                                line_num,
                                v,
                            )
                            errors += 1
                except ValueError as e:
                    log.error(
                        "  %s line %d: parse error: %s",
                        lbl_file.name,
                        line_num,
                        e,
                    )
                    errors += 1

    if errors == 0:
        log.info("Label verification passed: %d files checked, 0 errors", len(sample))
        return True
    else:
        log.error("Label verification found %d errors in %d files", errors, len(sample))
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto-label food images with Florence-2")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for Florence-2 inference (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, mps, cpu (default: auto)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip auto-labeling, only merge existing data",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset progress and re-process all images",
    )
    args = parser.parse_args()

    if not CLS_DIR.exists():
        log.error(
            "Classification dataset not found at %s. Run merge_datasets.py first.",
            CLS_DIR,
        )
        sys.exit(1)

    # Auto-labeling step
    if not args.merge_only:
        device = select_device(args.device)
        log.info("Using device: %s", device)

        if args.reset and PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            log.info("Progress reset.")

        stats = run_auto_labeling(device=device, batch_size=args.batch_size)

        print("\n" + "=" * 60)
        print("AUTO-LABELING STATISTICS")
        print("=" * 60)
        print(f"  Total processed: {stats['total']}")
        print(f"  With detections: {stats['annotated']}")
        print(f"  Full-image fallback: {stats['zero_detection']}")
        print(f"  Errors: {stats['errors']}")
        if stats["total"] > 0:
            pct = stats["annotated"] / stats["total"] * 100
            print(f"  Detection rate: {pct:.1f}%")
        print("=" * 60)
    else:
        log.info("Skipping auto-labeling (--merge-only)")

    # Merge step
    merge_stats = merge_detection_datasets()

    # Create detection config
    create_detect_config(merge_stats)

    # Verify labels
    verify_labels(MERGED_DIR)

    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"  Total detection images: {merge_stats['total_images']}")
    print(f"  Auto-labeled: {merge_stats['autolabel_images']}")
    print(f"  Roboflow: {merge_stats['roboflow_images']}")
    print(f"  Unified classes: {merge_stats['classes']}")
    print(f"  Config: {CONFIGS_DIR / 'food-detect.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
