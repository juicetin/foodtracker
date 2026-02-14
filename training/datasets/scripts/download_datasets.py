#!/usr/bin/env python3
"""
Download food datasets for the three-stage detection pipeline.

Downloads:
  1. Food-101 from HuggingFace (101 classes, ~101K images)
  2. ISIA Food-500 (500 classes, ~400K images) -- may be unreliable
  3. Roboflow food detection datasets (bounding box annotations)
  4. Roboflow food/not-food binary classification dataset

All datasets saved to training/datasets/raw/ with subdirectories per source.

Usage:
    python training/datasets/scripts/download_datasets.py [--skip-isia] [--roboflow-key KEY]
"""

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Resolve paths relative to the repo root (training/)
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent.parent  # training/
RAW_DIR = TRAINING_DIR / "datasets" / "raw"


def download_food101(output_dir: Path) -> dict:
    """Download Food-101 from HuggingFace datasets library."""
    food101_dir = output_dir / "food-101"
    if food101_dir.exists() and any(food101_dir.iterdir()):
        log.info("Food-101 already downloaded at %s, skipping.", food101_dir)
        # Count existing images
        image_count = sum(1 for _ in food101_dir.rglob("*.jpg"))
        image_count += sum(1 for _ in food101_dir.rglob("*.png"))
        return {"name": "Food-101", "images": image_count, "status": "cached"}

    log.info("Downloading Food-101 from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset("ethz/food101", trust_remote_code=True)
    except Exception as e:
        log.error("Failed to download Food-101: %s", e)
        return {"name": "Food-101", "images": 0, "status": f"failed: {e}"}

    food101_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0

    for split_name in ["train", "validation"]:
        if split_name not in ds:
            continue
        split = ds[split_name]
        split_dir = food101_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(split):
            label = example["label"]
            # Get the string label from the dataset features
            label_name = split.features["label"].int2str(label)
            label_dir = split_dir / label_name
            label_dir.mkdir(parents=True, exist_ok=True)

            image = example["image"]
            image_path = label_dir / f"{i:06d}.jpg"
            image.save(image_path, "JPEG")
            image_count += 1

            if image_count % 5000 == 0:
                log.info("  Saved %d images...", image_count)

    log.info("Food-101 download complete: %d images", image_count)
    return {"name": "Food-101", "images": image_count, "status": "downloaded"}


def download_isia500(output_dir: Path) -> dict:
    """Attempt to download ISIA Food-500 dataset.

    The dataset is hosted on a Chinese university server that may be unreliable.
    If download fails, we log a warning and continue.
    """
    isia_dir = output_dir / "isia-500"
    if isia_dir.exists() and any(isia_dir.iterdir()):
        log.info("ISIA-500 already present at %s, skipping.", isia_dir)
        image_count = sum(1 for _ in isia_dir.rglob("*.jpg"))
        image_count += sum(1 for _ in isia_dir.rglob("*.png"))
        return {"name": "ISIA-500", "images": image_count, "status": "cached"}

    isia_dir.mkdir(parents=True, exist_ok=True)

    log.info("Attempting ISIA Food-500 download (server may be unreliable)...")
    base_url = "http://123.57.42.89/Dataset_isia"
    test_url = f"{base_url}/ReadMe.txt"

    try:
        req = urllib.request.Request(test_url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        log.info("ISIA-500 server is reachable. Full download would take significant time.")
        log.warning(
            "ISIA-500 full download skipped in automated pipeline. "
            "To download manually, visit http://123.57.42.89/Dataset_isia/ "
            "and extract archives into %s",
            isia_dir,
        )
        return {
            "name": "ISIA-500",
            "images": 0,
            "status": "server reachable but download skipped (manual download recommended)",
        }
    except Exception as e:
        log.warning(
            "ISIA-500 server unreachable (%s). Continuing without ISIA-500. "
            "This is expected -- the server is often down.",
            e,
        )
        return {"name": "ISIA-500", "images": 0, "status": f"server unreachable: {e}"}


def download_roboflow_detection(output_dir: Path, api_key: Optional[str] = None) -> dict:
    """Download food detection datasets from Roboflow Universe with bounding box annotations."""
    det_dir = output_dir / "roboflow-detection"
    if det_dir.exists() and any(det_dir.iterdir()):
        log.info("Roboflow detection data already present at %s, skipping.", det_dir)
        image_count = sum(1 for _ in det_dir.rglob("*.jpg"))
        image_count += sum(1 for _ in det_dir.rglob("*.png"))
        return {"name": "Roboflow-Detection", "images": image_count, "status": "cached"}

    det_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        log.warning(
            "No Roboflow API key provided. Set ROBOFLOW_API_KEY environment variable "
            "or pass --roboflow-key to download detection datasets. "
            "Skipping Roboflow detection download."
        )
        return {
            "name": "Roboflow-Detection",
            "images": 0,
            "status": "skipped (no API key)",
        }

    log.info("Downloading food detection datasets from Roboflow Universe...")
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)

        # Dataset 1: A well-known food detection dataset with bounding boxes
        # "food-detection" datasets on Roboflow Universe with >1000 images
        datasets_to_try = [
            ("food-items-detection", "new", 2, "Food Items Detection"),
            ("food-detection-ow4g6", "new", 1, "Food Detection"),
            ("food-recognition-kqgi7", "new", 1, "Food Recognition"),
        ]

        total_images = 0
        downloaded = []

        for workspace_project, version_type, version_num, desc in datasets_to_try:
            try:
                log.info("  Downloading %s...", desc)
                project = rf.workspace().project(workspace_project)
                dataset = project.version(version_num).download(
                    "yolov8",
                    location=str(det_dir / workspace_project),
                )
                img_count = sum(
                    1 for _ in (det_dir / workspace_project).rglob("*.jpg")
                ) + sum(1 for _ in (det_dir / workspace_project).rglob("*.png"))
                total_images += img_count
                downloaded.append(f"{desc}: {img_count} images")
                log.info("  %s: %d images", desc, img_count)
            except Exception as e:
                log.warning("  Failed to download %s: %s", desc, e)

        status = f"downloaded ({'; '.join(downloaded)})" if downloaded else "no datasets downloaded"
        return {
            "name": "Roboflow-Detection",
            "images": total_images,
            "status": status,
        }

    except ImportError:
        log.error("roboflow package not installed. Run: pip install roboflow")
        return {"name": "Roboflow-Detection", "images": 0, "status": "failed (roboflow not installed)"}
    except Exception as e:
        log.error("Roboflow detection download failed: %s", e)
        return {"name": "Roboflow-Detection", "images": 0, "status": f"failed: {e}"}


def download_roboflow_binary(output_dir: Path, api_key: Optional[str] = None) -> dict:
    """Download food/not-food binary classification dataset from Roboflow Universe."""
    binary_dir = output_dir / "roboflow-binary"
    if binary_dir.exists() and any(binary_dir.iterdir()):
        log.info("Roboflow binary data already present at %s, skipping.", binary_dir)
        image_count = sum(1 for _ in binary_dir.rglob("*.jpg"))
        image_count += sum(1 for _ in binary_dir.rglob("*.png"))
        return {"name": "Roboflow-Binary", "images": image_count, "status": "cached"}

    binary_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        log.warning(
            "No Roboflow API key provided. Skipping binary dataset download. "
            "Will create a synthetic binary dataset from Food-101 in the merge step."
        )
        return {
            "name": "Roboflow-Binary",
            "images": 0,
            "status": "skipped (no API key -- will synthesize from Food-101)",
        }

    log.info("Downloading food/not-food binary dataset from Roboflow...")
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)

        datasets_to_try = [
            ("food-not-food", "new", 1, "Food Not Food"),
            ("food-classification-ij5pt", "new", 1, "Food Classification Binary"),
        ]

        total_images = 0
        downloaded = []

        for workspace_project, version_type, version_num, desc in datasets_to_try:
            try:
                log.info("  Downloading %s...", desc)
                project = rf.workspace().project(workspace_project)
                dataset = project.version(version_num).download(
                    "folder",
                    location=str(binary_dir / workspace_project),
                )
                img_count = sum(
                    1 for _ in (binary_dir / workspace_project).rglob("*.jpg")
                ) + sum(1 for _ in (binary_dir / workspace_project).rglob("*.png"))
                total_images += img_count
                downloaded.append(f"{desc}: {img_count} images")
            except Exception as e:
                log.warning("  Failed to download %s: %s", desc, e)

        status = f"downloaded ({'; '.join(downloaded)})" if downloaded else "no datasets downloaded"
        return {"name": "Roboflow-Binary", "images": total_images, "status": status}

    except ImportError:
        log.error("roboflow package not installed. Run: pip install roboflow")
        return {"name": "Roboflow-Binary", "images": 0, "status": "failed (roboflow not installed)"}
    except Exception as e:
        log.error("Roboflow binary download failed: %s", e)
        return {"name": "Roboflow-Binary", "images": 0, "status": f"failed: {e}"}


def print_summary(results: list[dict]) -> None:
    """Print download summary."""
    print("\n" + "=" * 60)
    print("DATASET DOWNLOAD SUMMARY")
    print("=" * 60)
    total_images = 0
    for r in results:
        total_images += r["images"]
        print(f"  {r['name']:25s} {r['images']:>8d} images  [{r['status']}]")
    print("-" * 60)
    print(f"  {'TOTAL':25s} {total_images:>8d} images")
    print("=" * 60)

    # Save summary to JSON for downstream scripts
    summary_path = RAW_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"datasets": results, "total_images": total_images}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Download food training datasets")
    parser.add_argument(
        "--skip-isia",
        action="store_true",
        help="Skip ISIA Food-500 download attempt",
    )
    parser.add_argument(
        "--roboflow-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_DIR),
        help=f"Output directory (default: {RAW_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Food-101 (guaranteed, from HuggingFace)
    results.append(download_food101(output_dir))

    # 2. ISIA-500 (may fail, that's OK)
    if not args.skip_isia:
        results.append(download_isia500(output_dir))
    else:
        log.info("Skipping ISIA-500 download (--skip-isia flag)")
        results.append({"name": "ISIA-500", "images": 0, "status": "skipped (--skip-isia)"})

    # 3. Roboflow detection datasets (need API key)
    results.append(download_roboflow_detection(output_dir, api_key=args.roboflow_key))

    # 4. Roboflow binary dataset (need API key)
    results.append(download_roboflow_binary(output_dir, api_key=args.roboflow_key))

    print_summary(results)

    # Exit with warning if no detection data was obtained
    has_detection = any(
        r["name"] == "Roboflow-Detection" and r["images"] > 0 for r in results
    )
    if not has_detection:
        log.warning(
            "No detection datasets with bounding boxes were downloaded. "
            "The auto-label step (auto_label.py) will generate bounding boxes "
            "from classification images using Florence-2."
        )


if __name__ == "__main__":
    main()
