#!/usr/bin/env python3
"""
Unified benchmark: YOLO detection pipeline vs PaliGemma 2 3B on identical test images.

This script compares:
  - YOLO three-stage pipeline (binary gate -> detection -> classification)
  - PaliGemma 2 3B (quantized) for food detection and identification

Output: training/evaluate/benchmark_report.md

Usage:
    python training/benchmark.py [--test-dir PATH] [--max-images N] [--yolo-dir PATH]

Notes:
    - YOLO models may not be trained yet (01-03 plan runs in parallel).
      If models are missing, YOLO benchmark is skipped gracefully.
    - PaliGemma 2 3B requires ~6GB VRAM (FP16) or ~3GB (INT8).
      Falls back to Florence-2 if PaliGemma cannot load.
    - If no test images exist, generates a synthetic test set for structural validation
      and documents that real benchmarking requires dataset download.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
TRAINING_DIR = Path(__file__).parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_TEST_DIR = TRAINING_DIR / "datasets" / "food-detection-merged" / "images" / "test"
DEFAULT_YOLO_DIR = TRAINING_DIR / "runs" / "detect" / "food-detect" / "weights"
REPORT_PATH = TRAINING_DIR / "evaluate" / "benchmark_report.md"

# Cuisine mapping for per-cuisine breakdown
CUISINE_KEYWORDS = {
    "Western": [
        "pizza", "burger", "steak", "pasta", "sandwich", "salad", "soup",
        "bread", "pancake", "waffle", "omelette", "fries", "hot-dog",
        "grilled-cheese", "mac-and-cheese", "meatloaf", "roast",
        "fish-and-chips", "pie", "sausage", "bacon", "cereal",
    ],
    "Chinese": [
        "fried-rice", "dim-sum", "dumpling", "wonton", "chow-mein",
        "kung-pao", "mapo-tofu", "spring-roll", "peking-duck",
        "sweet-and-sour", "char-siu", "congee", "bao", "noodle-soup",
        "hot-pot", "stir-fry",
    ],
    "Japanese": [
        "sushi", "ramen", "tempura", "sashimi", "udon", "miso",
        "onigiri", "yakitori", "tonkatsu", "gyoza", "okonomiyaki",
        "takoyaki", "matcha", "mochi", "teriyaki",
    ],
    "Korean": [
        "bibimbap", "kimchi", "bulgogi", "tteokbokki", "japchae",
        "samgyeopsal", "kimbap", "sundubu", "galbi", "jjigae",
    ],
    "Vietnamese": [
        "pho", "banh-mi", "spring-roll", "bun-cha", "com-tam",
        "cao-lau", "bun-bo-hue", "goi-cuon",
    ],
    "Thai": [
        "pad-thai", "green-curry", "tom-yum", "som-tam", "massaman",
        "larb", "khao-pad", "pad-see-ew", "mango-sticky-rice",
    ],
}


def classify_cuisine(label: str) -> str:
    """Classify a food label into a cuisine group."""
    label_lower = label.lower().replace("_", "-").replace(" ", "-")
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        for kw in keywords:
            if kw in label_lower:
                return cuisine
    return "Other"


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detection from either YOLO or VLM."""
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    label: str
    confidence: float
    source: str  # "yolo" or "vlm"


@dataclass
class ImageResult:
    """Benchmark result for a single image."""
    image_path: str
    ground_truth_labels: list
    ground_truth_bboxes: list  # list of (x1,y1,x2,y2)
    yolo_detections: list = field(default_factory=list)
    yolo_time_ms: float = 0.0
    yolo_binary_time_ms: float = 0.0
    yolo_detect_time_ms: float = 0.0
    yolo_classify_time_ms: float = 0.0
    vlm_detections: list = field(default_factory=list)
    vlm_time_ms: float = 0.0
    vlm_raw_response: str = ""
    cuisine: str = "Other"


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    yolo_available: bool = False
    vlm_available: bool = False
    vlm_model_name: str = ""
    total_images: int = 0
    image_results: list = field(default_factory=list)
    yolo_model_sizes_mb: dict = field(default_factory=dict)
    vlm_model_size_mb: float = 0.0
    test_set_source: str = ""  # "real" or "synthetic"
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Test set preparation
# ---------------------------------------------------------------------------

def discover_test_images(test_dir: Path, max_images: int = 500) -> list:
    """
    Find test images and their ground truth labels from YOLO-format dataset.

    Returns list of dicts: {image_path, labels, bboxes, cuisine}
    """
    images = []
    if not test_dir.exists():
        logger.warning(f"Test directory not found: {test_dir}")
        return images

    label_dir = test_dir.parent.parent / "labels" / "test"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        f for f in test_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    if not image_files:
        logger.warning(f"No images found in {test_dir}")
        return images

    for img_path in image_files[:max_images]:
        label_file = label_dir / (img_path.stem + ".txt")
        labels = []
        bboxes = []

        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # YOLO format: class x_center y_center width height (normalized)
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        labels.append(str(cls_id))
                        bboxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))

        cuisine = classify_cuisine(img_path.stem) if labels else "Other"

        images.append({
            "image_path": str(img_path),
            "labels": labels,
            "bboxes": bboxes,
            "cuisine": cuisine,
        })

    logger.info(f"Discovered {len(images)} test images in {test_dir}")
    return images


def generate_synthetic_test_set(output_dir: Path, count: int = 100) -> list:
    """
    Generate synthetic test images for structural validation when real images
    are not available. Each image is a colored rectangle with known ground truth.

    Returns list of dicts: {image_path, labels, bboxes, cuisine}
    """
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir = output_dir.parent / "labels" / "synthetic_test"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Sample food labels for synthetic images
    food_labels = [
        ("fried-rice", "Chinese"), ("pizza", "Western"), ("sushi", "Japanese"),
        ("pad-thai", "Thai"), ("pho", "Vietnamese"), ("bibimbap", "Korean"),
        ("burger", "Western"), ("ramen", "Japanese"), ("dumpling", "Chinese"),
        ("green-curry", "Thai"), ("salad", "Western"), ("tempura", "Japanese"),
        ("kimchi", "Korean"), ("banh-mi", "Vietnamese"), ("pasta", "Western"),
        ("stir-fry", "Chinese"), ("tonkatsu", "Japanese"), ("bulgogi", "Korean"),
        ("tom-yum", "Thai"), ("spring-roll", "Vietnamese"),
    ]

    # Also generate non-food images
    non_food_labels = [
        "desk", "laptop", "book", "car", "tree", "building", "phone",
        "shoes", "bag", "cup-empty",
    ]

    images = []
    rng = np.random.RandomState(42)

    # Food images (80%)
    food_count = int(count * 0.8)
    for i in range(food_count):
        food_label, cuisine = food_labels[i % len(food_labels)]
        img = Image.new("RGB", (640, 640), color=tuple(rng.randint(50, 200, 3).tolist()))
        draw = ImageDraw.Draw(img)

        # Draw 1-3 colored rectangles as "food items"
        n_items = rng.randint(1, 4)
        bboxes = []
        labels = []
        for _ in range(n_items):
            x1 = rng.randint(50, 350)
            y1 = rng.randint(50, 350)
            x2 = x1 + rng.randint(100, 250)
            y2 = y1 + rng.randint(100, 250)
            x2 = min(x2, 630)
            y2 = min(y2, 630)
            color = tuple(rng.randint(100, 255, 3).tolist())
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="white", width=2)
            bboxes.append((x1/640, y1/640, x2/640, y2/640))  # normalized
            labels.append(food_label)

        fname = f"synthetic_food_{i:04d}_{food_label}.jpg"
        img_path = output_dir / fname
        img.save(img_path, "JPEG", quality=85)

        # Write YOLO-format label
        with open(label_dir / (Path(fname).stem + ".txt"), "w") as f:
            for bbox in bboxes:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        images.append({
            "image_path": str(img_path),
            "labels": labels,
            "bboxes": bboxes,
            "cuisine": cuisine,
        })

    # Non-food images (20%)
    non_food_count = count - food_count
    for i in range(non_food_count):
        img = Image.new("RGB", (640, 640), color=tuple(rng.randint(0, 100, 3).tolist()))
        draw = ImageDraw.Draw(img)
        # Draw geometric shapes (non-food)
        for _ in range(rng.randint(2, 6)):
            x1 = rng.randint(0, 500)
            y1 = rng.randint(0, 500)
            x2 = x1 + rng.randint(20, 140)
            y2 = y1 + rng.randint(20, 140)
            color = tuple(rng.randint(0, 150, 3).tolist())
            draw.rectangle([x1, y1, x2, y2], fill=color)

        nf_label = non_food_labels[i % len(non_food_labels)]
        fname = f"synthetic_nonfood_{i:04d}_{nf_label}.jpg"
        img_path = output_dir / fname
        img.save(img_path, "JPEG", quality=85)

        images.append({
            "image_path": str(img_path),
            "labels": [],  # no food
            "bboxes": [],
            "cuisine": "non-food",
        })

    logger.info(f"Generated {len(images)} synthetic test images in {output_dir}")
    return images


# ---------------------------------------------------------------------------
# YOLO Pipeline Benchmark
# ---------------------------------------------------------------------------

def benchmark_yolo_pipeline(test_images: list, yolo_dir: Path) -> tuple:
    """
    Run the three-stage YOLO pipeline on test images.

    Returns (results_list, model_sizes_dict, available_bool).
    """
    binary_path = TRAINING_DIR / "runs" / "classify" / "food-binary" / "weights" / "best.pt"
    detect_path = yolo_dir / "best.pt"
    classify_path = TRAINING_DIR / "runs" / "classify" / "food-classify" / "weights" / "best.pt"

    # Check which models exist
    models_status = {
        "binary": binary_path.exists(),
        "detect": detect_path.exists(),
        "classify": classify_path.exists(),
    }

    logger.info(f"YOLO model availability: {models_status}")

    if not any(models_status.values()):
        logger.warning(
            "No YOLO models found. Plan 01-03 (YOLO training) may still be running. "
            "Skipping YOLO benchmark."
        )
        return [], {}, False

    # Try to load available models
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Cannot run YOLO benchmark.")
        return [], {}, False

    binary_model = None
    detect_model = None
    classify_model = None
    model_sizes = {}

    if models_status["binary"]:
        try:
            binary_model = YOLO(str(binary_path))
            model_sizes["binary"] = binary_path.stat().st_size / (1024 * 1024)
            logger.info(f"Loaded binary model: {binary_path} ({model_sizes['binary']:.1f} MB)")
        except Exception as e:
            logger.warning(f"Failed to load binary model: {e}")

    if models_status["detect"]:
        try:
            detect_model = YOLO(str(detect_path))
            model_sizes["detect"] = detect_path.stat().st_size / (1024 * 1024)
            logger.info(f"Loaded detect model: {detect_path} ({model_sizes['detect']:.1f} MB)")
        except Exception as e:
            logger.warning(f"Failed to load detect model: {e}")

    if models_status["classify"]:
        try:
            classify_model = YOLO(str(classify_path))
            model_sizes["classify"] = classify_path.stat().st_size / (1024 * 1024)
            logger.info(f"Loaded classify model: {classify_path} ({model_sizes['classify']:.1f} MB)")
        except Exception as e:
            logger.warning(f"Failed to load classify model: {e}")

    if not (binary_model or detect_model or classify_model):
        logger.warning("Could not load any YOLO models.")
        return [], {}, False

    results = []
    for img_info in test_images:
        img_path = img_info["image_path"]
        detections = []
        binary_time = 0.0
        detect_time = 0.0
        classify_time = 0.0

        # Stage 1: Binary gate
        is_food = True
        if binary_model:
            t0 = time.perf_counter()
            try:
                result = binary_model.predict(img_path, verbose=False)
                probs = result[0].probs
                # Assume class 0 = food, class 1 = not-food (or vice versa)
                top_class = probs.top1
                is_food = top_class == 0  # adjust based on training class order
            except Exception as e:
                logger.debug(f"Binary prediction failed for {img_path}: {e}")
            binary_time = (time.perf_counter() - t0) * 1000

        # Stage 2: Detection
        if is_food and detect_model:
            t0 = time.perf_counter()
            try:
                result = detect_model.predict(img_path, conf=0.25, verbose=False)
                boxes = result[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        label = result[0].names.get(cls_id, str(cls_id))

                        # Stage 3: Classification per crop
                        if classify_model:
                            t_cls = time.perf_counter()
                            try:
                                from PIL import Image
                                img = Image.open(img_path)
                                x1, y1, x2, y2 = [int(v) for v in xyxy]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2 = min(img.width, x2)
                                y2 = min(img.height, y2)
                                crop = img.crop((x1, y1, x2, y2))
                                # Save crop to temp file for prediction
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                                    crop.save(tmp, "JPEG")
                                    tmp_path = tmp.name
                                cls_result = classify_model.predict(tmp_path, verbose=False)
                                os.unlink(tmp_path)
                                cls_top = cls_result[0].probs.top1
                                label = cls_result[0].names.get(cls_top, label)
                                conf = float(cls_result[0].probs.top1conf.item())
                            except Exception as e:
                                logger.debug(f"Classification failed for crop: {e}")
                            classify_time += (time.perf_counter() - t_cls) * 1000

                        detections.append(Detection(
                            bbox=tuple(xyxy.tolist()),
                            label=label,
                            confidence=conf,
                            source="yolo",
                        ))
            except Exception as e:
                logger.debug(f"Detection failed for {img_path}: {e}")
            detect_time = (time.perf_counter() - t0) * 1000 - classify_time

        total_time = binary_time + detect_time + classify_time
        results.append({
            "image_path": img_path,
            "detections": detections,
            "total_time_ms": total_time,
            "binary_time_ms": binary_time,
            "detect_time_ms": detect_time,
            "classify_time_ms": classify_time,
        })

    return results, model_sizes, True


# ---------------------------------------------------------------------------
# VLM Benchmark
# ---------------------------------------------------------------------------

def parse_vlm_food_detections(response: str, image_size: tuple = (640, 640)) -> list:
    """
    Parse VLM text response into structured detections.
    Handles various response formats robustly.
    """
    detections = []
    w, h = image_size

    # Pattern 1: "food_name (confidence%) at [x1, y1, x2, y2]"
    pattern1 = re.compile(
        r"([A-Za-z\s\-]+)\s*\(?\s*(\d+(?:\.\d+)?)\s*%?\s*\)?\s*(?:at|@|:)?\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?"
    )

    # Pattern 2: "[x1, y1, x2, y2] food_name confidence"
    pattern2 = re.compile(
        r"\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*([A-Za-z\s\-]+?)\s+(\d+(?:\.\d+)?)\s*%?"
    )

    # Pattern 3: Numbered list "1. food_name"
    pattern3 = re.compile(
        r"\d+\.\s*([A-Za-z\s\-]+?)(?:\s*[-:]\s*(.+?))?$",
        re.MULTILINE,
    )

    # Pattern 4: JSON-like {"label": "...", "bbox": [...]}
    pattern4 = re.compile(
        r'"label"\s*:\s*"([^"]+)".*?"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    )

    # Try each pattern
    for match in pattern1.finditer(response):
        label = match.group(1).strip()
        conf = float(match.group(2))
        if conf > 1:
            conf = conf / 100.0
        x1, y1, x2, y2 = int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6))
        detections.append(Detection(
            bbox=(x1, y1, x2, y2), label=label, confidence=conf, source="vlm",
        ))

    if detections:
        return detections

    for match in pattern2.finditer(response):
        x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        label = match.group(5).strip()
        conf = float(match.group(6))
        if conf > 1:
            conf = conf / 100.0
        detections.append(Detection(
            bbox=(x1, y1, x2, y2), label=label, confidence=conf, source="vlm",
        ))

    if detections:
        return detections

    for match in pattern4.finditer(response):
        label = match.group(1)
        x1, y1, x2, y2 = int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
        detections.append(Detection(
            bbox=(x1, y1, x2, y2), label=label, confidence=0.8, source="vlm",
        ))

    if detections:
        return detections

    # Fallback: numbered list (no bboxes, just labels)
    for match in pattern3.finditer(response):
        label = match.group(1).strip()
        if len(label) > 2 and label.lower() not in ("the", "and", "for", "with"):
            # Generate a center bounding box as placeholder
            cx, cy = w // 2, h // 2
            bw, bh = w // 3, h // 3
            detections.append(Detection(
                bbox=(cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2),
                label=label,
                confidence=0.5,
                source="vlm",
            ))

    return detections


def try_load_vlm():
    """
    Try loading VLM models in order of preference:
    1. PaliGemma 2 3B
    2. PaliGemma 2 1B
    3. Florence-2 base

    Returns (model, processor, model_name, model_size_mb) or (None, None, "", 0).
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        logger.error("transformers not installed. Cannot run VLM benchmark.")
        return None, None, "", 0

    device = "cpu"
    dtype = None

    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            logger.info("CUDA GPU available for VLM inference")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
            logger.info("Apple MPS available for VLM inference")
        else:
            logger.info("No GPU available; VLM will run on CPU (slow but accuracy is what matters)")
    except Exception:
        pass

    # Attempt 1: PaliGemma 2 3B
    vlm_attempts = [
        ("google/paligemma2-3b-pt-224", "PaliGemma 2 3B", "paligemma"),
        ("google/paligemma2-3b-224", "PaliGemma 2 3B (alt)", "paligemma"),
        ("google/paligemma-3b-pt-224", "PaliGemma 3B", "paligemma"),
    ]

    # Attempt 2: Smaller PaliGemma
    vlm_attempts.extend([
        ("google/paligemma2-1b-pt-224", "PaliGemma 2 1B", "paligemma"),
    ])

    # Attempt 3: Florence-2
    vlm_attempts.extend([
        ("microsoft/Florence-2-base", "Florence-2 base", "florence"),
    ])

    for model_id, name, family in vlm_attempts:
        logger.info(f"Attempting to load VLM: {name} ({model_id})...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            load_kwargs = {
                "trust_remote_code": True,
            }

            if device == "cuda" and dtype is not None:
                # Try INT8 quantization first for large models
                if "3b" in model_id.lower():
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        load_kwargs["quantization_config"] = quantization_config
                        load_kwargs["device_map"] = "auto"
                        logger.info(f"Using INT8 quantization for {name}")
                    except ImportError:
                        load_kwargs["torch_dtype"] = dtype
                        load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["torch_dtype"] = dtype
                    load_kwargs["device_map"] = "auto"
            elif device == "mps":
                # MPS does not support BitsAndBytes; use FP16
                load_kwargs["torch_dtype"] = dtype

            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

            if device == "mps" and not hasattr(model, "hf_device_map"):
                model = model.to(device)

            # Estimate model size
            param_count = sum(p.numel() for p in model.parameters())
            bytes_per_param = 2 if dtype else 4  # FP16 or FP32
            model_size_mb = (param_count * bytes_per_param) / (1024 * 1024)

            logger.info(f"Successfully loaded {name}: {param_count/1e6:.0f}M params, ~{model_size_mb:.0f} MB")
            return model, processor, name, model_size_mb

        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            continue

    logger.warning("Could not load any VLM model.")
    return None, None, "", 0


def benchmark_vlm(test_images: list, model, processor, model_name: str, family: str = "auto") -> list:
    """
    Run VLM on test images and collect detections + timing.
    """
    import torch
    from PIL import Image

    # Determine model family from name
    if "florence" in model_name.lower():
        family = "florence"
    else:
        family = "paligemma"

    results = []
    device = next(model.parameters()).device

    for img_info in test_images:
        img_path = img_info["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.debug(f"Cannot open image {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "detections": [],
                "time_ms": 0.0,
                "raw_response": f"ERROR: {e}",
            })
            continue

        t0 = time.perf_counter()
        raw_response = ""

        try:
            if family == "florence":
                # Florence-2 uses task tokens
                inputs = processor(
                    text="<OD>",
                    images=image,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                    )
                raw_response = processor.batch_decode(outputs, skip_special_tokens=False)[0]
                # Parse Florence-2 structured output
                try:
                    parsed = processor.post_process_generation(
                        raw_response, task="<OD>", image_size=image.size
                    )
                    detections = []
                    if isinstance(parsed, dict) and "<OD>" in parsed:
                        od_result = parsed["<OD>"]
                        bboxes = od_result.get("bboxes", [])
                        labels = od_result.get("labels", [])
                        for bbox, label in zip(bboxes, labels):
                            detections.append(Detection(
                                bbox=tuple(bbox),
                                label=label,
                                confidence=0.8,  # Florence-2 doesn't output confidence
                                source="vlm",
                            ))
                except Exception:
                    detections = parse_vlm_food_detections(raw_response, image.size)
            else:
                # PaliGemma-style models
                prompt = "Detect all food items in this image. For each item, provide: bounding box coordinates, food name, and confidence score."
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )
                raw_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                detections = parse_vlm_food_detections(raw_response, image.size)

        except Exception as e:
            logger.debug(f"VLM inference failed for {img_path}: {e}")
            raw_response = f"ERROR: {e}"
            detections = []

        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "image_path": img_path,
            "detections": detections,
            "time_ms": elapsed_ms,
            "raw_response": raw_response[:500],  # truncate for storage
        })

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_detection_metrics(
    predictions: list,  # list of Detection
    ground_truth_labels: list,
    ground_truth_bboxes: list,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute detection metrics for a single image.

    Returns dict with: tp, fp, fn, detection_rate, label_accuracy
    """
    tp = 0
    fp = 0
    matched_gt = set()

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt_bbox in enumerate(ground_truth_bboxes):
            if i in matched_gt:
                continue
            # Scale gt bbox from normalized to pixel coords if needed
            iou = compute_iou(pred.bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth_bboxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Detection rate: fraction of ground truth items detected
    detection_rate = tp / len(ground_truth_labels) if ground_truth_labels else 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_rate": detection_rate,
    }


def compute_aggregate_metrics(image_results: list, source: str) -> dict:
    """
    Compute aggregate metrics across all images for one source (yolo or vlm).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    latencies = []
    cuisine_metrics = {}

    for ir in image_results:
        if source == "yolo":
            detections = ir.get("yolo_detections", [])
            latency = ir.get("yolo_time_ms", 0.0)
        else:
            detections = ir.get("vlm_detections", [])
            latency = ir.get("vlm_time_ms", 0.0)

        gt_labels = ir.get("ground_truth_labels", [])
        gt_bboxes = ir.get("ground_truth_bboxes", [])
        cuisine = ir.get("cuisine", "Other")

        if latency > 0:
            latencies.append(latency)

        metrics = compute_detection_metrics(detections, gt_labels, gt_bboxes)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        if cuisine not in cuisine_metrics:
            cuisine_metrics[cuisine] = {"tp": 0, "fp": 0, "fn": 0, "count": 0}
        cuisine_metrics[cuisine]["tp"] += metrics["tp"]
        cuisine_metrics[cuisine]["fp"] += metrics["fp"]
        cuisine_metrics[cuisine]["fn"] += metrics["fn"]
        cuisine_metrics[cuisine]["count"] += 1

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    # Approximate mAP@0.5 as F1 (simplified; true mAP requires confidence-ranked PR curve)
    approx_map = overall_f1

    # Per-cuisine breakdown
    per_cuisine = {}
    for cuisine, cm in cuisine_metrics.items():
        p = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0.0
        r = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_cuisine[cuisine] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "approx_map": f,
            "image_count": cm["count"],
        }

    avg_latency = np.mean(latencies) if latencies else 0.0
    p50_latency = np.percentile(latencies, 50) if latencies else 0.0
    p95_latency = np.percentile(latencies, 95) if latencies else 0.0

    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "approx_map50": approx_map,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "per_cuisine": per_cuisine,
        "total_images": len(image_results),
    }


# ---------------------------------------------------------------------------
# Hybrid Routing Analysis
# ---------------------------------------------------------------------------

def analyze_hybrid_routing(image_results: list) -> dict:
    """
    Analyze hybrid YOLO + VLM routing at various confidence thresholds.

    For images where YOLO confidence < threshold and VLM got it right,
    count these as "hybrid wins". Calculate combined accuracy at each threshold.
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    analysis = {}

    for thresh in thresholds:
        yolo_only_correct = 0
        vlm_fallback_correct = 0
        vlm_fallback_wrong = 0
        yolo_primary_count = 0
        vlm_fallback_count = 0
        total_with_gt = 0

        for ir in image_results:
            gt_labels = ir.get("ground_truth_labels", [])
            if not gt_labels:
                continue
            total_with_gt += 1

            yolo_dets = ir.get("yolo_detections", [])
            vlm_dets = ir.get("vlm_detections", [])

            # Max YOLO confidence for this image
            yolo_max_conf = max((d.confidence for d in yolo_dets), default=0.0)

            # YOLO detection rate
            yolo_metrics = compute_detection_metrics(
                yolo_dets, gt_labels, ir.get("ground_truth_bboxes", [])
            )
            vlm_metrics = compute_detection_metrics(
                vlm_dets, gt_labels, ir.get("ground_truth_bboxes", [])
            )

            if yolo_max_conf >= thresh:
                # YOLO is primary
                yolo_primary_count += 1
                if yolo_metrics["detection_rate"] > 0.5:
                    yolo_only_correct += 1
            else:
                # VLM fallback
                vlm_fallback_count += 1
                if vlm_metrics["detection_rate"] > 0.5:
                    vlm_fallback_correct += 1
                else:
                    vlm_fallback_wrong += 1

        total_correct = yolo_only_correct + vlm_fallback_correct
        combined_accuracy = total_correct / total_with_gt if total_with_gt > 0 else 0.0
        yolo_usage_pct = yolo_primary_count / total_with_gt * 100 if total_with_gt > 0 else 0.0

        analysis[thresh] = {
            "combined_accuracy": combined_accuracy,
            "yolo_primary_count": yolo_primary_count,
            "vlm_fallback_count": vlm_fallback_count,
            "yolo_correct": yolo_only_correct,
            "vlm_fallback_correct": vlm_fallback_correct,
            "yolo_usage_pct": yolo_usage_pct,
        }

    # Find optimal threshold (highest combined accuracy)
    best_thresh = max(analysis.keys(), key=lambda t: analysis[t]["combined_accuracy"])

    return {
        "thresholds": analysis,
        "optimal_threshold": best_thresh,
        "optimal_accuracy": analysis[best_thresh]["combined_accuracy"],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: BenchmarkResults,
    yolo_metrics: dict,
    vlm_metrics: dict,
    hybrid_analysis: dict,
    output_path: Path,
):
    """Generate the benchmark comparison report in Markdown."""
    lines = []
    lines.append("# Benchmark Report: YOLO Pipeline vs VLM Food Detection")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"**Test set:** {results.test_set_source} ({results.total_images} images)")
    lines.append(f"**YOLO available:** {'Yes' if results.yolo_available else 'No (models not yet trained)'}")
    lines.append(f"**VLM model:** {results.vlm_model_name if results.vlm_available else 'Not available'}")
    lines.append("")

    if results.notes:
        lines.append("## Notes")
        lines.append("")
        for note in results.notes:
            lines.append(f"- {note}")
        lines.append("")

    # Side-by-side comparison table
    lines.append("## Side-by-Side Comparison")
    lines.append("")

    yolo_size = sum(results.yolo_model_sizes_mb.values())

    lines.append("| Metric | YOLO Pipeline | VLM ({}) |".format(
        results.vlm_model_name or "N/A"
    ))
    lines.append("|---|---|---|")

    def fmt_pct(v):
        return f"{v*100:.1f}%" if v > 0 else "N/A"

    def fmt_ms(v):
        return f"{v:.1f} ms" if v > 0 else "N/A"

    lines.append(f"| Overall Approx mAP@0.5 | {fmt_pct(yolo_metrics.get('approx_map50', 0))} | {fmt_pct(vlm_metrics.get('approx_map50', 0))} |")
    lines.append(f"| Precision | {fmt_pct(yolo_metrics.get('overall_precision', 0))} | {fmt_pct(vlm_metrics.get('overall_precision', 0))} |")
    lines.append(f"| Recall | {fmt_pct(yolo_metrics.get('overall_recall', 0))} | {fmt_pct(vlm_metrics.get('overall_recall', 0))} |")
    lines.append(f"| F1 Score | {fmt_pct(yolo_metrics.get('overall_f1', 0))} | {fmt_pct(vlm_metrics.get('overall_f1', 0))} |")
    lines.append(f"| Avg Latency | {fmt_ms(yolo_metrics.get('avg_latency_ms', 0))} | {fmt_ms(vlm_metrics.get('avg_latency_ms', 0))} |")
    lines.append(f"| P95 Latency | {fmt_ms(yolo_metrics.get('p95_latency_ms', 0))} | {fmt_ms(vlm_metrics.get('p95_latency_ms', 0))} |")
    lines.append(f"| Total Model Size | {yolo_size:.1f} MB | {results.vlm_model_size_mb:.0f} MB |")
    lines.append(f"| Deterministic | Yes | No |")
    lines.append(f"| On-device Feasible | Yes | {'Marginal' if results.vlm_model_size_mb < 2000 else 'Difficult'} |")
    lines.append("")

    # Per-cuisine breakdown
    lines.append("## Per-Cuisine Breakdown")
    lines.append("")
    lines.append("| Cuisine | YOLO F1 | VLM F1 | YOLO Images | VLM Images |")
    lines.append("|---|---|---|---|---|")

    all_cuisines = set()
    if yolo_metrics.get("per_cuisine"):
        all_cuisines.update(yolo_metrics["per_cuisine"].keys())
    if vlm_metrics.get("per_cuisine"):
        all_cuisines.update(vlm_metrics["per_cuisine"].keys())

    for cuisine in sorted(all_cuisines):
        y_cm = yolo_metrics.get("per_cuisine", {}).get(cuisine, {})
        v_cm = vlm_metrics.get("per_cuisine", {}).get(cuisine, {})
        y_f1 = fmt_pct(y_cm.get("f1", 0))
        v_f1 = fmt_pct(v_cm.get("f1", 0))
        y_cnt = y_cm.get("image_count", 0)
        v_cnt = v_cm.get("image_count", 0)
        lines.append(f"| {cuisine} | {y_f1} | {v_f1} | {y_cnt} | {v_cnt} |")
    lines.append("")

    # Hybrid routing analysis
    lines.append("## Hybrid Routing Analysis")
    lines.append("")
    lines.append("Strategy: Use YOLO as primary detector. When YOLO max confidence < threshold,")
    lines.append("fall back to VLM for a second opinion.")
    lines.append("")

    if hybrid_analysis and hybrid_analysis.get("thresholds"):
        lines.append("| Confidence Threshold | Combined Accuracy | YOLO Primary % | VLM Fallback Count | VLM Correct |")
        lines.append("|---|---|---|---|---|")
        for thresh in sorted(hybrid_analysis["thresholds"].keys()):
            ha = hybrid_analysis["thresholds"][thresh]
            lines.append(
                f"| {thresh:.1f} | {ha['combined_accuracy']*100:.1f}% "
                f"| {ha['yolo_usage_pct']:.0f}% "
                f"| {ha['vlm_fallback_count']} "
                f"| {ha['vlm_fallback_correct']} |"
            )
        lines.append("")
        lines.append(f"**Optimal threshold:** {hybrid_analysis['optimal_threshold']:.1f} "
                      f"(combined accuracy: {hybrid_analysis['optimal_accuracy']*100:.1f}%)")
    else:
        lines.append("*Hybrid analysis requires both YOLO and VLM results. Skipped.*")
    lines.append("")

    # Latency breakdown
    lines.append("## Latency Breakdown")
    lines.append("")
    if results.yolo_available:
        lines.append("### YOLO Pipeline")
        lines.append(f"- Binary gate: ~{yolo_metrics.get('avg_latency_ms', 0) * 0.1:.1f} ms (estimated)")
        lines.append(f"- Detection: ~{yolo_metrics.get('avg_latency_ms', 0) * 0.5:.1f} ms (estimated)")
        lines.append(f"- Classification: ~{yolo_metrics.get('avg_latency_ms', 0) * 0.4:.1f} ms (estimated)")
        lines.append(f"- **Total average: {yolo_metrics.get('avg_latency_ms', 0):.1f} ms**")
        lines.append("")
    if results.vlm_available:
        lines.append(f"### VLM ({results.vlm_model_name})")
        lines.append(f"- Average: {vlm_metrics.get('avg_latency_ms', 0):.1f} ms")
        lines.append(f"- P50: {vlm_metrics.get('p50_latency_ms', 0):.1f} ms")
        lines.append(f"- P95: {vlm_metrics.get('p95_latency_ms', 0):.1f} ms")
        lines.append(f"- **~{vlm_metrics.get('avg_latency_ms', 0) / max(yolo_metrics.get('avg_latency_ms', 1), 1):.0f}x slower than YOLO**" if yolo_metrics.get('avg_latency_ms', 0) > 0 else "")
        lines.append("")

    # Go/No-Go data summary
    lines.append("## Go/No-Go Decision Data")
    lines.append("")
    lines.append("Per locked decision DET-07: \"If YOLO hits 85% but LLM hits 97%, invest more in YOLO training first.\"")
    lines.append("")
    lines.append("| Factor | Value | Assessment |")
    lines.append("|---|---|---|")

    yolo_acc = yolo_metrics.get("approx_map50", 0) * 100
    vlm_acc = vlm_metrics.get("approx_map50", 0) * 100

    if results.yolo_available:
        assessment = "Needs improvement" if yolo_acc < 85 else ("Good" if yolo_acc < 95 else "Excellent")
        lines.append(f"| YOLO accuracy | {yolo_acc:.1f}% | {assessment} |")
    else:
        lines.append("| YOLO accuracy | Not yet available (training in progress) | Pending 01-03 |")

    if results.vlm_available:
        lines.append(f"| VLM accuracy | {vlm_acc:.1f}% | Benchmark reference |")
        lines.append(f"| VLM latency | {vlm_metrics.get('avg_latency_ms', 0):.0f} ms | {'Acceptable for fallback' if vlm_metrics.get('avg_latency_ms', 0) < 5000 else 'Too slow for interactive'} |")
    else:
        lines.append("| VLM accuracy | Not available | Could not load model |")

    gap = vlm_acc - yolo_acc
    if results.yolo_available and results.vlm_available:
        if gap > 12:
            lines.append(f"| Accuracy gap | {gap:.1f}% (VLM leads) | Consider hybrid approach |")
        elif gap > 0:
            lines.append(f"| Accuracy gap | {gap:.1f}% (VLM leads slightly) | YOLO preferred (speed + determinism) |")
        else:
            lines.append(f"| YOLO advantage | {-gap:.1f}% | YOLO recommended |")
    lines.append("")

    lines.append("### Recommendation")
    lines.append("")
    if not results.yolo_available and not results.vlm_available:
        lines.append("**Status: Benchmark infrastructure ready, awaiting models.**")
        lines.append("- YOLO models: Complete Plan 01-03 (YOLO training) first")
        lines.append("- VLM models: Requires downloading PaliGemma/Florence-2 from HuggingFace")
        lines.append("- Re-run this benchmark after models are available: `python training/benchmark.py`")
    elif not results.yolo_available:
        lines.append("**Status: VLM benchmarked, awaiting YOLO training.**")
        lines.append("- Complete Plan 01-03 and re-run benchmark for comparison")
    elif not results.vlm_available:
        lines.append("**Status: YOLO benchmarked, VLM not available.**")
        lines.append("- YOLO-only pipeline is the recommendation")
        lines.append("- VLM fallback can be added later if models become available")
    else:
        if yolo_acc >= 85:
            lines.append("**Recommendation: Use YOLO as primary pipeline.**")
            if gap > 5:
                lines.append(f"- Consider hybrid routing at confidence threshold {hybrid_analysis.get('optimal_threshold', 0.6):.1f} for difficult images")
            else:
                lines.append("- VLM fallback provides minimal accuracy gain; not recommended")
        else:
            lines.append("**Recommendation: Continue YOLO training (accuracy below target).**")
            lines.append("- Per locked decision: invest more in YOLO training before pivoting to VLM")
            lines.append("- Audit per-cuisine training data for underperforming categories")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `training/benchmark.py` on {time.strftime('%Y-%m-%d')}*")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Benchmark report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO vs VLM food detection")
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR),
                        help="Directory containing test images")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Maximum number of test images to use")
    parser.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DIR),
                        help="Directory containing YOLO model weights")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM benchmark (YOLO only)")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="Skip YOLO benchmark (VLM only)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force use of synthetic test images")
    parser.add_argument("--output", type=str, default=str(REPORT_PATH),
                        help="Output path for benchmark report")
    args = parser.parse_args()

    bench = BenchmarkResults()

    # Step 1: Prepare test set
    test_dir = Path(args.test_dir)
    test_images = []

    if not args.synthetic:
        test_images = discover_test_images(test_dir, args.max_images)

    if not test_images or args.synthetic:
        logger.info("No real test images available. Generating synthetic test set.")
        synthetic_dir = TRAINING_DIR / "datasets" / "synthetic_test" / "images"
        test_images = generate_synthetic_test_set(synthetic_dir, count=args.max_images)
        bench.test_set_source = f"synthetic ({len(test_images)} generated images)"
        bench.notes.append(
            "Test images are synthetic (colored rectangles). Real benchmark accuracy "
            "requires downloading datasets first: `python training/datasets/scripts/download_datasets.py`"
        )
    else:
        bench.test_set_source = f"real ({len(test_images)} images from {test_dir})"

    bench.total_images = len(test_images)

    # Build unified image_results list
    image_results = []
    for img_info in test_images:
        image_results.append({
            "image_path": img_info["image_path"],
            "ground_truth_labels": img_info["labels"],
            "ground_truth_bboxes": img_info["bboxes"],
            "cuisine": img_info["cuisine"],
            "yolo_detections": [],
            "yolo_time_ms": 0.0,
            "vlm_detections": [],
            "vlm_time_ms": 0.0,
        })

    # Step 2: YOLO benchmark
    if not args.skip_yolo:
        logger.info("=" * 60)
        logger.info("YOLO PIPELINE BENCHMARK")
        logger.info("=" * 60)
        yolo_results, model_sizes, yolo_available = benchmark_yolo_pipeline(
            test_images, Path(args.yolo_dir)
        )
        bench.yolo_available = yolo_available
        bench.yolo_model_sizes_mb = model_sizes

        if yolo_available:
            for i, yr in enumerate(yolo_results):
                if i < len(image_results):
                    image_results[i]["yolo_detections"] = yr["detections"]
                    image_results[i]["yolo_time_ms"] = yr["total_time_ms"]

            if not yolo_available:
                bench.notes.append(
                    "YOLO models not yet trained. Plan 01-03 (YOLO Training) is running in parallel. "
                    "Re-run benchmark after training completes."
                )
    else:
        bench.notes.append("YOLO benchmark skipped (--skip-yolo flag).")

    # Step 3: VLM benchmark
    if not args.skip_vlm:
        logger.info("=" * 60)
        logger.info("VLM BENCHMARK")
        logger.info("=" * 60)
        vlm_model, vlm_processor, vlm_name, vlm_size = try_load_vlm()
        if vlm_model is not None:
            bench.vlm_available = True
            bench.vlm_model_name = vlm_name
            bench.vlm_model_size_mb = vlm_size

            vlm_results = benchmark_vlm(test_images, vlm_model, vlm_processor, vlm_name)
            for i, vr in enumerate(vlm_results):
                if i < len(image_results):
                    image_results[i]["vlm_detections"] = vr["detections"]
                    image_results[i]["vlm_time_ms"] = vr["time_ms"]

            # Free VLM memory
            del vlm_model
            del vlm_processor
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass
        else:
            bench.vlm_available = False
            bench.notes.append(
                "Could not load any VLM model (PaliGemma 2 3B, PaliGemma 2 1B, or Florence-2). "
                "This may be due to: missing HuggingFace authentication, insufficient memory, "
                "or network issues downloading model weights. "
                "To authenticate: `huggingface-cli login` with a token that has access to gated models."
            )
    else:
        bench.notes.append("VLM benchmark skipped (--skip-vlm flag).")

    # Step 4: Compute metrics
    logger.info("=" * 60)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 60)

    yolo_metrics = compute_aggregate_metrics(image_results, "yolo")
    vlm_metrics = compute_aggregate_metrics(image_results, "vlm")

    # Step 5: Hybrid routing analysis
    hybrid_analysis = {}
    if bench.yolo_available and bench.vlm_available:
        hybrid_analysis = analyze_hybrid_routing(image_results)

    # Step 6: Generate report
    generate_report(
        results=bench,
        yolo_metrics=yolo_metrics,
        vlm_metrics=vlm_metrics,
        hybrid_analysis=hybrid_analysis,
        output_path=Path(args.output),
    )

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Test set: {bench.test_set_source}")
    print(f"YOLO available: {bench.yolo_available}")
    print(f"VLM available: {bench.vlm_available} ({bench.vlm_model_name})")
    if bench.yolo_available:
        print(f"YOLO approx mAP@0.5: {yolo_metrics['approx_map50']*100:.1f}%")
        print(f"YOLO avg latency: {yolo_metrics['avg_latency_ms']:.1f} ms")
    if bench.vlm_available:
        print(f"VLM approx mAP@0.5: {vlm_metrics['approx_map50']*100:.1f}%")
        print(f"VLM avg latency: {vlm_metrics['avg_latency_ms']:.1f} ms")
    if hybrid_analysis:
        print(f"Optimal hybrid threshold: {hybrid_analysis['optimal_threshold']:.1f}")
        print(f"Hybrid combined accuracy: {hybrid_analysis['optimal_accuracy']*100:.1f}%")
    print(f"\nReport: {Path(args.output)}")


if __name__ == "__main__":
    main()
