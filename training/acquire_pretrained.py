#!/usr/bin/env python3
"""
Acquire pre-trained ML models for the food detection pipeline.

Downloads Google AIY Food V1 (classification/binary gate) and exports
YOLO26n COCO (detection) to TFLite. Validates both models via dummy
inference, generates a model manifest, and copies assets to the mobile
app directory.

These are zero-training baselines to unblock on-device testing of the
three-stage pipeline: binary gate -> detection -> classification.

Per RESEARCH.md / CONTEXT.md:
- AIY Food V1: Apache 2.0, 224x224 input, 2023 food classes, ~6MB
- YOLO26n COCO: FP16 TFLite, 640x640 input, 80 COCO classes
- NMS is NOT baked in (done in JavaScript for portability)

Usage:
    python training/acquire_pretrained.py
    python training/acquire_pretrained.py --validate
    python training/acquire_pretrained.py --output-dir DIR
    python training/acquire_pretrained.py --skip-export
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen, urlretrieve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "exports"
MOBILE_ASSETS_DIR = TRAINING_DIR.parent / "apps" / "mobile" / "assets" / "models"

# Google AIY Food V1 download URLs (ordered by preference)
# Primary: Kaggle Models API (tar.gz archive containing 1.tflite)
# Fallback: Legacy GCS direct download (may return 403)
AIY_FOOD_URLS = [
    (
        "https://www.kaggle.com/api/v1/models/google/aiy/tfLite/"
        "vision-classifier-food-v1/1/download",
        "kaggle_archive",  # tar.gz archive
    ),
    (
        "https://storage.googleapis.com/tfhub-lite-models/"
        "google/lite-model/aiy/vision/classifier/food_V1/1.tflite",
        "direct",  # direct .tflite file
    ),
]

# YOLO fallback chain: try newest first
YOLO_FALLBACKS = ["yolo26n", "yolo11n", "yolov8n"]

# 80 COCO class names at correct indices (0-79)
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# COCO food class indices (0-indexed)
COCO_FOOD_CLASS_IDS = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Acquire pre-trained models (AIY Food V1 + YOLO26n COCO) "
            "for the food detection pipeline."
        )
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation step on models (default: always validate)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Export destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip YOLO export, only download AIY (useful when YOLO export was already done)",
    )
    return parser.parse_args()


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Progress indicator for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(
            f"\r  Downloading: {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)",
            end="",
            flush=True,
        )
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\r  Downloaded: {mb_down:.1f} MB", end="", flush=True)


def _download_url(url: str, dest_path: str) -> None:
    """Download a URL to a file path with User-Agent header."""
    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (foodtracker-ml-pipeline)")

    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(
                        f"\r  Downloading: {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)",
                        end="",
                        flush=True,
                    )
        print()  # newline after progress


def download_aiy_food(output_dir: Path) -> Path:
    """
    Download the Google AIY Food V1 TFLite model.

    Tries Kaggle Models API first (tar.gz archive), falls back to GCS
    direct download. Returns the path to the downloaded model file.
    Raises RuntimeError if all download sources fail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "aiy_food_v1.tflite"

    if dest.exists():
        file_size = dest.stat().st_size
        if file_size > 5 * 1024 * 1024:  # >5MB
            log.info(
                "AIY Food V1 already exists: %s (%.2f MB)",
                dest,
                file_size / (1024 * 1024),
            )
            return dest
        else:
            log.warning(
                "AIY Food V1 exists but too small (%.2f MB), re-downloading",
                file_size / (1024 * 1024),
            )

    errors: list[str] = []
    for url, url_type in AIY_FOOD_URLS:
        log.info("Trying AIY Food V1 download: %s (%s)...", url_type, url[:80])
        try:
            if url_type == "kaggle_archive":
                # Kaggle API returns a tar.gz archive containing the model
                with tempfile.TemporaryDirectory() as tmpdir:
                    archive_path = os.path.join(tmpdir, "model.tar.gz")
                    _download_url(url, archive_path)

                    # Extract .tflite from archive
                    with tarfile.open(archive_path, "r:gz") as tar:
                        members = tar.getnames()
                        tflite_members = [
                            m for m in members if m.endswith(".tflite")
                        ]
                        if not tflite_members:
                            raise RuntimeError(
                                f"No .tflite file in archive: {members}"
                            )
                        tar.extract(tflite_members[0], tmpdir)
                        extracted = os.path.join(tmpdir, tflite_members[0])
                        shutil.copy2(extracted, str(dest))

            elif url_type == "direct":
                _download_url(url, str(dest))

            # Verify file size
            file_size = dest.stat().st_size
            if file_size < 5 * 1024 * 1024:
                raise RuntimeError(
                    f"Downloaded file too small: {file_size} bytes (expected >5MB)"
                )

            log.info(
                "AIY Food V1 downloaded: %s (%.2f MB)",
                dest,
                file_size / (1024 * 1024),
            )
            return dest

        except Exception as exc:
            msg = f"{url_type}: {exc}"
            log.warning("  Download failed: %s", msg)
            errors.append(msg)
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            continue

    raise RuntimeError(
        f"All AIY Food V1 download sources failed:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def export_yolo_tflite(output_dir: Path) -> Path:
    """
    Export YOLO26n COCO to FP16 TFLite.

    Follows the YOLO_FALLBACKS pattern: tries yolo26n first, falls back
    to yolo11n if export fails.

    Returns the path to the exported TFLite model.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "ultralytics is not installed. "
            "Install with: pip install ultralytics"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "yolo26n_coco.tflite"

    if dest.exists():
        file_size = dest.stat().st_size
        if file_size > 5 * 1024 * 1024:
            log.info(
                "YOLO TFLite already exists: %s (%.2f MB)",
                dest,
                file_size / (1024 * 1024),
            )
            return dest

    for model_name in YOLO_FALLBACKS:
        log.info("Trying YOLO export: %s -> FP16 TFLite (640px, no NMS)...", model_name)
        try:
            model = YOLO(f"{model_name}.pt")
            exported_path = model.export(
                format="tflite",
                imgsz=640,
                half=True,
                nms=False,
            )
            exported = Path(str(exported_path))

            # ultralytics may return the directory; find the .tflite file
            if not exported.exists() or exported.is_dir():
                search_dir = exported if exported.is_dir() else exported.parent
                tflite_files = list(search_dir.glob("*.tflite"))
                if not tflite_files:
                    log.warning(
                        "Export of %s produced no .tflite file in %s",
                        model_name,
                        search_dir,
                    )
                    continue
                exported = tflite_files[0]

            # Move to output directory with canonical name
            shutil.copy2(str(exported), str(dest))
            file_size = dest.stat().st_size

            log.info(
                "YOLO export success: %s -> %s (%.2f MB)",
                model_name,
                dest,
                file_size / (1024 * 1024),
            )
            return dest

        except Exception as exc:
            log.warning("YOLO export failed for %s: %s", model_name, exc)
            continue

    raise RuntimeError(
        f"All YOLO fallbacks failed: {YOLO_FALLBACKS}. "
        "Ensure ultralytics and tensorflow are installed."
    )


def _load_interpreter(model_path: str):
    """Load a TFLite interpreter with fallback chain."""
    # Try ai-edge-litert first (modern replacement for tflite-runtime)
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass

    # Then tflite_runtime
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]
        return tflite.Interpreter(model_path=model_path)
    except ImportError:
        pass

    # Finally tensorflow
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        pass

    raise RuntimeError(
        "No TFLite interpreter available. Install one of: "
        "ai-edge-litert, tflite-runtime, tensorflow"
    )


def validate_model(model_path: Path, expected_name: str) -> dict:
    """
    Validate a TFLite model by loading it and running dummy inference.

    Returns a dict with model metadata (shapes, dtypes, normalization info).
    """
    import numpy as np  # type: ignore[import-untyped]

    log.info("Validating %s: %s", expected_name, model_path)

    interpreter = _load_interpreter(str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"].tolist()
    raw_dtype = input_details[0]["dtype"]
    # Normalize dtype to clean string (e.g., "float32" not "<class 'numpy.float32'>")
    input_dtype = raw_dtype.__name__ if hasattr(raw_dtype, "__name__") else str(raw_dtype)

    # Determine input normalization from dtype and quantization params
    quant_params = input_details[0].get("quantization_parameters", {})
    scales = quant_params.get("scales", [])
    zero_points = quant_params.get("zero_points", [])

    if "uint8" in input_dtype:
        if len(scales) > 0 and scales[0] != 0:
            input_normalization = "uint8_quantized"
        else:
            input_normalization = "uint8_0_255"
    elif "float" in input_dtype:
        # Most MobileNet models use [-1, 1] or [0, 1]
        # We document the dtype; actual range needs empirical testing
        input_normalization = "float32_0_1"
    else:
        input_normalization = f"unknown_{input_dtype}"

    # Run dummy inference
    dummy_input = np.zeros(input_shape, dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    output_shape = list(output.shape)
    raw_out_dtype = output_details[0]["dtype"]
    output_dtype = raw_out_dtype.__name__ if hasattr(raw_out_dtype, "__name__") else str(raw_out_dtype)

    log.info(
        "  %s: input=%s (%s), output=%s (%s), normalization=%s",
        expected_name,
        input_shape,
        input_dtype,
        output_shape,
        output_dtype,
        input_normalization,
    )

    # Log quantization params for AIY model (critical for preprocessing)
    if quant_params and (len(scales) > 0 or len(zero_points) > 0):
        log.info(
            "  Quantization: scales=%s, zero_points=%s",
            scales[:5] if len(scales) > 5 else scales,
            zero_points[:5] if len(zero_points) > 5 else zero_points,
        )

    return {
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "input_normalization": input_normalization,
        "quantization_scales": scales.tolist() if hasattr(scales, "tolist") else list(scales),
        "quantization_zero_points": (
            zero_points.tolist() if hasattr(zero_points, "tolist") else list(zero_points)
        ),
    }


def extract_aiy_labels(model_path: Path, output_dir: Path) -> Path | None:
    """
    Try to extract labels from AIY Food V1 TFLite model metadata.

    TFLite models with metadata are zip-extractable. If labels are embedded,
    save them to output_dir/labels_food_v1.txt.

    Returns the path to extracted labels, or None if not found.
    """
    log.info("Attempting to extract labels from AIY Food V1 metadata...")

    try:
        with zipfile.ZipFile(str(model_path), "r") as zf:
            names = zf.namelist()
            log.info("  Found metadata files: %s", names)

            # Prefer English labels over Freebase MIDs
            label_candidates = [
                n for n in names
                if "label" in n.lower() or "class" in n.lower() or n.endswith(".txt")
            ]
            # Sort: prefer *-en.txt (English) over generic label files
            label_candidates.sort(key=lambda x: (0 if "-en" in x else 1, x))

            if label_candidates:
                label_file = label_candidates[0]
                labels_raw = zf.read(label_file).decode("utf-8")
                dest = output_dir / "labels_food_v1.txt"
                dest.write_text(labels_raw)
                label_count = len([line for line in labels_raw.strip().split("\n") if line.strip()])
                log.info(
                    "  Extracted %d labels from %s -> %s",
                    label_count,
                    label_file,
                    dest,
                )
                return dest
            else:
                log.warning("  No label files found in model metadata")
                return None

    except zipfile.BadZipFile:
        log.warning("  Model is not zip-extractable (no embedded metadata)")
        return None
    except Exception as exc:
        log.warning("  Failed to extract metadata: %s", exc)
        return None


def copy_models_to_assets(
    aiy_path: Path,
    yolo_path: Path | None,
) -> dict[str, Path]:
    """
    Copy model files to the mobile assets directory.

    AIY Food V1 is used for both binary gate and classification.
    YOLO26n is used for detection.

    Returns a dict mapping stage names to their asset paths.
    """
    MOBILE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    # binary.tflite = AIY Food V1 (binary gate: max confidence > threshold = food)
    dest_binary = MOBILE_ASSETS_DIR / "binary.tflite"
    shutil.copy2(str(aiy_path), str(dest_binary))
    paths["binary"] = dest_binary
    log.info("Copied: %s -> %s", aiy_path.name, dest_binary)

    # classify.tflite = AIY Food V1 (same model, full classification)
    dest_classify = MOBILE_ASSETS_DIR / "classify.tflite"
    shutil.copy2(str(aiy_path), str(dest_classify))
    paths["classify"] = dest_classify
    log.info("Copied: %s -> %s", aiy_path.name, dest_classify)

    # detect.tflite = YOLO26n COCO
    if yolo_path and yolo_path.exists():
        dest_detect = MOBILE_ASSETS_DIR / "detect.tflite"
        shutil.copy2(str(yolo_path), str(dest_detect))
        paths["detect"] = dest_detect
        log.info("Copied: %s -> %s", yolo_path.name, dest_detect)

    return paths


def generate_manifest(
    asset_paths: dict[str, Path],
    validation_results: dict[str, dict],
) -> Path:
    """
    Generate model_manifest.json compatible with the PackManager system.

    The manifest follows the format from export_mobile.py.
    """
    models = []

    # Binary gate (AIY Food V1)
    if "binary" in asset_paths:
        binary_path = asset_paths["binary"]
        binary_meta = validation_results.get("aiy", {})
        aiy_input_size = binary_meta.get("input_shape", [1, 192, 192, 3])[1]
        aiy_num_classes = binary_meta.get("output_shape", [1, 2024])[-1]
        aiy_quant = "uint8_quantized" if "uint8" in binary_meta.get("input_dtype", "") else "none"
        models.append({
            "id": "yolo-binary-v1",
            "stage": "binary",
            "version": "1.0.0",
            "file": "binary.tflite",
            "sizeBytes": binary_path.stat().st_size,
            "format": "tflite",
            "quantisation": aiy_quant,
            "inputSize": aiy_input_size,
            "numClasses": aiy_num_classes,
            "inputNormalization": binary_meta.get("input_normalization", "unknown"),
            "inputDtype": binary_meta.get("input_dtype", "unknown"),
            "outputShape": binary_meta.get("output_shape", []),
            "name": "AIY Food V1 (Binary Gate)",
            "description": f"Google AIY Food V1 model repurposed as binary gate. Max confidence across {aiy_num_classes} food classes > threshold = food detected.",
        })

    # Detection (YOLO26n COCO)
    if "detect" in asset_paths:
        detect_path = asset_paths["detect"]
        detect_meta = validation_results.get("yolo", {})
        models.append({
            "id": "yolo-detect-v1",
            "stage": "detect",
            "version": "1.0.0",
            "file": "detect.tflite",
            "sizeBytes": detect_path.stat().st_size,
            "format": "tflite",
            "quantisation": "fp16",
            "inputSize": 640,
            "numClasses": 80,
            "inputNormalization": detect_meta.get("input_normalization", "float32_0_1"),
            "inputDtype": detect_meta.get("input_dtype", "float32"),
            "outputShape": detect_meta.get("output_shape", []),
            "name": "YOLO26n COCO Detection",
            "description": "YOLO26n pre-trained on COCO. 80 classes including 10 food items. NMS performed in JavaScript.",
        })

    # Classification (AIY Food V1)
    if "classify" in asset_paths:
        classify_path = asset_paths["classify"]
        classify_meta = validation_results.get("aiy", {})
        models.append({
            "id": "yolo-classify-v1",
            "stage": "classify",
            "version": "1.0.0",
            "file": "classify.tflite",
            "sizeBytes": classify_path.stat().st_size,
            "format": "tflite",
            "quantisation": aiy_quant,
            "inputSize": aiy_input_size,
            "numClasses": aiy_num_classes,
            "inputNormalization": classify_meta.get("input_normalization", "unknown"),
            "inputDtype": classify_meta.get("input_dtype", "unknown"),
            "outputShape": classify_meta.get("output_shape", []),
            "name": "AIY Food V1 (Classification)",
            "description": f"Google AIY Food V1 model for {aiy_num_classes}-class food classification.",
        })

    manifest = {
        "version": "1.0.0",
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "pipeline": "three-stage",
        "quantisation": "mixed",
        "notes": (
            "Pre-trained baseline models. AIY Food V1 for binary gate + classify "
            "(no custom training). YOLO26n COCO FP16 for detection. "
            "NMS is performed in JavaScript, not baked into models."
        ),
        "models": models,
    }

    manifest_path = MOBILE_ASSETS_DIR / "model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written: %s (%d models)", manifest_path, len(models))

    return manifest_path


def generate_labels_coco() -> Path:
    """
    Generate labels_coco.json with all 80 COCO class names and food class IDs.
    """
    labels = {
        "classNames": COCO_CLASS_NAMES,
        "count": len(COCO_CLASS_NAMES),
        "foodClassIds": COCO_FOOD_CLASS_IDS,
        "foodClassNames": {
            str(idx): COCO_CLASS_NAMES[idx] for idx in COCO_FOOD_CLASS_IDS
        },
    }

    dest = MOBILE_ASSETS_DIR / "labels_coco.json"
    with open(dest, "w") as f:
        json.dump(labels, f, indent=2)
    log.info("COCO labels written: %s (%d classes, %d food)", dest, len(COCO_CLASS_NAMES), len(COCO_FOOD_CLASS_IDS))

    return dest


def generate_labels_food_v1(extracted_labels_path: Path | None) -> Path:
    """
    Generate labels_food_v1.json from extracted labels or a fallback stub.
    """
    dest = MOBILE_ASSETS_DIR / "labels_food_v1.json"

    if extracted_labels_path and extracted_labels_path.exists():
        raw = extracted_labels_path.read_text()
        labels = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        data = {
            "source": "extracted_from_model",
            "count": len(labels),
            "classNames": labels,
        }
        log.info("Food V1 labels written: %s (%d classes)", dest, len(labels))
    else:
        data = {
            "source": "not_extracted",
            "count": 2023,
            "note": (
                "AIY Food V1 label extraction deferred; binary gate uses "
                "max-confidence approach which does not need labels"
            ),
        }
        log.info("Food V1 labels stub written: %s (labels not extracted)", dest)

    with open(dest, "w") as f:
        json.dump(data, f, indent=2)

    return dest


def main() -> int:
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = TRAINING_DIR / output_dir

    should_validate = not args.no_validate

    log.info("=" * 60)
    log.info("Acquiring pre-trained models for food detection pipeline")
    log.info("=" * 60)

    # Step 1: Download AIY Food V1
    log.info("")
    log.info("--- Step 1: Download AIY Food V1 ---")
    aiy_path = download_aiy_food(output_dir)

    # Step 2: Export YOLO26n COCO to TFLite
    yolo_path = None
    if not args.skip_export:
        log.info("")
        log.info("--- Step 2: Export YOLO26n COCO -> TFLite ---")
        yolo_path = export_yolo_tflite(output_dir)
    else:
        # Check if existing export exists
        candidate = output_dir / "yolo26n_coco.tflite"
        if candidate.exists():
            yolo_path = candidate
            log.info("Using existing YOLO export: %s", yolo_path)
        else:
            log.warning("YOLO export skipped and no existing file found")

    # Step 3: Validate models
    validation_results: dict[str, dict] = {}
    if should_validate:
        log.info("")
        log.info("--- Step 3: Validate models ---")
        try:
            aiy_meta = validate_model(aiy_path, "AIY Food V1")
            validation_results["aiy"] = aiy_meta

            # Verify AIY expectations
            # Note: Actual model uses 192x192 (not 224x224 as initially assumed)
            # and has 2024 classes (not 2023). These are the real model properties.
            expected_aiy_inputs = [[1, 192, 192, 3], [1, 224, 224, 3]]
            if aiy_meta["input_shape"] not in expected_aiy_inputs:
                log.warning(
                    "AIY input shape unexpected: got %s (expected one of %s)",
                    aiy_meta["input_shape"],
                    expected_aiy_inputs,
                )
            else:
                log.info("  AIY input shape %s confirmed", aiy_meta["input_shape"])

            expected_aiy_outputs = [[1, 2024], [1, 2023]]
            if aiy_meta["output_shape"] not in expected_aiy_outputs:
                log.warning(
                    "AIY output shape unexpected: got %s (expected one of %s)",
                    aiy_meta["output_shape"],
                    expected_aiy_outputs,
                )
            else:
                log.info("  AIY output shape %s confirmed", aiy_meta["output_shape"])

        except Exception as exc:
            log.error("AIY validation failed: %s", exc)
            return 1

        if yolo_path:
            try:
                yolo_meta = validate_model(yolo_path, "YOLO26n COCO")
                validation_results["yolo"] = yolo_meta

                # Verify YOLO expectations
                expected_yolo_input = [1, 640, 640, 3]
                if yolo_meta["input_shape"] != expected_yolo_input:
                    log.warning(
                        "YOLO input shape mismatch: expected %s, got %s",
                        expected_yolo_input,
                        yolo_meta["input_shape"],
                    )

                # YOLO output could be [1, 84, 8400] or [1, 8400, 84]
                expected_shapes = [[1, 84, 8400], [1, 8400, 84]]
                if yolo_meta["output_shape"] not in expected_shapes:
                    log.warning(
                        "YOLO output shape unexpected: got %s (expected one of %s)",
                        yolo_meta["output_shape"],
                        expected_shapes,
                    )
                else:
                    log.info("  YOLO output shape %s confirmed", yolo_meta["output_shape"])

                if "float32" not in yolo_meta["output_dtype"]:
                    log.warning(
                        "YOLO output dtype: expected float32, got %s",
                        yolo_meta["output_dtype"],
                    )
                else:
                    log.info("  YOLO output dtype float32 confirmed")

            except Exception as exc:
                log.error("YOLO validation failed: %s", exc)
                return 1

    # Step 4: Extract AIY labels
    log.info("")
    log.info("--- Step 4: Extract AIY Food V1 labels ---")
    extracted_labels = extract_aiy_labels(aiy_path, output_dir)

    # Step 5: Copy models to mobile assets
    log.info("")
    log.info("--- Step 5: Copy models to mobile assets ---")
    asset_paths = copy_models_to_assets(aiy_path, yolo_path)

    # Step 6: Generate model manifest
    log.info("")
    log.info("--- Step 6: Generate model manifest ---")
    generate_manifest(asset_paths, validation_results)

    # Step 7: Generate COCO labels
    log.info("")
    log.info("--- Step 7: Generate COCO labels ---")
    generate_labels_coco()

    # Step 8: Generate Food V1 labels
    log.info("")
    log.info("--- Step 8: Generate Food V1 labels ---")
    generate_labels_food_v1(extracted_labels)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("Acquisition complete!")
    log.info("=" * 60)

    for name, path in asset_paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info("  %s: %s (%.2f MB)", name, path, size_mb)

    log.info("  manifest: %s", MOBILE_ASSETS_DIR / "model_manifest.json")
    log.info("  labels_coco: %s", MOBILE_ASSETS_DIR / "labels_coco.json")
    log.info("  labels_food_v1: %s", MOBILE_ASSETS_DIR / "labels_food_v1.json")

    if validation_results:
        log.info("")
        log.info("Validation results:")
        for name, meta in validation_results.items():
            log.info(
                "  %s: input=%s (%s, norm=%s), output=%s (%s)",
                name,
                meta["input_shape"],
                meta["input_dtype"],
                meta["input_normalization"],
                meta["output_shape"],
                meta["output_dtype"],
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
