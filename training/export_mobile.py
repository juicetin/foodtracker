#!/usr/bin/env python3
"""
Export trained YOLO models to TFLite format for mobile deployment.

Exports all three pipeline stages (binary, detect, classify) from their
training run best.pt checkpoints to FP16 TFLite models.

Per RESEARCH.md:
- Export TFLite only (not CoreML) -- react-native-fast-tflite's CoreML
  delegate handles iOS hardware acceleration from .tflite files directly.
  Shipping a separate .mlmodel would double bundle size for no gain.
- Use FP16 (half=True) for good size/accuracy tradeoff. INT8 quantisation
  requires a calibration dataset and risks accuracy loss on food images
  with subtle colour differences.
- Do NOT use nms=True -- YOLO's built-in TFLite NMS op is not supported
  by react-native-fast-tflite's LiteRT runtime. We perform NMS in
  JavaScript for cross-platform portability.
- Do NOT use end2end=True -- incompatible with TFLite NMS export path
  and adds an opaque post-processing op that can't be customised.

Usage:
    python training/export_mobile.py
    python training/export_mobile.py --binary-weights runs/binary/best.pt
    python training/export_mobile.py --detect-weights runs/detect/best.pt
    python training/export_mobile.py --classify-weights runs/classify/best.pt
    python training/export_mobile.py --output-dir exports/
    python training/export_mobile.py --validate  # Run validation on exported models
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent
RUNS_DIR = TRAINING_DIR / "runs"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "exports"

# Pipeline stage configurations
# binary/classify use 224px (small classification CNNs)
# detect uses 640px (full YOLO detection head)
STAGES = {
    "binary": {"imgsz": 224, "default_weights": "runs/binary/best.pt"},
    "detect": {"imgsz": 640, "default_weights": "runs/detect/best.pt"},
    "classify": {"imgsz": 224, "default_weights": "runs/classify/best.pt"},
}

# YOLO base model fallback chain: try newest first, fall back to older
YOLO_FALLBACKS = ["yolo26n", "yolo11n", "yolov8n"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export YOLO models to FP16 TFLite for mobile deployment."
    )
    parser.add_argument(
        "--binary-weights",
        type=str,
        default=None,
        help="Path to binary classifier weights (default: runs/binary/best.pt)",
    )
    parser.add_argument(
        "--detect-weights",
        type=str,
        default=None,
        help="Path to detection model weights (default: runs/detect/best.pt)",
    )
    parser.add_argument(
        "--classify-weights",
        type=str,
        default=None,
        help="Path to classify model weights (default: runs/classify/best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Export destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run a quick forward pass on exported models to verify they load",
    )
    return parser.parse_args()


def resolve_weights(stage: str, cli_override: str | None) -> Path | None:
    """
    Resolve the weights path for a pipeline stage.

    Priority: CLI override > default path. Returns None if file not found.
    """
    if cli_override:
        path = Path(cli_override)
        if not path.is_absolute():
            path = TRAINING_DIR / path
        if path.exists():
            return path
        log.warning("CLI weights path not found: %s", path)
        return None

    default_path = TRAINING_DIR / STAGES[stage]["default_weights"]
    if default_path.exists():
        return default_path

    log.warning(
        "Default weights not found for stage '%s': %s (skipping)",
        stage,
        default_path,
    )
    return None


def load_yolo_model(weights_path: Path):
    """
    Load a YOLO model from weights, with fallback chain for base models.

    When weights_path points to a trained checkpoint (best.pt), loads directly.
    For base/pretrained models, tries yolo26n -> yolo11n -> yolov8n.

    Returns:
        A YOLO model instance, or None if loading fails.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        log.error(
            "ultralytics is not installed. "
            "Install with: pip install ultralytics"
        )
        return None

    # If weights exist on disk, load them directly
    if weights_path.exists():
        try:
            model = YOLO(str(weights_path))
            log.info("Loaded model from %s", weights_path)
            return model
        except Exception as exc:
            log.error("Failed to load weights %s: %s", weights_path, exc)

    # Fallback: try base models (useful for testing export pipeline
    # before training runs produce checkpoints)
    for base_name in YOLO_FALLBACKS:
        try:
            model = YOLO(f"{base_name}.pt")
            log.info("Loaded fallback base model: %s", base_name)
            return model
        except Exception:
            continue

    log.error("Could not load any YOLO model for %s", weights_path)
    return None


def export_stage(
    stage: str,
    weights_path: Path,
    output_dir: Path,
) -> dict | None:
    """
    Export a single pipeline stage to FP16 TFLite.

    Args:
        stage: Pipeline stage name ('binary', 'detect', 'classify').
        weights_path: Path to the .pt checkpoint.
        output_dir: Directory to write the .tflite file.

    Returns:
        Metadata dict for the manifest, or None on failure.
    """
    imgsz = STAGES[stage]["imgsz"]
    log.info(
        "Exporting stage '%s' (imgsz=%d, FP16) from %s",
        stage,
        imgsz,
        weights_path,
    )

    model = load_yolo_model(weights_path)
    if model is None:
        return None

    try:
        # Export to TFLite with FP16 quantisation.
        # IMPORTANT: nms=False -- we do NMS in JS for portability.
        # IMPORTANT: half=True -- FP16 for size/accuracy tradeoff.
        exported_path = model.export(
            format="tflite",
            imgsz=imgsz,
            half=True,
            nms=False,
        )
        exported = Path(str(exported_path))

        if not exported.exists():
            # ultralytics sometimes returns the directory; find the .tflite
            search_dir = exported if exported.is_dir() else exported.parent
            tflite_files = list(search_dir.glob("*.tflite"))
            if not tflite_files:
                log.error("Export succeeded but no .tflite file found in %s", search_dir)
                return None
            exported = tflite_files[0]

        # Move to output directory with canonical name
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{stage}.tflite"
        exported.rename(dest)
        file_size = dest.stat().st_size

        log.info(
            "Exported %s -> %s (%.2f MB)",
            stage,
            dest,
            file_size / (1024 * 1024),
        )

        # Try to extract model metadata
        num_classes = None
        try:
            num_classes = model.model.nc if hasattr(model.model, "nc") else None
        except Exception:
            pass

        return {
            "id": f"pipeline-{stage}",
            "stage": stage,
            "version": "1.0.0",
            "file": str(dest.name),
            "sizeBytes": file_size,
            "format": "tflite",
            "quantisation": "fp16",
            "inputSize": imgsz,
            "numClasses": num_classes,
        }

    except Exception as exc:
        log.error("Export failed for stage '%s': %s", stage, exc)
        return None


def validate_exported_models(output_dir: Path, manifest: dict) -> bool:
    """
    Run a dummy forward pass on each exported model to verify it loads.

    Returns True if all models pass validation.
    """
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError:
        log.warning("numpy not installed; skipping validation")
        return False

    all_ok = True
    for entry in manifest.get("models", []):
        model_path = output_dir / entry["file"]
        imgsz = entry["inputSize"]

        if not model_path.exists():
            log.error("Validation: model file not found: %s", model_path)
            all_ok = False
            continue

        try:
            # Try loading with tflite_runtime first, fall back to tensorflow
            try:
                import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]
            except ImportError:
                import tensorflow.lite as tflite  # type: ignore[import-untyped]

            interpreter = tflite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Create dummy input matching expected shape
            input_shape = input_details[0]["shape"]
            dummy_input = np.zeros(input_shape, dtype=np.float32)
            interpreter.set_tensor(input_details[0]["index"], dummy_input)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]["index"])
            log.info(
                "Validation OK: %s -> output shape %s",
                entry["stage"],
                output_data.shape,
            )

        except Exception as exc:
            log.error("Validation FAILED for %s: %s", entry["stage"], exc)
            all_ok = False

    return all_ok


def main() -> int:
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = TRAINING_DIR / output_dir

    weight_overrides = {
        "binary": args.binary_weights,
        "detect": args.detect_weights,
        "classify": args.classify_weights,
    }

    manifest_entries: list[dict] = []
    skipped: list[str] = []

    for stage in STAGES:
        weights = resolve_weights(stage, weight_overrides[stage])
        if weights is None:
            skipped.append(stage)
            continue

        result = export_stage(stage, weights, output_dir)
        if result is not None:
            manifest_entries.append(result)
        else:
            skipped.append(stage)

    # Generate model manifest
    manifest = {
        "version": "1.0.0",
        "exportedAt": __import__("datetime").datetime.now().isoformat(),
        "pipeline": "three-stage",
        "quantisation": "fp16",
        "notes": (
            "FP16 TFLite models for react-native-fast-tflite. "
            "NMS is performed in JavaScript, not baked into the model."
        ),
        "models": manifest_entries,
    }

    manifest_path = output_dir / "model_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written to %s", manifest_path)

    # Summary
    if manifest_entries:
        log.info(
            "Successfully exported %d/%d stages",
            len(manifest_entries),
            len(STAGES),
        )
    if skipped:
        log.warning("Skipped stages (missing weights): %s", ", ".join(skipped))

    # Validation pass
    if args.validate and manifest_entries:
        log.info("Running validation on exported models...")
        ok = validate_exported_models(output_dir, manifest)
        if not ok:
            log.warning("Some models failed validation")
            return 1

    if not manifest_entries and len(skipped) == len(STAGES):
        log.warning(
            "No models exported. Provide weight files or train models first."
        )
        # Return 0 -- missing weights is expected before training
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
