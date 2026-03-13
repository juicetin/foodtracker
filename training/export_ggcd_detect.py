#!/usr/bin/env python3
"""
Export GGCD YOLOv8n detection model to TFLite for mobile deployment.

Single-purpose script for the GGCD (Global Gastronomic Culinary Dataset) YOLOv8n
model with 241 food-specific classes. Separate from export_mobile.py (which handles
the general 3-stage pipeline).

Conversion path:
  1. Ultralytics direct TFLite export with INT8 quantization (needs TensorFlow)
  2. ONNX export -> onnx2tf with INT8 post-training quantization (needs onnx2tf)
  3. Docker-based ONNX -> onnx2tf (recommended when Python >= 3.13, no TF wheels)
  4. Ultralytics FP16 export (last resort, with warning)

The Docker path (--docker) is recommended on systems where TensorFlow is not
available (e.g. Python 3.13+). It uses the tensorflow/tensorflow:2.18.0 image
with onnx2tf installed inside the container.

Per project conventions:
  - nms=False (NMS performed in JavaScript for cross-platform portability)
  - Input size 640x640
  - Output: [1, 245, 8400] (4 bbox + 241 classes)

Usage:
    python training/export_ggcd_detect.py --validate
    python training/export_ggcd_detect.py --quantization int8 --validate
    python training/export_ggcd_detect.py --quantization fp16 --validate
    python training/export_ggcd_detect.py --docker --validate
    python training/export_ggcd_detect.py --weights /path/to/model.pt --output-dir exports/
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_WEIGHTS = Path(
    "/home/me/media/foodtracker-ml/models/ggcd-yolo/yolov8n_ggcd.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "apps" / "mobile" / "assets" / "models"
GGCD_CLASSES_PATH = Path(
    "/home/me/media/foodtracker-ml/models/ggcd-yolo/ggcd_classes.json"
)

INPUT_SIZE = 640
EXPECTED_NUM_CLASSES = 241


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export GGCD YOLOv8n to TFLite for mobile deployment."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to GGCD YOLOv8n PyTorch weights (default: {DEFAULT_WEIGHTS})",
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
        help="Run validation on the exported TFLite model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int8", "fp16"],
        default="int8",
        help="Quantization mode: int8 (default) or fp16",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help=(
            "Use Docker (tensorflow/tensorflow:2.18.0) for ONNX->TFLite conversion. "
            "Recommended when TensorFlow is not installable (Python >= 3.13)."
        ),
    )
    return parser.parse_args()


def export_tflite_int8(weights_path: Path, output_dir: Path) -> Path | None:
    """
    Export GGCD YOLOv8n to INT8 TFLite via ultralytics direct export.

    Uses ultralytics built-in INT8 quantization with coco128 calibration data.
    Falls back to ONNX -> onnx2tf if direct export fails.

    Returns:
        Path to the exported .tflite file, or None on failure.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(str(weights_path))
    log.info("Loaded GGCD YOLOv8n model from %s", weights_path)
    log.info("Model classes: %d", model.model.nc if hasattr(model.model, "nc") else -1)

    # Attempt 1: ultralytics direct INT8 TFLite export
    log.info("Attempting ultralytics direct INT8 TFLite export...")
    try:
        exported_path = model.export(
            format="tflite",
            imgsz=INPUT_SIZE,
            int8=True,
            nms=False,
        )
        tflite_path = _find_tflite(Path(str(exported_path)))
        if tflite_path:
            log.info("INT8 TFLite export succeeded: %s", tflite_path)
            return tflite_path
    except Exception as exc:
        log.warning("Ultralytics direct INT8 export failed: %s", exc)

    # Attempt 2: ONNX -> onnx2tf with INT8
    log.info("Attempting ONNX -> onnx2tf INT8 export...")
    try:
        onnx_path = model.export(format="onnx", imgsz=INPUT_SIZE, nms=False)
        onnx_path = Path(str(onnx_path))
        if onnx_path.exists():
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onnx2tf",
                    "-i",
                    str(onnx_path),
                    "-o",
                    str(output_dir / "onnx2tf_out"),
                    "-oiqt",  # INT8 quantization
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                onnx2tf_dir = output_dir / "onnx2tf_out"
                tflite_path = _find_tflite(onnx2tf_dir)
                if tflite_path:
                    log.info("onnx2tf INT8 export succeeded: %s", tflite_path)
                    return tflite_path
            else:
                log.warning("onnx2tf failed: %s", result.stderr[:500])
    except Exception as exc:
        log.warning("ONNX -> onnx2tf INT8 export failed: %s", exc)

    return None


def export_tflite_fp16(weights_path: Path) -> Path | None:
    """
    Export GGCD YOLOv8n to FP16 TFLite via ultralytics direct export.

    Returns:
        Path to the exported .tflite file, or None on failure.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(str(weights_path))
    log.info("Attempting FP16 TFLite export...")

    try:
        exported_path = model.export(
            format="tflite",
            imgsz=INPUT_SIZE,
            half=True,
            nms=False,
        )
        tflite_path = _find_tflite(Path(str(exported_path)))
        if tflite_path:
            log.info("FP16 TFLite export succeeded: %s", tflite_path)
            return tflite_path
    except Exception as exc:
        log.error("FP16 TFLite export failed: %s", exc)

    return None


def _find_tflite(path: Path) -> Path | None:
    """Find a .tflite file from an export result path."""
    if path.is_file() and path.suffix == ".tflite":
        return path

    # ultralytics sometimes returns a directory or the saved_model path
    search_dir = path if path.is_dir() else path.parent
    tflite_files = sorted(search_dir.rglob("*.tflite"))
    if tflite_files:
        return tflite_files[0]

    return None


def validate_tflite(model_path: Path) -> dict | None:
    """
    Validate the exported TFLite model by loading and inspecting shapes.

    Returns:
        Dict with input/output details, or None on failure.
    """
    import numpy as np  # type: ignore[import-untyped]

    # Try ai-edge-litert first, then tflite_runtime, then tensorflow
    interpreter = None
    runtime_name = None
    try:
        import ai_edge_litert.interpreter as litert  # type: ignore[import-untyped]

        interpreter = litert.Interpreter(model_path=str(model_path))
        runtime_name = "ai-edge-litert"
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]

            interpreter = tflite.Interpreter(model_path=str(model_path))
            runtime_name = "tflite_runtime"
        except ImportError:
            try:
                import tensorflow as tf  # type: ignore[import-untyped]

                interpreter = tf.lite.Interpreter(model_path=str(model_path))
                runtime_name = "tensorflow"
            except ImportError:
                log.warning(
                    "No TFLite runtime available (ai-edge-litert, tflite_runtime, "
                    "or tensorflow). Skipping runtime validation."
                )
                # Fall back to file-level validation only
                return _validate_file_only(model_path)

    log.info("Using %s for validation", runtime_name)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = tuple(input_details[0]["shape"])
    input_dtype = input_details[0]["dtype"]
    output_shape = tuple(output_details[0]["shape"])

    log.info("Input shape: %s, dtype: %s", input_shape, input_dtype)
    log.info("Output shape: %s", output_shape)

    # Check input shape: expect [1, 640, 640, 3] (NHWC)
    if input_shape != (1, INPUT_SIZE, INPUT_SIZE, 3):
        log.warning(
            "Unexpected input shape: %s (expected [1, %d, %d, 3])",
            input_shape,
            INPUT_SIZE,
            INPUT_SIZE,
        )

    # Check output contains 245 (4 bbox + 241 classes) in one dimension
    expected_dim = 4 + EXPECTED_NUM_CLASSES  # 245
    if expected_dim not in output_shape:
        log.warning(
            "Output shape %s does not contain expected dimension %d "
            "(4 bbox coords + %d classes)",
            output_shape,
            expected_dim,
            EXPECTED_NUM_CLASSES,
        )

    # Determine quantization type
    quant_type = "unknown"
    if input_dtype == np.float32:
        quant_type = "float32 input (may be INT8 with quantize/dequantize ops)"
    elif input_dtype == np.uint8:
        quant_type = "INT8 (uint8 input)"
    elif input_dtype == np.float16:
        quant_type = "FP16"

    # Run dummy inference
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    if input_dtype == np.uint8:
        dummy_input = np.zeros(input_shape, dtype=np.uint8)

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    log.info(
        "Inference OK: output shape %s, quantization: %s",
        output_data.shape,
        quant_type,
    )

    return {
        "input_shape": list(input_shape),
        "input_dtype": str(input_dtype),
        "output_shape": list(output_shape),
        "quantization": quant_type,
    }


def _validate_file_only(model_path: Path) -> dict:
    """File-level validation when no TFLite runtime is available."""
    file_size = model_path.stat().st_size
    log.info(
        "File-level validation: %s (%.2f MB)",
        model_path.name,
        file_size / (1024 * 1024),
    )
    # Check file header for TFLite magic bytes
    with open(model_path, "rb") as f:
        header = f.read(8)
    # FlatBuffer files start with specific bytes; TFLite uses FlatBuffers
    if len(header) >= 4:
        log.info("File header (hex): %s", header[:8].hex())

    return {
        "input_shape": "unknown (no runtime)",
        "input_dtype": "unknown",
        "output_shape": "unknown",
        "quantization": "unknown (no runtime to inspect)",
    }


def export_tflite_docker(
    weights_path: Path, output_dir: Path, quantization: str = "int8"
) -> tuple[Path | None, str]:
    """
    Export GGCD YOLOv8n to TFLite using Docker container.

    Pipeline: PyTorch -> ONNX (via ultralytics) -> TFLite (via onnx2tf in Docker).
    This is the recommended path when TensorFlow is not installable on the host
    (e.g. Python >= 3.13 where no TF wheels exist).

    Args:
        weights_path: Path to the GGCD YOLOv8n PyTorch weights.
        output_dir: Directory to write the .tflite file.
        quantization: 'int8' or 'fp16'.

    Returns:
        Tuple of (path to exported .tflite, quantization used) or (None, '').
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    # Step 1: Export to ONNX using ultralytics (works without TF)
    model = YOLO(str(weights_path))
    log.info("Loaded GGCD YOLOv8n model from %s", weights_path)
    log.info(
        "Model classes: %d", model.model.nc if hasattr(model.model, "nc") else -1
    )

    onnx_path = weights_path.with_suffix(".onnx")
    if not onnx_path.exists():
        log.info("Exporting to ONNX first...")
        try:
            exported = model.export(format="onnx", imgsz=INPUT_SIZE, nms=False)
            onnx_path = Path(str(exported))
        except Exception as exc:
            log.error("ONNX export failed: %s", exc)
            return None, ""

    if not onnx_path.exists():
        log.error("ONNX file not found after export: %s", onnx_path)
        return None, ""

    log.info("ONNX model ready: %s", onnx_path)

    # Step 2: Convert ONNX -> TFLite using Docker
    docker_image = "tensorflow/tensorflow:2.18.0"
    onnx_dir = str(onnx_path.parent)
    onnx_name = onnx_path.name

    quant_flag = "-oiqt" if quantization == "int8" else ""
    # Select the right output model based on quantization
    if quantization == "int8":
        model_pattern = "dynamic_range_quant"
    else:
        model_pattern = "float16"

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{onnx_dir}:/models",
        "-v",
        f"{output_dir}:/output",
        docker_image,
        "bash",
        "-c",
        (
            f"pip install -q onnx2tf onnx onnxruntime flatbuffers 2>&1 | tail -1 && "
            f"onnx2tf -i /models/{onnx_name} -o /tmp/tflite_out "
            f"{quant_flag} --non_verbose 2>&1 | tail -5 && "
            f"MODEL=$(find /tmp/tflite_out -name '*{model_pattern}*.tflite' | head -1) && "
            f"if [ -n \"$MODEL\" ]; then "
            f"  cp \"$MODEL\" /output/detect.tflite && "
            f"  echo 'DEPLOYED: '$(ls -la /output/detect.tflite); "
            f"else "
            f"  FALLBACK=$(find /tmp/tflite_out -name '*.tflite' | head -1) && "
            f"  cp \"$FALLBACK\" /output/detect.tflite && "
            f"  echo 'DEPLOYED (fallback): '$(ls -la /output/detect.tflite); "
            f"fi"
        ),
    ]

    log.info("Running Docker conversion: %s", docker_image)
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            log.error("Docker conversion failed: %s", result.stderr[:500])
            return None, ""

        log.info("Docker output: %s", result.stdout[-200:])
    except Exception as exc:
        log.error("Docker conversion error: %s", exc)
        return None, ""

    dest = output_dir / "detect.tflite"
    if dest.exists():
        log.info(
            "Docker export succeeded: %s (%.2f MB)",
            dest,
            dest.stat().st_size / (1024 * 1024),
        )
        return dest, quantization

    return None, ""


def deploy_labels(output_dir: Path) -> bool:
    """
    Copy GGCD class labels to the output directory.

    Returns True on success.
    """
    if not GGCD_CLASSES_PATH.exists():
        log.error("GGCD classes file not found: %s", GGCD_CLASSES_PATH)
        return False

    with open(GGCD_CLASSES_PATH) as f:
        classes_data = json.load(f)

    if classes_data.get("count") != EXPECTED_NUM_CLASSES:
        log.error(
            "Expected %d classes, got %d",
            EXPECTED_NUM_CLASSES,
            classes_data.get("count", 0),
        )
        return False

    if len(classes_data.get("classNames", [])) != EXPECTED_NUM_CLASSES:
        log.error(
            "Expected %d class names, got %d",
            EXPECTED_NUM_CLASSES,
            len(classes_data.get("classNames", [])),
        )
        return False

    dest = output_dir / "labels_detect.json"
    shutil.copy2(GGCD_CLASSES_PATH, dest)
    log.info("Deployed labels to %s (%d classes)", dest, EXPECTED_NUM_CLASSES)
    return True


def main() -> int:
    """Main entry point."""
    args = parse_args()
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    if not weights_path.exists():
        log.error("Weights file not found: %s", weights_path)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export TFLite
    quantization_used = args.quantization
    tflite_path = None

    if args.docker:
        # Docker-based export (recommended for Python >= 3.13)
        tflite_path, quantization_used = export_tflite_docker(
            weights_path, output_dir, args.quantization
        )
    elif args.quantization == "int8":
        tflite_path = export_tflite_int8(weights_path, output_dir)
        if tflite_path is None:
            log.warning(
                "INT8 export failed, falling back to FP16. "
                "Re-attempt INT8 with Docker onnx2tf container if needed."
            )
            tflite_path = export_tflite_fp16(weights_path)
            quantization_used = "fp16"
    else:
        tflite_path = export_tflite_fp16(weights_path)

    if tflite_path is None:
        log.error("All export attempts failed")
        return 1

    # Move to output directory with canonical name (Docker export writes directly)
    dest = output_dir / "detect.tflite"
    if tflite_path != dest:
        shutil.copy2(tflite_path, dest)

    file_size = dest.stat().st_size
    log.info(
        "Deployed detect.tflite -> %s (%.2f MB, %s)",
        dest,
        file_size / (1024 * 1024),
        quantization_used,
    )

    # Deploy labels
    if not deploy_labels(output_dir):
        return 1

    # Validation
    if args.validate:
        log.info("Validating exported model...")
        result = validate_tflite(dest)
        if result is None:
            log.error("Validation failed")
            return 1
        log.info("Validation result: %s", json.dumps(result, indent=2))

    log.info("Export complete. Quantization: %s", quantization_used)
    log.info("Model size: %.2f MB", file_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
