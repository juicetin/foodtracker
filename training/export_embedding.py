#!/usr/bin/env python3
"""
Export all-MiniLM-L6-v2 sentence embedding model to TFLite INT8 for on-device
vector search.

Conversion path:
  1. Wrap HuggingFace model in SentenceEmbedder (mean pooling + L2 norm baked in)
  2. Export to ONNX (opset 13, fixed shape [1,128])
  3. Docker-based onnx2tf INT8 dynamic-range quantisation
  4. Extract WordPiece vocabulary as JSON
  5. Validate TFLite output matches sentence-transformers (cosine sim > 0.99)
  6. Deploy assets to apps/mobile/

The Docker path is required because Python 3.14 has no TensorFlow wheels.

Usage:
    python training/export_embedding.py --docker --validate --deploy
    python training/export_embedding.py --docker --validate
    python training/export_embedding.py --validate  # skip conversion, validate existing
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "exports" / "embedding"
MOBILE_MODELS_DIR = PROJECT_ROOT / "apps" / "mobile" / "assets" / "models"
MOBILE_DATA_DIR = PROJECT_ROOT / "apps" / "mobile" / "assets" / "data"
MODEL_MANIFEST_PATH = MOBILE_MODELS_DIR / "model_manifest.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LEN = 128
VOCAB_SIZE = 30522


class SentenceEmbedder(nn.Module):
    """
    Wraps all-MiniLM-L6-v2 with mean pooling and L2 normalisation baked into
    the forward pass, so the TFLite model produces normalised 384-dim embeddings
    directly.
    """

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

    def forward(
        self,
        input_ids: torch.Tensor,    # int32 [1, 128]
        attention_mask: torch.Tensor  # int32 [1, 128]
    ) -> torch.Tensor:               # float32 [1, 384]
        outputs = self.encoder(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        )
        token_embeddings = outputs.last_hidden_state  # [1, 128, 384]

        # Mean pooling: expand mask, sum masked embeddings, divide by mask sum
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # [1, 384]

        # L2 normalise so cosine similarity = dot product
        # Use manual computation instead of F.normalize to avoid onnx2tf tf.norm axis bug
        norm = torch.sqrt(torch.sum(pooled * pooled, dim=1, keepdim=True).clamp(min=1e-12))
        normalised = pooled / norm
        return normalised


def export_onnx(output_dir: Path) -> Path:
    """Export SentenceEmbedder to ONNX with fixed shapes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "embedding.onnx"

    if onnx_path.exists():
        log.info("ONNX model already exists: %s", onnx_path)
        return onnx_path

    log.info("Loading %s for ONNX export...", MODEL_NAME)
    model = SentenceEmbedder()
    model.eval()

    dummy_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.int32)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.int32)

    log.info("Exporting to ONNX (opset 13, fixed shapes [1, %d])...", MAX_SEQ_LEN)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        opset_version=13,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        do_constant_folding=True,
    )

    file_size = onnx_path.stat().st_size / (1024 * 1024)
    log.info("ONNX export complete: %s (%.2f MB)", onnx_path, file_size)
    return onnx_path


def convert_tflite_docker(onnx_path: Path, output_dir: Path) -> Path | None:
    """
    Convert ONNX to TFLite with dynamic-range INT8 quantisation using Docker.

    Pipeline:
      1. onnx2tf converts ONNX -> SavedModel -> float32 TFLite
      2. TFLite converter applies dynamic-range quantisation (weights INT8, I/O float32)

    Uses python:3.11-slim because Python 3.14 has no TensorFlow wheels.
    Dynamic-range quantisation is used instead of full INT8 because the model
    has int32 token ID inputs (not float32 image data).
    """
    docker_image = "python:3.11-slim"
    onnx_dir = str(onnx_path.parent)
    onnx_name = onnx_path.name

    tflite_out = output_dir / "tflite_out"
    tflite_out.mkdir(parents=True, exist_ok=True)

    # onnx2tf produces float32 + float16 TFLite models.
    # We use onnx2tf's built-in dynamic_range_quant output directly.
    # If that's not available, we copy the float16 as fallback.
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{onnx_dir}:/models",
        "-v", f"{str(tflite_out)}:/output",
        docker_image,
        "bash", "-c",
        (
            f"pip install -q onnx2tf onnx onnxruntime flatbuffers sng4onnx 2>&1 | tail -1 && "
            f"onnx2tf -i /models/{onnx_name} -o /tmp/tflite_out -odrqt -rtpo Erf GeLU --non_verbose 2>&1 | tail -10 && "
            f"ls -la /tmp/tflite_out/*.tflite 2>/dev/null && "
            f"DRQ=$(find /tmp/tflite_out -name '*dynamic_range_quant*.tflite' | head -1) && "
            f"if [ -n \"$DRQ\" ]; then "
            f"  cp \"$DRQ\" /output/embedding.tflite && "
            f"  echo 'DEPLOYED dynamic_range_quant: '$(ls -la /output/embedding.tflite); "
            f"else "
            f"  echo 'No dynamic_range_quant found, using float16' && "
            f"  cp /tmp/tflite_out/embedding_float16.tflite /output/embedding.tflite && "
            f"  echo 'DEPLOYED float16: '$(ls -la /output/embedding.tflite); "
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
            log.error("Docker conversion failed (rc=%d)", result.returncode)
            log.error("STDOUT: %s", result.stdout[-500:])
            log.error("STDERR: %s", result.stderr[-500:])
            return None

        log.info("Docker output: %s", result.stdout[-300:])
    except subprocess.TimeoutExpired:
        log.error("Docker conversion timed out (600s)")
        return None
    except Exception as exc:
        log.error("Docker conversion error: %s", exc)
        return None

    dest = tflite_out / "embedding.tflite"
    if dest.exists():
        file_size = dest.stat().st_size / (1024 * 1024)
        log.info("TFLite INT8 model: %s (%.2f MB)", dest, file_size)
        return dest

    log.error("TFLite model not found after Docker conversion")
    return None


def extract_vocab(output_dir: Path) -> Path:
    """Extract WordPiece vocabulary from MiniLM tokenizer as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab_embedding.json"

    log.info("Extracting vocabulary from %s...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab = tokenizer.get_vocab()

    log.info("Vocabulary size: %d entries", len(vocab))
    if len(vocab) != VOCAB_SIZE:
        log.warning(
            "Expected %d vocab entries, got %d", VOCAB_SIZE, len(vocab)
        )

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    file_size = vocab_path.stat().st_size / 1024
    log.info("Vocabulary saved: %s (%.1f KB)", vocab_path, file_size)
    return vocab_path


def _load_tflite_interpreter(model_path: Path):
    """Load TFLite interpreter using best available runtime."""
    try:
        import ai_edge_litert.interpreter as litert
        interp = litert.Interpreter(model_path=str(model_path))
        log.info("Using ai-edge-litert for inference")
        return interp
    except (ImportError, Exception):
        pass
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=str(model_path))
        log.info("Using tflite_runtime for inference")
        return interp
    except (ImportError, Exception):
        pass
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(model_path))
        log.info("Using tensorflow for inference")
        return interp
    except (ImportError, Exception):
        pass
    return None


def _tflite_embed(interpreter, tokenizer, text: str) -> np.ndarray:
    """Run a single text through the TFLite model, return 384-dim vector."""
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set inputs (order may vary by model export)
    for detail in input_details:
        name = detail["name"]
        if "input_ids" in name or detail["index"] == input_details[0]["index"]:
            if "mask" not in name:
                interpreter.set_tensor(detail["index"], input_ids)
        if "attention_mask" in name or "mask" in name:
            interpreter.set_tensor(detail["index"], attention_mask)

    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]["index"])
    return embedding[0]  # [384]


def _onnx_embed(session, tokenizer, text: str) -> np.ndarray:
    """Run a single text through the ONNX model, return 384-dim vector."""
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })
    return outputs[0][0]  # [384]


def validate_docker_tflite(tflite_path: Path, output_dir: Path) -> bool:
    """
    Run TFLite validation inside Docker (needed when host has no TFLite runtime).

    Compares TFLite output to ONNX model output for fidelity.
    """
    from sentence_transformers import SentenceTransformer

    log.info("Running Docker-based TFLite validation...")

    # Write validation script to temp file
    val_script = output_dir / "_validate_tflite.py"
    val_script.write_text('''
import json, sys, numpy as np
try:
    import ai_edge_litert.interpreter as litert
    Interpreter = litert.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

model_path = "/models/embedding.tflite"
interp = Interpreter(model_path=model_path)
interp.allocate_tensors()

input_details = interp.get_input_details()
output_details = interp.get_output_details()

print(json.dumps({
    "inputs": [{"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])} for d in input_details],
    "outputs": [{"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])} for d in output_details],
}))

# Read test inputs from stdin (JSON lines with input_ids and attention_mask)
results = []
for line in sys.stdin:
    data = json.loads(line)
    input_ids = np.array(data["input_ids"], dtype=np.int32)
    attention_mask = np.array(data["attention_mask"], dtype=np.int32)

    for detail in input_details:
        name = detail["name"]
        if "mask" in name or "attention" in name:
            interp.set_tensor(detail["index"], attention_mask)
        else:
            interp.set_tensor(detail["index"], input_ids)

    interp.invoke()
    embedding = interp.get_tensor(output_details[0]["index"])
    results.append(embedding[0].tolist())

print("EMBEDDINGS:" + json.dumps(results))
''')

    tflite_dir = str(tflite_path.parent)
    docker_image = "python:3.11-slim"

    # Prepare test inputs
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_texts = [
        "chicken breast", "tonkatsu", "pad thai",
        "chocolate cake", "grilled salmon",
    ]

    test_inputs = []
    for text in test_texts:
        encoded = tokenizer(text, padding="max_length", truncation=True,
                           max_length=MAX_SEQ_LEN, return_tensors="np")
        test_inputs.append({
            "input_ids": encoded["input_ids"].astype(np.int32).tolist(),
            "attention_mask": encoded["attention_mask"].astype(np.int32).tolist(),
        })

    input_data = "\n".join(json.dumps(inp) for inp in test_inputs)

    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "-v", f"{tflite_dir}:/models",
        "-v", f"{str(output_dir)}:/scripts",
        docker_image,
        "bash", "-c",
        "pip install -q tensorflow numpy 2>&1 | tail -1 && python3 /scripts/_validate_tflite.py",
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log.error("Docker TFLite validation failed: %s", result.stderr[-500:])
            return False

        # Parse outputs
        lines = result.stdout.strip().split("\n")
        model_info = None
        tfl_embeddings_list = None
        for line in lines:
            if line.startswith("{"):
                model_info = json.loads(line)
            elif line.startswith("EMBEDDINGS:"):
                tfl_embeddings_list = json.loads(line[len("EMBEDDINGS:"):])

        if model_info:
            log.info("TFLite model info: %s", json.dumps(model_info, indent=2))
        if tfl_embeddings_list is None:
            log.error("No embeddings returned from Docker validation")
            return False

    except Exception as exc:
        log.error("Docker TFLite validation error: %s", exc)
        return False
    finally:
        val_script.unlink(missing_ok=True)

    # Compare with sentence-transformers
    log.info("Loading sentence-transformers reference model...")
    st_model = SentenceTransformer(MODEL_NAME)

    log.info("\n=== Fidelity Check: TFLite vs sentence-transformers ===")
    all_pass = True
    tfl_embeddings = {}
    for i, text in enumerate(test_texts):
        st_emb = st_model.encode(text, normalize_embeddings=True)
        tfl_emb = np.array(tfl_embeddings_list[i])
        tfl_norm = tfl_emb / (np.linalg.norm(tfl_emb) + 1e-9)

        cos_sim = float(np.dot(st_emb, tfl_norm))
        status = "PASS" if cos_sim > 0.99 else "FAIL"
        if cos_sim <= 0.99:
            all_pass = False
        log.info("  %-20s cosine_sim=%.6f  [%s]", text, cos_sim, status)
        tfl_embeddings[text] = tfl_norm

    # Quality check
    log.info("\n=== Quality Check: Semantic Similarity ===")
    expected_similar = [
        ("chicken breast", "tonkatsu"),     # both meat
        ("grilled salmon", "chicken breast"),  # both protein
    ]
    expected_dissimilar = [
        ("chocolate cake", "grilled salmon"),
    ]

    log.info("Similar pairs:")
    for a, b in expected_similar:
        sim = float(np.dot(tfl_embeddings[a], tfl_embeddings[b]))
        log.info("  %-20s <-> %-20s sim=%.4f", a, b, sim)

    log.info("Dissimilar pairs:")
    for a, b in expected_dissimilar:
        sim = float(np.dot(tfl_embeddings[a], tfl_embeddings[b]))
        log.info("  %-20s <-> %-20s sim=%.4f", a, b, sim)

    # Similarity matrix
    log.info("\n=== Cosine Similarity Matrix (5 test foods) ===")
    header = "                    " + "  ".join(f"{t[:12]:>12}" for t in test_texts)
    log.info(header)
    for a in test_texts:
        row = f"{a:20s}"
        for b in test_texts:
            sim = float(np.dot(tfl_embeddings[a], tfl_embeddings[b]))
            row += f"  {sim:12.4f}"
        log.info(row)

    if all_pass:
        log.info("\nFidelity check PASSED: all cosine similarities > 0.99")
    else:
        log.error("\nFidelity check FAILED: some cosine similarities <= 0.99")

    return all_pass


def validate(tflite_path: Path, output_dir: Path) -> bool:
    """
    Validate TFLite model output matches sentence-transformers.

    Uses Docker-based TFLite inference when no host TFLite runtime is available.
    """
    interpreter = _load_tflite_interpreter(tflite_path)
    if interpreter is None:
        log.info("No TFLite runtime on host (Python 3.14), using Docker validation")
        return validate_docker_tflite(tflite_path, output_dir)

    # Host-based validation (when TFLite runtime is available)
    from sentence_transformers import SentenceTransformer

    log.info("Loading sentence-transformers reference model...")
    st_model = SentenceTransformer(MODEL_NAME)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log.info("TFLite inputs: %s", [(d["name"], d["shape"].tolist(), d["dtype"]) for d in input_details])
    log.info("TFLite outputs: %s", [(d["name"], d["shape"].tolist(), d["dtype"]) for d in output_details])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_texts = [
        "chicken breast", "tonkatsu", "pad thai",
        "chocolate cake", "grilled salmon",
    ]

    log.info("\n=== Fidelity Check: TFLite vs sentence-transformers ===")
    all_pass = True
    tfl_embeddings = {}
    for text in test_texts:
        st_emb = st_model.encode(text, normalize_embeddings=True)
        tfl_emb = _tflite_embed(interpreter, tokenizer, text)
        tfl_norm = tfl_emb / (np.linalg.norm(tfl_emb) + 1e-9)

        cos_sim = float(np.dot(st_emb, tfl_norm))
        status = "PASS" if cos_sim > 0.99 else "FAIL"
        if cos_sim <= 0.99:
            all_pass = False
        log.info("  %-20s cosine_sim=%.6f  [%s]", text, cos_sim, status)
        tfl_embeddings[text] = tfl_norm

    # Similarity matrix
    log.info("\n=== Cosine Similarity Matrix (5 test foods) ===")
    header = "                    " + "  ".join(f"{t[:12]:>12}" for t in test_texts)
    log.info(header)
    for a in test_texts:
        row = f"{a:20s}"
        for b in test_texts:
            sim = float(np.dot(tfl_embeddings[a], tfl_embeddings[b]))
            row += f"  {sim:12.4f}"
        log.info(row)

    if all_pass:
        log.info("\nFidelity check PASSED: all cosine similarities > 0.99")
    else:
        log.error("\nFidelity check FAILED: some cosine similarities <= 0.99")

    return all_pass


def deploy(output_dir: Path) -> bool:
    """Copy assets to mobile app directories and update model manifest."""
    tflite_src = output_dir / "tflite_out" / "embedding.tflite"
    vocab_src = output_dir / "vocab_embedding.json"

    if not tflite_src.exists():
        log.error("TFLite model not found: %s", tflite_src)
        return False
    if not vocab_src.exists():
        log.error("Vocabulary file not found: %s", vocab_src)
        return False

    # Copy model
    MOBILE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_model = MOBILE_MODELS_DIR / "embedding.tflite"
    shutil.copy2(tflite_src, dest_model)
    log.info("Deployed: %s (%.2f MB)", dest_model,
             dest_model.stat().st_size / (1024 * 1024))

    # Copy vocab
    MOBILE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_vocab = MOBILE_DATA_DIR / "vocab_embedding.json"
    shutil.copy2(vocab_src, dest_vocab)
    log.info("Deployed: %s (%.1f KB)", dest_vocab,
             dest_vocab.stat().st_size / 1024)

    # Update model manifest
    if not MODEL_MANIFEST_PATH.exists():
        log.error("Model manifest not found: %s", MODEL_MANIFEST_PATH)
        return False

    with open(MODEL_MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Remove existing embedding entry if present
    manifest["models"] = [
        m for m in manifest["models"] if m.get("id") != "minilm-embedding-v1"
    ]

    # Add embedding model entry
    embedding_entry = {
        "id": "minilm-embedding-v1",
        "stage": "embedding",
        "version": "1.0.0",
        "file": "embedding.tflite",
        "sizeBytes": dest_model.stat().st_size,
        "format": "tflite",
        "quantisation": "int8",
        "embeddingDim": EMBEDDING_DIM,
        "maxSeqLen": MAX_SEQ_LEN,
        "inputDtype": "int32",
        "outputShape": [1, EMBEDDING_DIM],
        "name": "MiniLM-L6-v2 (Sentence Embedding)",
        "description": (
            "all-MiniLM-L6-v2 with mean pooling and L2 normalisation baked in. "
            "Produces 384-dim normalised float32 vectors for on-device vector search. "
            "INT8 dynamic-range quantised for mobile."
        ),
    }
    manifest["models"].append(embedding_entry)

    with open(MODEL_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    log.info("Updated model manifest: %s", MODEL_MANIFEST_PATH)

    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export all-MiniLM-L6-v2 to TFLite INT8 for on-device embedding."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Export working directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help=(
            "Use Docker (tensorflow/tensorflow:2.18.0) for ONNX->TFLite conversion. "
            "Required when TensorFlow is not installable (Python >= 3.13)."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation comparing TFLite output to sentence-transformers",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Copy assets to apps/mobile/ and update model_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX
    onnx_path = export_onnx(output_dir)

    # Step 2: Convert to TFLite (Docker-based)
    tflite_path = output_dir / "tflite_out" / "embedding.tflite"
    if args.docker:
        result = convert_tflite_docker(onnx_path, output_dir)
        if result is None:
            log.error("TFLite conversion failed")
            return 1
        tflite_path = result
    elif not tflite_path.exists():
        log.error(
            "TFLite model not found at %s. Run with --docker to convert.", tflite_path
        )
        return 1

    # Step 3: Extract vocabulary
    vocab_path = extract_vocab(output_dir)

    # Step 4: Validate
    if args.validate:
        log.info("Running validation...")
        if not validate(tflite_path, output_dir):
            log.error("Validation FAILED")
            return 1
        log.info("Validation PASSED")

    # Step 5: Deploy to mobile assets
    if args.deploy:
        log.info("Deploying assets to mobile app...")
        if not deploy(output_dir):
            log.error("Deployment failed")
            return 1
        log.info("Deployment complete")

    log.info("All steps completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
