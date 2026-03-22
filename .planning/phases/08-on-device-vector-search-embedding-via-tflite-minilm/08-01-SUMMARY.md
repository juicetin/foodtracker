---
phase: 08-on-device-vector-search-embedding-via-tflite-minilm
plan: 01
subsystem: ml-export
tags: [minilm, tflite, embedding, onnx2tf, sentence-transformers, vector-search]

requires:
  - phase: 02.5-food-knowledge-graph
    provides: "SentenceTransformer embeddings in build_kg.py for USDA food search"
provides:
  - "embedding.tflite: 22.1MB dynamic-range INT8 TFLite model producing 384-dim normalised vectors"
  - "vocab_embedding.json: 30522-entry WordPiece vocabulary for on-device tokenisation"
  - "model_manifest.json: updated with minilm-embedding-v1 entry"
  - "export_embedding.py: reproducible export pipeline with Docker-based conversion"
affects: [08-02, mobile-embedding-service, on-device-vector-search]

tech-stack:
  added: [onnx2tf, torch.onnx, python:3.11-slim Docker]
  patterns: [Docker-based onnx2tf with -rtpo for Flex op avoidance, dynamic-range INT8 quantisation for non-image models]

key-files:
  created:
    - training/export_embedding.py
    - apps/mobile/assets/data/vocab_embedding.json
  modified:
    - apps/mobile/assets/models/model_manifest.json

key-decisions:
  - "Dynamic-range INT8 quantisation (not full INT8) because model has int32 token inputs, not float32 images"
  - "python:3.11-slim Docker image (not tensorflow/tensorflow:2.18.0) because onnx2tf upgrades break pre-installed TF"
  - "-rtpo Erf GeLU flag to replace Flex ops with pseudo operators for TFLite-native compatibility"
  - "Manual L2 norm (sqrt+sum) instead of F.normalize to avoid onnx2tf tf.norm axis conversion bug"

patterns-established:
  - "Docker-based validation: when host Python lacks TFLite runtime, pipe inputs to Docker container for inference"
  - "onnx2tf -rtpo for BERT-family models: always replace Erf and GeLU to avoid FlexErf Select TF ops"

requirements-completed: [EMB-01, EMB-02]

duration: 49min
completed: 2026-03-22
---

# Phase 08 Plan 01: MiniLM TFLite Export Summary

**MiniLM-L6-v2 exported to 22.1MB dynamic-range INT8 TFLite with baked-in mean pooling and L2 norm, validated at >0.998 cosine similarity vs sentence-transformers**

## Performance

- **Duration:** 49 min
- **Started:** 2026-03-22T08:30:46Z
- **Completed:** 2026-03-22T09:20:15Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- SentenceEmbedder PyTorch wrapper with mean pooling + L2 normalisation baked into graph
- ONNX export (opset 13) -> Docker onnx2tf -> dynamic-range INT8 TFLite (22.1MB)
- WordPiece vocabulary extracted (30522 entries) to JSON for on-device tokenisation
- Docker-based TFLite validation confirms cosine sim > 0.998 for all 5 test food names
- Model manifest updated with minilm-embedding-v1 entry (stage: embedding, 384-dim, maxSeqLen 128)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create MiniLM TFLite export script with validation** - `a90832b6` (feat)

## Files Created/Modified
- `training/export_embedding.py` - Full export pipeline: ONNX export, Docker onnx2tf conversion, vocab extraction, validation, deployment
- `apps/mobile/assets/data/vocab_embedding.json` - WordPiece vocabulary (30522 entries, 519 KB)
- `apps/mobile/assets/models/model_manifest.json` - Added minilm-embedding-v1 entry
- `apps/mobile/assets/models/embedding.tflite` - 22.1MB dynamic-range INT8 model (gitignored, regenerated via export script)

## Decisions Made
- **Dynamic-range INT8 over full INT8:** Full INT8 quantisation (`-oiqt`) requires float32 4D image inputs for calibration. MiniLM has int32 token ID inputs, so dynamic-range quantisation (weights INT8, I/O float32) is the correct approach. Produces 22.1MB (vs 90MB float32, 45MB float16).
- **python:3.11-slim Docker image:** The tensorflow/tensorflow:2.18.0 image breaks when onnx2tf pip-installs incompatible TF versions. Clean Python 3.11 image avoids version conflicts.
- **-rtpo Erf GeLU flag:** Without this, onnx2tf produces TFLite models with FlexErf Select TF ops that require the Flex delegate at runtime. The -rtpo flag replaces these with TFLite-native pseudo operators.
- **Manual L2 norm computation:** torch.nn.functional.normalize produces an ONNX LpNormalization op that onnx2tf converts to tf.norm with axis=1, which fails on 2D tensors. Manual sqrt(sum(x*x)) uses simpler ops that convert cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Docker image compatibility with onnx2tf**
- **Found during:** Task 1 (Docker conversion)
- **Issue:** tensorflow/tensorflow:2.18.0 image has TF pre-installed; onnx2tf pip install upgrades TF, breaking native .so linkage (undefined symbol errors)
- **Fix:** Switched to python:3.11-slim image where onnx2tf installs its own compatible TF
- **Files modified:** training/export_embedding.py
- **Verification:** Docker conversion completes successfully

**2. [Rule 3 - Blocking] FlexErf TF Select ops in TFLite model**
- **Found during:** Task 1 (TFLite validation)
- **Issue:** GELU activation in BERT produces Erf ONNX op -> onnx2tf maps to FlexErf (Select TF op) requiring Flex delegate at inference time
- **Fix:** Added -rtpo Erf GeLU flag to onnx2tf to replace with TFLite-native pseudo operators
- **Files modified:** training/export_embedding.py
- **Verification:** TFLite model loads in standard interpreter without Flex delegate

**3. [Rule 3 - Blocking] tf.norm axis bug in onnx2tf conversion**
- **Found during:** Task 1 (ONNX->TFLite conversion)
- **Issue:** F.normalize -> ONNX LpNormalization -> onnx2tf tf.norm with axis=1 fails on 2D tensors
- **Fix:** Replaced F.normalize with manual sqrt(sum(x*x)) using basic ops
- **Files modified:** training/export_embedding.py
- **Verification:** Conversion succeeds, output matches sentence-transformers (cosine sim > 0.998)

**4. [Rule 3 - Blocking] No TFLite runtime for Python 3.14**
- **Found during:** Task 1 (validation step)
- **Issue:** Neither ai-edge-litert, tflite-runtime, nor tensorflow have wheels for Python 3.14
- **Fix:** Implemented Docker-based TFLite validation: tokenise on host, pipe inputs to Docker container, compare outputs with sentence-transformers on host
- **Files modified:** training/export_embedding.py
- **Verification:** Validation runs end-to-end, all cosine sims > 0.998

---

**Total deviations:** 4 auto-fixed (4 blocking)
**Impact on plan:** All auto-fixes necessary to overcome toolchain incompatibilities. No scope creep.

## Issues Encountered
- onnx2tf version management is fragile across TF Docker images -- pinning to clean Python image is more reliable
- Dynamic range quantisation (not full INT8) for non-image models is a pattern to document for future model exports

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TFLite model and vocabulary ready for mobile embedding service (Plan 02)
- Model requires TFLite interpreter without Flex delegate (pure TFLite ops)
- Note: embedding.tflite is gitignored; regenerate via `python training/export_embedding.py --docker --deploy`

---
*Phase: 08-on-device-vector-search-embedding-via-tflite-minilm*
*Completed: 2026-03-22*
