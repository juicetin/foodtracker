# Phase 1: Food Detection Foundation - Research

**Researched:** 2026-02-12
**Domain:** On-device food detection ML (YOLO, VLMs), recipe knowledge graphs, portion estimation
**Confidence:** MEDIUM-HIGH (stack well-verified, some dataset/training specifics need validation)

## Summary

This phase requires building a multi-stage food detection pipeline: binary food/not-food classification, object detection with bounding boxes, dish-level classification for hidden ingredient inference, and portion estimation from visual cues. The primary approach is YOLO-based on-device detection, with an on-device small VLM as the hybrid/benchmark alternative.

YOLO26 (released January 2026) is the clear choice for the detection backbone. Its NMS-free architecture exports cleanly to CoreML and TFLite, runs 43% faster on CPU than YOLO11-N, and the Ultralytics training pipeline is mature and well-documented. The biggest challenge is **dataset preparation** -- Food-101 and ISIA-500 are classification datasets (no bounding boxes), so auto-labeling or manual annotation is required to convert them for YOLO object detection training. The alternative is sourcing pre-annotated food detection datasets from Roboflow Universe, which have far fewer categories.

For the hidden ingredient knowledge graph, a dish-to-ingredient-to-nutrient structure built on SQLite with recursive CTEs is the pragmatic choice for a mobile-first app. RecipeNLG (2.2M recipes with structured ingredient lists) and Recipe1M+ provide the initial data, combined with USDA/AFCD nutrition entries for prepared foods. Neo4j is overkill for the scale of this graph (a few thousand dishes) and adds infrastructure complexity.

**Primary recommendation:** Use YOLO26-N for detection, YOLO26-N-cls for dish classification, auto-label datasets using Florence-2 or Grounding DINO for bounding box generation, benchmark against PaliGemma 2 3B (quantized) for the hybrid go/no-go decision, and build the recipe knowledge graph in SQLite.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Target accuracy: >95% correct identification** -- the bar is high; user wants to trust auto-logging with minimal review
- **No paid per-photo API calls** -- cloud LLM inference (OpenAI, Google, etc.) is unacceptable at any cost per image
- **All inference must run on-device** -- even if an LLM is used, it must be a small multimodal model running locally on the phone (e.g. PaliGemma, Florence-2, or similar)
- **YOLO is the preferred approach** for speed, reproducibility, and reliability
- **If YOLO hits 85% but LLM hits 97%, invest more in YOLO training first** -- don't pivot prematurely
- **Hybrid approach acceptable**: traditional ML + on-device small LLM, using whichever returns higher confidence per detection (not strictly fallback-only)
- **Benchmark on both**: public datasets (Food-101, ISIA-500 for breadth) AND user's real food photos (for real-world validation)
- **Traditional ML preferred over LLM** for speed and reproducibility -- LLM inference is slower, non-deterministic
- **Priority cuisines: Australian/Western + East Asian** (Chinese, Japanese, Korean, Vietnamese, Thai)
- **Category count: as many as possible** -- combine multiple datasets to maximize coverage
- **Best-guess for unknown foods** -- model should guess the closest known dish rather than saying "unrecognised"
- **Target accuracy: +/-10% weight estimation** when visual cues are available
- **Smart prompting for reference objects** -- suggest including a reference object if confidence is low
- **Smart fallback**: USDA standard serving size as default, extrapolate from user's previous servings if available
- **Full recipe breakdown** -- list all nutrition-significant ingredients, not just major components
- **Data sources**: public recipe databases + nutrition DB implied ingredients + user corrections
- **Full knowledge graph** -- structured dish->ingredient->nutrient relationships with variant tracking

### Claude's Discretion
- Knowledge graph database technology choice (Neo4j, SQLite with graph queries, etc.)
- Exact YOLO model variant selection (v8n, v11n, YOLO26, etc.)
- On-device LLM model selection for hybrid approach (PaliGemma, Florence-2, etc.)
- Training pipeline implementation (Ultralytics, custom PyTorch, etc.)
- Model quantisation strategy (INT8, FP16, etc.) for mobile deployment
- Benchmark evaluation framework design

### Deferred Ideas (OUT OF SCOPE)
- Cloud LLM via subscription (TOS concerns) -- documented in CONTEXT.md, ignore for this phase
</user_constraints>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Ultralytics | >=8.3 (YOLO26 support) | YOLO26 training, validation, export | De facto standard for YOLO training; supports detection + classification + export to CoreML/TFLite in one API |
| YOLO26-N (nano) | yolo26n.pt | On-device food detection (bounding boxes) | 2.4M params, 5.4B FLOPs, 38.9ms CPU, NMS-free, clean CoreML/TFLite export |
| YOLO26-N-cls | yolo26n-cls.pt | Dish-level classification (Food-101/ISIA-500 classes) | Same architecture for classification task; Ultralytics unified API |
| react-native-fast-tflite | ^1.6.1 | On-device TFLite inference in React Native | JSI zero-copy, GPU delegates (CoreML on iOS, NNAPI/OpenGL on Android), ~40-60% faster than bridge-based |
| PyTorch | >=2.0 | Training backend for YOLO and VLM fine-tuning | Required by Ultralytics and HuggingFace transformers |
| coremltools | >=7.0 | CoreML model conversion/validation for iOS | Apple's official converter; needed to validate CoreML output post-export |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PaliGemma 2 3B | google/paligemma2-3b-224 | Hybrid on-device VLM for benchmark comparison | Go/no-go benchmark; potential hybrid confidence-based routing |
| Florence-2-base | microsoft/Florence-2-base | Auto-labeling bounding boxes on classification datasets | 0.23B params; use during dataset prep to generate YOLO-format annotations from Food-101/ISIA-500 |
| Roboflow | (service) | Dataset management, augmentation, annotation format conversion | Merge multiple datasets into unified YOLO format with augmentation |
| Depth Anything V2 Small | depth-anything-v2-small | Monocular depth estimation for portion sizing | 25M params; Qualcomm AI Hub has TFLite-optimized version for Android |
| RecipeNLG | (dataset, 2.2M recipes) | Dish-to-ingredient mapping seed data | Initial population of knowledge graph |
| USDA FoodData Central API | v1 | Ingredient-to-nutrient lookup | Already used in spike; DEMO_KEY for dev, free key for production |
| SQLite | 3.x | Knowledge graph storage (dish->ingredient->nutrient) | Recursive CTEs for graph traversal; zero-dependency mobile deployment |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| YOLO26-N | YOLO11-N | 43% slower CPU inference; similar mAP (~39.9 vs 40.9); older DFL architecture complicates CoreML export |
| YOLO26-N | YOLOv8-N | Even older; YOLO26 is direct successor with cleaner export pipeline |
| PaliGemma 2 3B | Florence-2 (0.23B) | Much smaller but weaker at food recognition; ~1s on T4 GPU, likely 3-5s+ on mobile CPU; designed for GPU |
| SQLite | Neo4j | Overkill for ~2K-5K dish nodes; adds server dependency; SQLite is embedded and mobile-native |
| SQLite | SQLite-Graph (Cypher extension) | Interesting but immature; standard recursive CTEs are sufficient for dish->ingredient->nutrient traversal |
| react-native-fast-tflite | TensorFlow.js for RN | Much slower; no JSI zero-copy; bridge overhead kills real-time inference |

**Installation (training environment):**
```bash
pip install ultralytics>=8.3.0 torch>=2.0.0 torchvision>=0.15.0
pip install transformers accelerate  # For PaliGemma/Florence-2 benchmark
pip install coremltools>=7.0
pip install roboflow  # Dataset management
```

**Installation (React Native app):**
```bash
yarn add react-native-fast-tflite
# iOS: add to Podfile for CoreML delegate
# Android: configure gradle for GPU/NNAPI delegate
```

## Architecture Patterns

### Recommended Project Structure
```
spike/food-detection-poc/         # Existing notebook (keep for reference)
training/
├── datasets/
│   ├── food-binary/              # food vs not-food (binary classification)
│   ├── food-detection/           # merged YOLO-format detection dataset
│   ├── food-classification/      # Food-101 + ISIA-500 merged for cls
│   └── scripts/
│       ├── merge_datasets.py     # Combine Food-101, ISIA-500, Roboflow sources
│       ├── auto_label.py         # Florence-2 bounding box generation
│       └── audit_cuisines.py     # Verify Asian/Western coverage
├── configs/
│   ├── food-binary.yaml          # Binary food/not-food dataset config
│   ├── food-detect.yaml          # Detection dataset config
│   └── food-classify.yaml        # Classification dataset config
├── train_binary.py               # Train food/not-food classifier
├── train_detect.py               # Train YOLO26-N detector
├── train_classify.py             # Train YOLO26-N-cls dish classifier
├── benchmark.py                  # Unified benchmark: YOLO vs VLM
├── export_mobile.py              # Export to CoreML + TFLite + validate
└── evaluate/
    ├── eval_detection.py         # mAP, per-cuisine breakdown
    ├── eval_classification.py    # Top-1/Top-5 accuracy, confusion matrix
    └── eval_portion.py           # Weight estimation error analysis
knowledge-graph/
├── schema.sql                    # SQLite schema for dish->ingredient->nutrient
├── seed_recipenlg.py             # Parse RecipeNLG into graph
├── seed_usda.py                  # Import USDA prepared food ingredients
├── query.py                      # Graph traversal queries (recursive CTEs)
└── export_mobile.py              # Export as mobile-ready SQLite DB
apps/mobile/
├── src/ml/
│   ├── models/                   # .tflite / .mlmodel files
│   ├── FoodDetector.ts           # YOLO26 detection wrapper
│   ├── DishClassifier.ts         # YOLO26-cls wrapper
│   ├── PortionEstimator.ts       # Depth + geometry estimation
│   └── InferenceRouter.ts        # Hybrid confidence-based routing
└── src/data/
    └── food-knowledge.db         # SQLite knowledge graph (bundled)
```

### Pattern 1: Multi-Stage Detection Pipeline
**What:** Three-model pipeline: binary gate -> detection -> classification
**When to use:** Every food photo analysis

```python
# Stage 1: Binary food/not-food classification (fast gate)
binary_model = YOLO("yolo26n-cls-food-binary.pt")
result = binary_model.predict(image)
if result[0].probs.top1 != FOOD_CLASS:
    return {"is_food": False}

# Stage 2: Object detection (bounding boxes around food items)
detect_model = YOLO("yolo26n-food-detect.pt")
detections = detect_model.predict(image, conf=0.25)

# Stage 3: Dish classification (per-crop)
classify_model = YOLO("yolo26n-cls-food.pt")
for det in detections[0].boxes:
    crop = image[det.xyxy]  # crop bounding box region
    dish = classify_model.predict(crop)
    # -> Look up ingredients in knowledge graph
```

### Pattern 2: Hybrid YOLO + VLM Confidence Routing
**What:** Run YOLO pipeline first; if confidence < threshold, run on-device VLM for second opinion
**When to use:** Low-confidence detections or unusual foods

```python
# YOLO detection
yolo_result = detect_model.predict(image, conf=0.25)
yolo_confidence = max(box.conf for box in yolo_result[0].boxes)

if yolo_confidence < 0.6:  # tunable threshold
    # Fall back to on-device VLM
    vlm_result = paligemma_model.generate(image, "What food is in this image?")
    # Parse VLM response -> structured output
    return merge_results(yolo_result, vlm_result)
```

### Pattern 3: YOLO26 Export and Validation
**What:** Export to mobile formats and validate outputs match PyTorch
**When to use:** After training, before mobile deployment

```python
from ultralytics import YOLO

model = YOLO("runs/detect/food-detect/weights/best.pt")

# Export to CoreML (iOS)
model.export(format="coreml", int8=True, nms=False)  # NMS-free native in YOLO26

# Export to TFLite (Android)
model.export(format="tflite", int8=True)

# CRITICAL: Validate exported model outputs match PyTorch
import coremltools as ct
coreml_model = ct.models.MLModel("best.mlpackage")
# Compare predictions on 100 test images between PyTorch and CoreML
# Flag any image where IoU < 0.95 between PyTorch and CoreML boxes
```

### Pattern 4: SQLite Knowledge Graph with Recursive CTEs
**What:** Graph traversal for dish -> ingredients -> variants
**When to use:** Ingredient inference after dish classification

```sql
-- Schema
CREATE TABLE dishes (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    canonical_id INTEGER REFERENCES dishes(id),  -- variant-of relationship
    cuisine TEXT
);

CREATE TABLE ingredients (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    usda_fdc_id INTEGER
);

CREATE TABLE dish_ingredients (
    dish_id INTEGER REFERENCES dishes(id),
    ingredient_id INTEGER REFERENCES ingredients(id),
    weight_pct REAL,         -- proportion of total dish weight
    is_nutrition_significant BOOLEAN DEFAULT TRUE,
    source TEXT,             -- 'recipenlg', 'usda', 'user_correction'
    confidence REAL DEFAULT 0.5,
    PRIMARY KEY (dish_id, ingredient_id)
);

-- Recursive CTE: Find all ingredients for a dish including its canonical variant
WITH RECURSIVE dish_tree AS (
    SELECT id, canonical_id FROM dishes WHERE name = 'nasi goreng'
    UNION ALL
    SELECT d.id, d.canonical_id
    FROM dishes d JOIN dish_tree dt ON d.id = dt.canonical_id
)
SELECT DISTINCT i.name, di.weight_pct, di.source
FROM dish_tree dt
JOIN dish_ingredients di ON di.dish_id = dt.id
JOIN ingredients i ON i.id = di.ingredient_id
WHERE di.is_nutrition_significant = TRUE
ORDER BY di.weight_pct DESC;
```

### Anti-Patterns to Avoid
- **Training YOLO on classification-only data without bounding boxes:** Food-101 and ISIA-500 have no bounding box annotations. You MUST either auto-label with Florence-2/Grounding DINO or use pre-annotated detection datasets. Training YOLO detection on classification data will silently produce garbage.
- **Exporting CoreML with NMS pipeline:** YOLO26 is NMS-free. Older YOLO versions required `nms=False` explicitly in export. YOLO26 handles this natively, but if you accidentally use an older model, CoreML NMS pipeline export breaks Ultralytics inference.
- **Skipping export validation:** CoreML export can succeed without errors but produce incorrect/missing bounding boxes (documented Ultralytics issue #22309). Always compare PyTorch vs exported model on a test set.
- **Using Food-101 accuracy as detection accuracy:** Food-101 measures classification (one food per image, no localization). Real-world photos have multiple foods, overlapping items, partial views. Detection accuracy (mAP@0.5) on real photos will be significantly lower.
- **Building the knowledge graph in Neo4j:** For a few thousand dishes, Neo4j adds infrastructure overhead (server process, Java dependency) with no performance benefit. SQLite recursive CTEs handle this graph size trivially and ship embedded in the mobile app.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bounding box auto-labeling | Manual annotation of 500K+ images | Florence-2 or Grounding DINO auto-labeling pipeline | Would take months manually; Florence-2 at 0.23B params runs fast on GPU and produces YOLO-format boxes |
| Dataset format conversion | Custom parser for each dataset format | Roboflow Universe export (YOLO format) or FiftyOne | Handles augmentation, train/val/test splits, format conversion, deduplication |
| NMS post-processing | Custom NMS implementation for mobile | YOLO26's native NMS-free output | YOLO26 eliminates NMS entirely; older approaches needed custom post-processing on-device |
| Model quantization | Manual quantization with TFLite converter | Ultralytics `export(int8=True)` | Handles INT8 quantization with calibration automatically during export |
| Recipe ingredient parsing | Custom NLP parser for recipe text | RecipeNLG's pre-parsed `ingredients` and `ner` fields | 2.2M recipes already have structured ingredient lists with NER tags |
| Depth estimation model | Custom monocular depth network | Depth Anything V2 Small (25M params) | State-of-the-art; Qualcomm AI Hub provides pre-optimized TFLite version |
| Food/not-food classification | Train from scratch | Fine-tune YOLO26-N-cls on food/not-food dataset | Binary classification is trivial for YOLO; Roboflow has ready-made food/not-food datasets |

**Key insight:** The biggest time sink in this phase is **dataset preparation**, not model architecture. YOLO26 and Ultralytics handle training/export beautifully. The hard part is getting high-quality bounding-box-annotated food images spanning 500+ categories across Western and Asian cuisines.

## Common Pitfalls

### Pitfall 1: Classification Dataset Used for Detection Training
**What goes wrong:** Food-101 (101 classes, 101K images) and ISIA-500 (500 classes, 400K images) are classification datasets -- one label per image, no bounding boxes. Training YOLO detection on these without annotation conversion produces a model that cannot localize food.
**Why it happens:** These are the largest food datasets, so it's tempting to use them directly. YOLO's `train()` may not error if the data format is close enough but wrong.
**How to avoid:** Use Food-101 and ISIA-500 for the classification model (YOLO26-N-cls). For the detection model, either (a) auto-label with Florence-2 to generate bounding boxes, or (b) source pre-annotated detection datasets from Roboflow Universe.
**Warning signs:** YOLO training loss is unusually low or mAP is 0% on validation.

### Pitfall 2: CoreML Export Produces Silent Failures
**What goes wrong:** YOLO model exports to CoreML without errors, but the CoreML model produces incorrect bounding boxes, missing detections, or boxes outside image boundaries.
**Why it happens:** Documented in Ultralytics GitHub issues (#22309, #14668, #13794). CoreML export pipeline has historically had bugs with NMS handling, output tensor ordering, and coordinate system differences.
**How to avoid:** After every export, run a validation script comparing PyTorch predictions vs CoreML predictions on 100+ test images. Check IoU between corresponding boxes. Flag any discrepancy > 5%.
**Warning signs:** CoreML model returns 0 detections on images where PyTorch model returns many. Bounding box coordinates are negative or exceed image dimensions.

### Pitfall 3: Training Data Bias Causes 15-50% Calorie Errors for Non-Western Cuisines
**What goes wrong:** Western food datasets dominate, so Asian dish identification and ingredient inference are significantly less accurate. "Pad Thai" might be confused with "lo mein", and ingredient proportions (coconut milk vs soy sauce) get wrong macros.
**Why it happens:** Food-101 is heavily European/American. ISIA-500 has broader coverage but still underrepresents some cuisines.
**How to avoid:** Audit dataset cuisine distribution before training. Supplement with targeted Asian food datasets (search Roboflow Universe for "Asian food", "Chinese food", "Japanese food"). Track per-cuisine mAP during evaluation.
**Warning signs:** Overall mAP looks good but per-cuisine breakdown shows <60% for Asian categories.

### Pitfall 4: TFLite Flatbuffer Validation Failure
**What goes wrong:** Exported TFLite file doesn't start with 'TFL3' magic header, indicating an invalid flatbuffer. The TFLite interpreter may still load and allocate tensors successfully, but inference produces garbage.
**Why it happens:** Ultralytics TFLite export sometimes produces files with incorrect headers, especially with INT8 quantization.
**How to avoid:** After export, validate the TFLite file: check magic header, run inference on test images, compare against PyTorch output.
**Warning signs:** TFLite file size is suspiciously small. Inference produces all-zero or all-same-class outputs.

### Pitfall 5: PaliGemma/VLM Too Slow for On-Device
**What goes wrong:** PaliGemma 2 3B quantized to INT8 is still ~3GB model size and takes 5-15 seconds per inference on a mobile phone CPU.
**Why it happens:** Even at 3B parameters with INT8 quantization, VLMs are orders of magnitude slower than YOLO26-N (2.4M params).
**How to avoid:** Benchmark VLM inference time on actual target devices (iPhone 15 Pro, Pixel 8 Pro) early. Set a latency budget (e.g., <2 seconds per photo). If VLM exceeds budget, it can only be used as a background fallback, not primary.
**Warning signs:** VLM inference on mobile takes >5s. Memory usage exceeds 2GB during inference.

### Pitfall 6: Portion Estimation Without Reference Objects is Very Inaccurate
**What goes wrong:** Without a known-size reference object in the photo, volumetric estimation from bounding boxes alone has 30-50%+ error.
**Why it happens:** Camera distance, angle, lens distortion, and depth ambiguity make pixel-to-centimeter conversion unreliable without calibration.
**How to avoid:** Implement the smart fallback chain: (1) if reference object detected, use geometry; (2) if no reference, use USDA standard serving size; (3) if user has history for this food, use their average. Always show confidence level to user.
**Warning signs:** Weight estimates for the same food vary wildly across photos. Estimated weights are >2x or <0.5x actual weights.

## Code Examples

### YOLO26 Training for Food Detection
```python
# Source: Ultralytics docs + Roboflow YOLO26 blog
from ultralytics import YOLO

# Load pretrained YOLO26 nano for detection
model = YOLO("yolo26n.pt")

# Fine-tune on food detection dataset
results = model.train(
    data="configs/food-detect.yaml",  # path to dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    device="mps",  # or "cuda" or "cpu"
    patience=20,   # early stopping
    augment=True,
    # YOLO26-specific: no NMS config needed (NMS-free native)
)

# Validate
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

### YOLO26 Classification Training (Dish Classifier)
```python
from ultralytics import YOLO

# Load pretrained YOLO26 nano for classification
model = YOLO("yolo26n-cls.pt")

# Fine-tune on Food-101 + ISIA-500 merged dataset
results = model.train(
    data="datasets/food-classification/",  # ImageNet-style folder structure
    epochs=50,
    imgsz=224,
    batch=64,
    device="mps",
)

# Evaluate
metrics = model.val()
print(f"Top-1 accuracy: {metrics.top1:.3f}")
print(f"Top-5 accuracy: {metrics.top5:.3f}")
```

### Export and Validate for Mobile
```python
from ultralytics import YOLO
import coremltools as ct
import numpy as np

model = YOLO("runs/detect/food-detect/weights/best.pt")

# Export to CoreML (iOS)
model.export(format="coreml", int8=True)

# Export to TFLite (Android)
model.export(format="tflite", int8=True)

# CRITICAL VALIDATION: Compare PyTorch vs CoreML outputs
def validate_export(pytorch_model, coreml_path, test_images):
    """Ensure CoreML outputs match PyTorch within tolerance."""
    coreml_model = ct.models.MLModel(coreml_path)
    mismatches = 0

    for img_path in test_images:
        pt_results = pytorch_model.predict(img_path, verbose=False)
        # Run same image through CoreML
        # ... (platform-specific inference)
        # Compare: box count, class IDs, confidence ranges, IoU

    print(f"Mismatches: {mismatches}/{len(test_images)}")
    assert mismatches / len(test_images) < 0.05, "Export validation failed: >5% mismatch"
```

### Auto-Labeling with Florence-2
```python
# Source: Roboflow Florence-2 docs, HuggingFace Florence-2 page
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base")

def auto_label_food_image(image_path: str) -> list[dict]:
    """Generate YOLO-format bounding box annotations using Florence-2."""
    image = Image.open(image_path)

    # Object detection prompt
    inputs = processor(
        text="<OD>",  # Florence-2 object detection task token
        images=image,
        return_tensors="pt"
    )
    outputs = model.generate(**inputs, max_new_tokens=1024)
    result = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result, task="<OD>", image_size=image.size)

    # Convert to YOLO format: class x_center y_center width height (normalized)
    annotations = []
    for bbox, label in zip(parsed["bboxes"], parsed["labels"]):
        x1, y1, x2, y2 = bbox
        w, h = image.size
        annotations.append({
            "class": label,
            "x_center": ((x1 + x2) / 2) / w,
            "y_center": ((y1 + y2) / 2) / h,
            "width": (x2 - x1) / w,
            "height": (y2 - y1) / h,
        })
    return annotations
```

### React Native TFLite Inference
```typescript
// Source: react-native-fast-tflite README
import { loadTensorflowModel } from 'react-native-fast-tflite';

// Load the food detection model
const model = await loadTensorflowModel(
  require('../assets/models/yolo26n-food-detect.tflite'),
  'core-ml' // iOS: use CoreML delegate for GPU acceleration
);

// Run inference on an image buffer
// Input: Uint8Array of RGB pixels, resized to 640x640x3
const inputBuffer = preprocessImage(imageData, 640, 640);
const outputs = model.runSync([inputBuffer]);

// Post-process: YOLO26 is NMS-free, outputs are final detections
// Output tensor: [batch, num_detections, 6] -> [x1, y1, x2, y2, confidence, class_id]
const detections = parseYOLO26Output(outputs[0], {
  confidenceThreshold: 0.25,
  imageWidth: originalWidth,
  imageHeight: originalHeight,
});
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| YOLOv8 with NMS post-processing | YOLO26 NMS-free end-to-end | Jan 2026 | No custom NMS needed on-device; cleaner CoreML/TFLite export; 43% faster CPU |
| DFL (Distribution Focal Loss) box decoding | Lighter hardware-friendly regression | Jan 2026 (YOLO26) | Better quantization tolerance; cleaner export to all mobile formats |
| Manual bounding box annotation | Auto-labeling with Florence-2 / Grounding DINO | 2024-2025 | Makes it feasible to convert classification datasets to detection datasets at scale |
| Bridge-based TFLite in React Native | JSI zero-copy (react-native-fast-tflite) | 2024 | 40-60% faster inference; eliminates JSON serialization overhead |
| Simple ellipsoid volume estimation | Monocular depth (Depth Anything V2) + 3D reconstruction | 2024-2025 (MFP3D paper) | Much better portion estimation from single images; no reference object needed |
| FoodKG on SPARQL/RDF | SQLite with recursive CTEs or SQLite-Graph | 2024-2025 | Same graph capabilities without server infrastructure; mobile-native |

**Deprecated/outdated:**
- **NNAPI delegate on Android 15+:** NNAPI is deprecated as of Android 15. Use GPU delegate (OpenGL) instead for react-native-fast-tflite on newer Android devices.
- **YOLOv5/v7 for new projects:** Ultralytics has moved to YOLO26 as the recommended version. v5/v7 still work but receive no new features.
- **NMS=True CoreML export:** Causes `TypeError` in Ultralytics inference. YOLO26 is NMS-free natively; older models should export with `nms=False`.

## Open Questions

1. **YOLO26 food detection mAP on Asian cuisines**
   - What we know: YOLO26-N achieves 40.9% mAP on COCO. Fine-tuned YOLOv8 on food datasets has achieved ~93% mAP@0.5 in controlled settings. Research shows 15-50% calorie errors for non-Western cuisines in training-data-biased models.
   - What's unclear: No published benchmarks for YOLO26 fine-tuned specifically on multi-cuisine food detection. The mAP gap between Western and Asian food categories is unknown until we train and evaluate.
   - Recommendation: Train, then audit per-cuisine accuracy. Budget time for targeted data augmentation on underperforming cuisines.

2. **PaliGemma 2 3B on-device inference latency**
   - What we know: PaliGemma 2 3B with quantization shows "no practical quality difference" per Google. The model is 3B params. YOLO26-N is 2.4M params (1000x smaller).
   - What's unclear: Actual inference latency on iPhone 15 Pro / Pixel 8 Pro. No published mobile benchmarks found. Converting PaliGemma to TFLite/CoreML is non-trivial (no official mobile export path).
   - Recommendation: Benchmark PaliGemma on-device early. If >5s per image, it can only serve as async background fallback, not interactive hybrid routing. Consider Florence-2-base (0.23B) as lighter alternative but it may still be too slow on mobile CPU.

3. **ISIA Food-500 dataset availability and quality**
   - What we know: 500 categories, ~400K images, covers Eastern and Western cuisines. Download requires 10 compressed packages from a Chinese university server.
   - What's unclear: Current download availability (the server at 123.57.42.89 may be unreliable). Image quality and annotation consistency for all 500 categories.
   - Recommendation: Attempt download early. Have Roboflow Universe food datasets as backup. Food-101 (101 categories, reliable download from Hugging Face) is the guaranteed fallback.

4. **Depth Anything V2 on-device for portion estimation vs simpler heuristics**
   - What we know: Depth Anything V2 Small (25M params) produces high-quality monocular depth maps. Qualcomm AI Hub has a TFLite-optimized version. Recent research (MFP3D, 2025) shows monocular depth significantly improves portion estimation.
   - What's unclear: Whether the accuracy improvement over simple geometry-based estimation (current spike approach) justifies adding a second ML model to the on-device pipeline. Battery/memory impact.
   - Recommendation: Start with the spike's geometry approach (EXIF + reference objects + heuristics). Run Depth Anything V2 as a research experiment. If portion estimation is the bottleneck after YOLO fine-tuning, add depth estimation.

5. **RecipeNLG coverage of priority cuisines**
   - What we know: 2.2M recipes with structured ingredient lists. Sourced primarily from English-language recipe sites.
   - What's unclear: How many recipes cover Chinese, Japanese, Korean, Vietnamese, Thai dishes. English-language bias may underrepresent authentic Asian recipes (e.g., "pad thai" may have Americanized ingredient lists).
   - Recommendation: After importing RecipeNLG, audit cuisine distribution. Supplement with cuisine-specific recipe sources or manual curation for priority cuisines.

## Sources

### Primary (HIGH confidence)
- [Ultralytics YOLO26 official docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo26.md) - model variants, training API, export formats
- [Roboflow YOLO26 blog](https://blog.roboflow.com/yolo26/) - detailed benchmark table with params, FLOPs, mAP, inference speeds
- [Ultralytics BusinessWire release](https://www.businesswire.com/news/home/20260114168538/en/Ultralytics-Launches-YOLO26-Setting-a-New-Global-Standard-for-Edge-First-Vision-AI) - YOLO26 launch confirmation, Jan 14 2026
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite) - API, delegates, version, VisionCamera integration
- [PaliGemma 2 Google DeepMind](https://deepmind.google/models/gemma/paligemma-2/) - model sizes, capabilities
- [Florence-2 HuggingFace](https://huggingface.co/microsoft/Florence-2-base) - 0.23B params, task capabilities
- [RecipeNLG HuggingFace](https://huggingface.co/datasets/mbien/recipe_nlg) - 2.2M recipes, structured format
- [Food-101 HuggingFace](https://huggingface.co/datasets/ethz/food101) - 101 classes, 101K images
- [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2) - model sizes, capabilities
- [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) - already validated in spike

### Secondary (MEDIUM confidence)
- [YOLO26 arxiv paper](https://arxiv.org/abs/2509.25164) - architectural details, benchmark methodology
- [Ultralytics YOLO Evolution paper](https://arxiv.org/abs/2510.09653) - YOLO26 vs YOLO11 vs YOLOv8 comparison
- [Ultralytics CoreML export issue #22309](https://github.com/ultralytics/ultralytics/issues/22309) - silent failure documentation
- [ISIA Food-500 paper](https://arxiv.org/abs/2008.05655) - dataset description, 500 categories
- [FoodKG](https://foodkg.github.io/) - knowledge graph construction methodology
- [Food portion estimation survey](https://arxiv.org/html/2602.05078) - state of the art in portion estimation 2025
- [MFP3D paper](https://arxiv.org/abs/2411.10492) - monocular food portion estimation with 3D point clouds
- [Qualcomm AI Hub - Depth Anything V2](https://huggingface.co/qualcomm/Depth-Anything-V2) - mobile-optimized TFLite

### Tertiary (LOW confidence)
- ISIA Food-500 download server (http://123.57.42.89) - availability unverified
- PaliGemma 2 3B mobile inference latency - no published mobile benchmarks found
- Florence-2 mobile inference latency - described as "several seconds on CPU", no specific mobile numbers
- RecipeNLG Asian cuisine coverage - percentage unknown, needs validation after import

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - YOLO26 and Ultralytics are well-documented with official benchmarks; react-native-fast-tflite is actively maintained with clear API
- Architecture (detection pipeline): HIGH - three-stage pipeline (binary -> detection -> classification) is standard in food recognition literature
- Architecture (knowledge graph): MEDIUM - SQLite recursive CTEs are well-proven, but RecipeNLG data quality for Asian cuisines needs validation
- Dataset preparation: MEDIUM - auto-labeling with Florence-2 is documented but converting classification datasets to detection format at scale is not yet validated for this specific use case
- Portion estimation: MEDIUM - geometric approach from spike works but accuracy target (+/-10%) is aggressive; may need Depth Anything V2
- Pitfalls: HIGH - CoreML export failures and dataset bias are well-documented with specific GitHub issues and research papers
- On-device VLM benchmark: LOW - no published mobile benchmarks for PaliGemma 2 3B; on-device deployment path unclear

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable; YOLO26 just released, unlikely to be superseded soon)
