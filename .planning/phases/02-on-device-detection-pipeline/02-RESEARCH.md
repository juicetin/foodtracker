# Phase 2: On-Device Detection Pipeline - Research

**Researched:** 2026-03-12
**Domain:** On-device ML inference (YOLO training, model export, React Native ML integration, detection UI)
**Confidence:** HIGH

## Summary

Phase 2 builds the complete food detection pipeline from YOLO model training through to a user-facing detection results screen. The carried-forward work from Phase 1 (training scripts 01-03, model export 01-05, mobile ML integration 01-06) provides the starting points. The core technical challenge is getting trained YOLO models exported to mobile formats (CoreML for iOS, TFLite for Android), loaded via react-native-fast-tflite, and wiring inference results through to an interactive bounding box UI with confidence indicators and portion estimation.

The project uses Ultralytics YOLO26-N models with a three-stage pipeline (binary gate, detection, classification) that already has training scripts in place. Export uses `model.export(format='tflite')` and `model.export(format='coreml')` from Ultralytics. On the mobile side, react-native-fast-tflite v2.0.0 provides cross-platform TFLite inference with CoreML delegate on iOS and GPU/NNAPI delegates on Android, loading models from filesystem paths (enabling PackManager integration). The detection UI uses existing installed libraries (@gorhom/bottom-sheet, react-native-reanimated, react-native-gesture-handler).

**Primary recommendation:** Use react-native-fast-tflite v2.0.0 as the unified inference runtime for both platforms, with CoreML delegate enabled on iOS and GPU delegate on Android. Export YOLO models to TFLite format only (not separate CoreML .mlpackage), since react-native-fast-tflite's CoreML delegate handles iOS hardware acceleration transparently. This avoids maintaining two separate model formats and two separate inference code paths.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Confidence-colored bounding box outlines with floating label chips (food name + confidence %) above each box
- Box colors map to confidence: green (>=80%), yellow (50-79%), red (<50%)
- Tapping a bounding box or list item opens a detail card via @gorhom/bottom-sheet showing: food name, confidence, portion estimate, macros preview, edit button
- Annotated photo is interactive: pinch-to-zoom and pan (react-native-gesture-handler already installed)
- Summary bar below photo: "4 items detected ~ ~650 cal ~ 45g protein" -- quick-glance aggregate
- Simple spinner with "Detecting foods..." text during inference -- no animated scanning effect, no inference timing shown
- Results persist on screen until user taps "Log Meal" or dismisses -- no auto-timeout
- "Log Meal" is a floating action button (FAB) at bottom-right with item count badge
- X button on each bounding box label chip for dismissal
- Swipe-to-dismiss also works as power-user shortcut (both patterns available)
- Removal must be undoable -- show undo toast after dismissal, item restored if tapped within timeout
- Manual "add missed item" deferred to Phase 3 (UI-02 manual food search)
- Threshold split: green >=80%, yellow 50-79%, red <50%
- All detected items auto-included in the meal regardless of confidence -- red items just have a visual flag encouraging review, not excluded
- Correction: tap bounding box -> bottom sheet -> tap food name -> search/replace from nutrition DB (Phase 1 bundled USDA + regional DBs)
- Local correction history stored in SQLite -- over time, suggest user's preferred label when similar detections recur (all on-device)
- Grams as primary unit, descriptive as secondary: "~150g (1 medium)"
- Portion adjustment via slider in the detail card: 0.5x to 3x of estimated portion, macros update in real-time
- When no reliable estimate available: use USDA standard serving size as fallback, show subtle "estimated" badge
- Reference object scaling: when standard dinner plate (~26cm) or common utensils are detected in the photo, use them to calibrate portion size
- All bounding boxes visible on photo simultaneously, with scrollable item list below
- Tapping either box or list item opens the same detail bottom sheet; tapping one highlights the other to show connection
- Items sorted by confidence (highest first) -- no grouping by tier
- Single photo per detection in this phase (multi-photo grouping is Phase 4)
- Meal type auto-detected from time of day: before 10am = breakfast, 10am-2pm = lunch, 2pm-5pm = snack, 5pm-9pm = dinner, 9pm+ = snack. User can change in summary bar.

### Claude's Discretion
- YOLO training hyperparameters and augmentation strategy
- CoreML/LiteRT export pipeline implementation details
- react-native-fast-tflite integration approach
- Inference router architecture (binary -> detect -> classify pipeline orchestration)
- Model pack format and metadata within the existing pack system
- Detail card layout and spacing
- Exact slider behavior and haptic feedback
- Undo toast duration and animation
- List item row design
- Cross-highlight animation between box and list

### Deferred Ideas (OUT OF SCOPE)
- Manual food search and add when detection misses items -- Phase 3 (UI-02)
- Multi-photo meal grouping (overhead + close-up merged detections) -- Phase 4
- Animated scanning effect during inference -- may revisit if simple spinner feels too basic
- Step-through review mode for individual items -- may revisit in Phase 5
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DET-01 | User can photograph food and get on-device identification of food items with bounding boxes via YOLO (CoreML/LiteRT) | YOLO training scripts exist, export pipeline documented, react-native-fast-tflite v2.0.0 provides unified runtime with CoreML delegate (iOS) and GPU delegate (Android), PackManager already supports model type packs |
| DET-05 | User sees confidence indicators (green/yellow/red) on detection results and can manually correct when confidence is low | YOLO output includes per-detection confidence scores; correction flow stores history in SQLite per locked decision; threshold splits defined |
| DET-06 | User sees portion estimates based on visual cues (plate size, reference objects, density tables) from the on-device portion estimator | PortionEstimator module already implemented with three-tier fallback (geometry/history/USDA default), density tables, and reference object support; needs mobile bridge |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| react-native-fast-tflite | 2.0.0 | Cross-platform TFLite inference with hardware delegates | Only maintained RN TFLite library with JSI/zero-copy performance; CoreML delegate for iOS, GPU for Android |
| ultralytics | >=8.3.0 | YOLO model training and export | Standard for YOLO training/export; supports YOLO26-N, CoreML and TFLite export formats |
| coremltools | >=7.0 | CoreML export validation (build-time) | Required by ultralytics for CoreML format export on macOS |
| @gorhom/bottom-sheet | 5.2 | Detail card for detected items | Already installed; locked decision for item detail display |
| react-native-reanimated | 4.2 | Animations (cross-highlight, slider, undo toast) | Already installed; powers bottom-sheet and gesture interactions |
| react-native-gesture-handler | 2.30 | Pinch-to-zoom, pan, swipe-to-dismiss | Already installed; locked decision for interactive photo |
| expo-image-picker | 17.0 | Camera/gallery photo capture | Already installed; entry point for detection flow |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| vision-camera-resize-plugin | latest | Image preprocessing before inference | If preprocessing needed to match model input dimensions |
| expo-haptics | latest | Haptic feedback on slider, undo toast | Detail card interactions per Claude's discretion |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| react-native-fast-tflite (TFLite on both platforms) | Native CoreML module for iOS + TFLite for Android | Two codepaths, two model formats, harder to maintain; marginal perf gain not worth complexity for photo-review (non-realtime) use case |
| react-native-fast-tflite | react-native-executorch | ExecuTorch is newer, less mature for YOLO; useObjectDetection hook exists but only for their pre-bundled models |
| react-native-fast-tflite | onnxruntime-react-native | ONNX runtime works but lacks the hardware delegate breadth of TFLite (CoreML, NNAPI, GPU) |

**Installation:**
```bash
cd apps/mobile && npm install react-native-fast-tflite
# Optional for haptics:
npx expo install expo-haptics
```

## Architecture Patterns

### Recommended Project Structure
```
apps/mobile/src/
├── services/
│   ├── detection/
│   │   ├── inferenceRouter.ts      # Orchestrates binary -> detect -> classify pipeline
│   │   ├── modelLoader.ts          # Loads models from PackManager paths via react-native-fast-tflite
│   │   ├── postProcess.ts          # NMS, bbox decoding, confidence thresholding
│   │   ├── portionBridge.ts        # Bridges Python PortionEstimator logic to TS
│   │   ├── correctionStore.ts      # SQLite correction history + suggestion engine
│   │   └── types.ts                # DetectionResult, BoundingBox, PortionEstimate types
│   └── packs/                      # Existing PackManager (already supports model type)
├── screens/
│   └── DetectionScreen.tsx         # Main detection results screen
├── components/
│   ├── detection/
│   │   ├── AnnotatedPhoto.tsx      # Photo with overlay bounding boxes, pinch-to-zoom
│   │   ├── BoundingBoxOverlay.tsx  # SVG/Canvas bounding box rendering with labels
│   │   ├── DetectionList.tsx       # Scrollable item list below photo
│   │   ├── DetectionListItem.tsx   # Single item row in list
│   │   ├── ItemDetailSheet.tsx     # @gorhom/bottom-sheet detail card
│   │   ├── PortionSlider.tsx       # 0.5x-3x portion adjustment slider
│   │   ├── SummaryBar.tsx          # "4 items ~ 650 cal ~ 45g protein" + meal type
│   │   ├── LogMealFAB.tsx          # Floating action button with badge
│   │   └── UndoToast.tsx           # Undo dismissal toast
│   └── ...
├── store/
│   └── useDetectionStore.ts        # Zustand store for detection session state
└── db/
    └── schema.ts                   # Add correction_history, detection_cache tables
```

### Pattern 1: Inference Router (Binary -> Detect -> Classify)
**What:** Three-stage pipeline that runs models sequentially, short-circuiting early when no food is detected.
**When to use:** Every detection invocation from the UI.
**Example:**
```typescript
// inferenceRouter.ts
interface InferenceResult {
  items: DetectedItem[];
  inferenceTimeMs: number;
  pipelineStages: { stage: string; timeMs: number }[];
}

async function runDetectionPipeline(
  imageUri: string,
  models: { binary: TensorflowModel; detect: TensorflowModel; classify: TensorflowModel }
): Promise<InferenceResult> {
  const start = performance.now();
  const stages: { stage: string; timeMs: number }[] = [];

  // Stage 1: Binary gate -- is this food?
  const stageStart1 = performance.now();
  const isFoodResult = await runBinaryGate(models.binary, imageUri);
  stages.push({ stage: 'binary', timeMs: performance.now() - stageStart1 });

  if (!isFoodResult.isFood) {
    return { items: [], inferenceTimeMs: performance.now() - start, pipelineStages: stages };
  }

  // Stage 2: Detection -- where is the food?
  const stageStart2 = performance.now();
  const detections = await runDetection(models.detect, imageUri);
  stages.push({ stage: 'detect', timeMs: performance.now() - stageStart2 });

  // Stage 3: Classification -- what food is it?
  const stageStart3 = performance.now();
  const items = await classifyDetections(models.classify, imageUri, detections);
  stages.push({ stage: 'classify', timeMs: performance.now() - stageStart3 });

  return { items, inferenceTimeMs: performance.now() - start, pipelineStages: stages };
}
```

### Pattern 2: Model Loading via PackManager
**What:** Load TFLite models from filesystem paths managed by PackManager.
**When to use:** App startup / detection screen mount.
**Example:**
```typescript
// modelLoader.ts
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { PackManager } from '../packs/packManager';

interface ModelSet {
  binary: TensorflowModel;
  detect: TensorflowModel;
  classify: TensorflowModel;
}

async function loadModelSet(): Promise<ModelSet> {
  const binaryPath = await PackManager.getPackFilePath('yolo-binary-v1');
  const detectPath = await PackManager.getPackFilePath('yolo-detect-v1');
  const classifyPath = await PackManager.getPackFilePath('yolo-classify-v1');

  if (!binaryPath || !detectPath || !classifyPath) {
    throw new Error('Model packs not installed. Download required.');
  }

  const [binary, detect, classify] = await Promise.all([
    loadTensorflowModel({ url: `file://${binaryPath}` }),
    loadTensorflowModel({ url: `file://${detectPath}` }),
    loadTensorflowModel({ url: `file://${classifyPath}` }),
  ]);

  return { binary, detect, classify };
}
```

### Pattern 3: YOLO Output Post-Processing in JavaScript
**What:** Decode raw YOLO output tensors into usable bounding boxes with confidence scores.
**When to use:** After every model.run() call for detection models.
**Example:**
```typescript
// postProcess.ts
interface RawDetection {
  x: number; y: number; w: number; h: number;
  confidence: number;
  classId: number;
  className: string;
}

/**
 * YOLO detection output shape: [1, 4+nc, numPredictions]
 * - First 4 rows: cx, cy, w, h (normalized 0-1)
 * - Remaining rows: per-class confidence scores
 *
 * Must transpose to [numPredictions, 4+nc] for processing.
 */
function decodeYoloOutput(
  output: Float32Array,
  numClasses: number,
  numPredictions: number,
  classNames: string[],
  confThreshold: number = 0.25,
): RawDetection[] {
  const detections: RawDetection[] = [];
  const stride = 4 + numClasses;

  for (let i = 0; i < numPredictions; i++) {
    // Transposed access: output[row * numPredictions + col]
    const cx = output[0 * numPredictions + i];
    const cy = output[1 * numPredictions + i];
    const w = output[2 * numPredictions + i];
    const h = output[3 * numPredictions + i];

    let maxConf = 0;
    let maxClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const conf = output[(4 + c) * numPredictions + i];
      if (conf > maxConf) {
        maxConf = conf;
        maxClassId = c;
      }
    }

    if (maxConf >= confThreshold) {
      detections.push({
        x: cx - w / 2, y: cy - h / 2, w, h,
        confidence: maxConf,
        classId: maxClassId,
        className: classNames[maxClassId] ?? `class_${maxClassId}`,
      });
    }
  }

  return nonMaxSuppression(detections, 0.45);
}

function nonMaxSuppression(
  detections: RawDetection[],
  iouThreshold: number,
): RawDetection[] {
  detections.sort((a, b) => b.confidence - a.confidence);
  const kept: RawDetection[] = [];

  for (const det of detections) {
    let dominated = false;
    for (const keptDet of kept) {
      if (iou(det, keptDet) > iouThreshold) {
        dominated = true;
        break;
      }
    }
    if (!dominated) kept.push(det);
  }

  return kept;
}

function iou(a: RawDetection, b: RawDetection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  return inter / (areaA + areaB - inter);
}
```

### Pattern 4: Zustand Detection Store (Write-First-Then-Refresh)
**What:** Manages detection session state following project convention.
**When to use:** Detection screen lifecycle.
**Example:**
```typescript
// useDetectionStore.ts
import { create } from 'zustand';

interface DetectedItem {
  id: string;
  className: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  portionEstimate: { weightG: number; confidence: string; method: string };
  portionMultiplier: number; // 0.5 - 3.0
  isRemoved: boolean;       // soft-delete for undo
  removedAt?: number;       // timestamp for undo timeout
}

interface DetectionState {
  photoUri: string | null;
  items: DetectedItem[];
  isDetecting: boolean;
  mealType: 'breakfast' | 'lunch' | 'snack' | 'dinner';
  // Actions
  setPhoto: (uri: string) => void;
  setItems: (items: DetectedItem[]) => void;
  removeItem: (id: string) => void;
  restoreItem: (id: string) => void;
  updatePortion: (id: string, multiplier: number) => void;
  correctItem: (id: string, newClassName: string) => void;
  setMealType: (type: DetectionState['mealType']) => void;
}
```

### Anti-Patterns to Avoid
- **Running all three models in parallel:** The pipeline is sequential by design. Binary gate saves compute when image is not food. Run sequentially: binary -> detect -> classify.
- **Shipping .mlpackage + .tflite separately:** Use TFLite for both platforms with CoreML delegate. One format, one codepath.
- **NMS in the model export:** Ultralytics `nms=True` export is incompatible with `end2end=True` for TFLite, and may cause issues on ARM64. Do NMS in JavaScript post-processing for full control.
- **Loading models on every detection:** Load models once on screen mount, cache in state, reuse across detections.
- **Blocking the JS thread during inference:** Use `model.run()` (async) not `model.runSync()` since this is photo-review, not live camera.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TFLite inference runtime | Native C++ bindings | react-native-fast-tflite v2.0.0 | JSI/zero-copy, hardware delegates, file:// loading, Expo config plugin |
| YOLO training pipeline | Custom PyTorch training loop | ultralytics YOLO API (model.train()) | Handles augmentation, scheduling, checkpointing, metrics, export |
| Model export to mobile formats | Manual ONNX -> TFLite conversion | ultralytics model.export(format='tflite'/'coreml') | Handles quantization, metadata, NMS options |
| Bottom sheet modal | Custom modal/slide-up | @gorhom/bottom-sheet 5.2 | Already installed, gesture-integrated, snap points |
| Portion estimation logic | JS reimplementation | Port PortionEstimator from Python | Complex density tables, reference object geometry, three-tier fallback already implemented |
| Image preprocessing/resize | Manual pixel manipulation | vision-camera-resize-plugin or canvas API | GPU-accelerated resize to model input dimensions |
| SHA-256 integrity checking | Custom hash implementation | PackManager.downloadPack() already handles | Existing pack system validates model downloads |

**Key insight:** The biggest trap in this phase is reimplementing the portion estimator from scratch in TypeScript. The Python PortionEstimator has 140+ entries in density tables, reference object geometry calculations, and a three-tier fallback chain. Port it faithfully; do not simplify.

## Common Pitfalls

### Pitfall 1: YOLO TFLite Output Tensor Transposition
**What goes wrong:** YOLO exports output as shape [1, 4+nc, 8400] but developers read it as [1, 8400, 4+nc], getting garbage bounding boxes.
**Why it happens:** The output is stored in row-major order but logically is [channels, predictions] not [predictions, channels].
**How to avoid:** Always transpose: access as `output[channel * numPredictions + predictionIndex]`. Verify with a known test image before building the full pipeline.
**Warning signs:** All detections cluster in top-left corner, or confidence scores are all near 0.

### Pitfall 2: CoreML Delegate Unsupported Ops
**What goes wrong:** Model loads on iOS but falls back to CPU silently, killing performance.
**Why it happens:** Some YOLO ops (custom NMS, certain activation functions) are not supported by the CoreML delegate. TFLite silently falls back to CPU for those ops.
**How to avoid:** Export WITHOUT `nms=True` (do NMS in JS). Test on a real iOS device and measure inference time -- if it's >500ms for YOLO-N, delegation likely failed. Check Xcode logs for delegate warnings.
**Warning signs:** Inference time on iOS is similar to or slower than Android mid-range devices.

### Pitfall 3: INT8 Quantization Accuracy Degradation
**What goes wrong:** INT8 quantized YOLO model produces significantly worse detections than FP32.
**Why it happens:** Bounding box regression and class prediction have different dynamic ranges; uniform INT8 quantization can clip important values. Research (YOLOv6+, 2025) documents this specific issue with TFLite INT8 conversion.
**How to avoid:** Start with FP16 (half=True) for a good size/accuracy tradeoff. Only attempt INT8 if FP16 model size is too large. If using INT8, provide representative calibration data via the `data` parameter during export.
**Warning signs:** mAP drops >5% from FP32 baseline after quantization.

### Pitfall 4: Model File Not Found at Runtime
**What goes wrong:** `loadTensorflowModel({ url: 'file://...' })` throws because the path from PackManager doesn't exist or has wrong permissions.
**Why it happens:** PackManager stores paths using expo-file-system URIs which may include `file://` prefix inconsistently. Path format differs between iOS and Android.
**How to avoid:** Always use `file://` prefix with the path from PackManager. Verify file existence before loading. Add a model health check on app startup.
**Warning signs:** Error "Failed to create TFLite interpreter from model" at runtime.

### Pitfall 5: Blocking the UI During Three-Stage Inference
**What goes wrong:** UI freezes for 200-500ms during inference, especially on mid-range Android.
**Why it happens:** Even async model.run() has synchronous preprocessing work (image resize, buffer creation) that runs on the JS thread.
**How to avoid:** Show the spinner immediately (locked decision). Offload image preprocessing to a separate worklet or use InteractionManager.runAfterInteractions. Profile on a real mid-range Android device (not emulator).
**Warning signs:** Dropped frames during detection, spinner appears late.

### Pitfall 6: Portion Estimator Port Accuracy Drift
**What goes wrong:** TypeScript port of PortionEstimator gives different results than Python version for the same inputs.
**Why it happens:** Floating point differences, different Math library implementations, or missed edge cases in the density table lookup.
**How to avoid:** Create a test suite with known input/output pairs from the Python version. Run the same test cases against both implementations. Include edge cases: flat foods, unknown dishes, very small/large bounding boxes.
**Warning signs:** Portion estimates differ by >10% from Python baseline for the same inputs.

## Code Examples

### YOLO Model Export Script
```python
# training/export_mobile.py
# Source: Ultralytics docs (https://docs.ultralytics.com/modes/export/)
from ultralytics import YOLO

def export_detection_model(weights_path: str, imgsz: int = 640):
    """Export YOLO detection model to TFLite for mobile deployment."""
    model = YOLO(weights_path)

    # FP16 TFLite -- good size/accuracy tradeoff for mobile
    model.export(
        format='tflite',
        imgsz=imgsz,
        half=True,       # FP16 quantization
        nms=False,        # Do NMS in JS for portability
    )

def export_classification_model(weights_path: str, imgsz: int = 224):
    """Export YOLO classification model to TFLite."""
    model = YOLO(weights_path)
    model.export(
        format='tflite',
        imgsz=imgsz,
        half=True,
    )
```

### Expo Config for react-native-fast-tflite
```json
// app.json plugins addition
{
  "plugins": [
    "expo-localization",
    ["react-native-fast-tflite", {
      "enableCoreMLDelegate": true,
      "enableAndroidGpuLibraries": true
    }]
  ]
}
```

### Metro Config Update
```javascript
// metro.config.js
const { getDefaultConfig } = require('expo/metro-config');
const config = getDefaultConfig(__dirname);
config.resolver.sourceExts.push('sql');
config.resolver.assetExts.push('tflite');  // Add TFLite model support
module.exports = config;
```

### Loading Model with CoreML Delegate (iOS)
```typescript
// Source: react-native-fast-tflite README
import { loadTensorflowModel } from 'react-native-fast-tflite';

// CoreML delegate is auto-enabled on iOS when Expo config plugin sets enableCoreMLDelegate
// No code change needed -- the library auto-selects best available delegate
const model = await loadTensorflowModel({
  url: `file://${modelFilePath}`,
});

// Run inference
const inputBuffer = preprocessImage(imageUri, 640, 640); // resize + normalize
const output = await model.run([inputBuffer]);
// output is Float32Array[] -- one array per output tensor
```

### Confidence Color Mapping
```typescript
// Locked decision: green >=80%, yellow 50-79%, red <50%
type ConfidenceLevel = 'high' | 'medium' | 'low';

function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.80) return 'high';
  if (confidence >= 0.50) return 'medium';
  return 'low';
}

const CONFIDENCE_COLORS: Record<ConfidenceLevel, string> = {
  high: '#22C55E',   // green-500
  medium: '#EAB308', // yellow-500
  low: '#EF4444',    // red-500
};
```

### Meal Type Auto-Detection
```typescript
// Locked decision: time-of-day based meal type
function autoDetectMealType(): 'breakfast' | 'lunch' | 'snack' | 'dinner' {
  const hour = new Date().getHours();
  if (hour < 10) return 'breakfast';
  if (hour < 14) return 'lunch';
  if (hour < 17) return 'snack';
  if (hour < 21) return 'dinner';
  return 'snack';
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TFLite + NNAPI | LiteRT with vendor NPU delegates | Android 15 (2024) | NNAPI deprecated; LiteRT provides Qualcomm QNN, MediaTek NeuroPilot delegates |
| Separate CoreML + TFLite codepaths | react-native-fast-tflite with CoreML delegate | 2025 | Single TFLite model works on both platforms; CoreML delegation is transparent |
| YOLOv8n | YOLO26-N | Jan 2026 | Improved backbone, better mobile performance; same export workflow |
| react-native-fast-tflite v1.x | v2.0.0 | Jan 2026 | Breaking release; updated API |
| Manual model bundling in app binary | PackManager model packs from R2 | Phase 1 design | Models delivered via same pack system as nutrition DBs; no base APK bloat |

**Deprecated/outdated:**
- **NNAPI:** Deprecated in Android 15. Use LiteRT GPU or vendor NPU delegates instead.
- **react-native-fast-tflite v1.x:** v2.0.0 released Jan 2026 with potential breaking changes.
- **Separate .mlpackage for iOS:** Not needed; react-native-fast-tflite CoreML delegate handles iOS hardware acceleration from .tflite files.

## Open Questions

1. **YOLO26-N availability in Ultralytics**
   - What we know: Training scripts reference `yolo26n.pt` with fallback chain to `yolo11n.pt` and `yolov8n.pt`. YOLO26 was released Jan 14, 2026.
   - What's unclear: Whether YOLO26-N specific checkpoints are available for classification tasks (`yolo26n-cls.pt`). The scripts have fallback chains which suggest uncertainty.
   - Recommendation: Use the fallback chain as-is. Start training with whatever loads; the export pipeline is format-agnostic.

2. **react-native-fast-tflite v2.0.0 breaking changes**
   - What we know: v2.0.0 released Jan 13, 2026. The API surface (loadTensorflowModel, run, runSync) appears stable.
   - What's unclear: Exact breaking changes from v1.x. The npm page doesn't list a migration guide.
   - Recommendation: Install v2.0.0 fresh (not upgrading from v1.x). Test with a sample model before integrating full pipeline.

3. **Image preprocessing for react-native-fast-tflite**
   - What we know: Models expect fixed-size input buffers (e.g., 640x640x3 float32 or uint8). react-native-fast-tflite works with TypedArrays.
   - What's unclear: Best approach to resize a camera photo to model input dimensions in React Native. Options: vision-camera-resize-plugin, expo-image-manipulator, custom native module, or canvas-based resize.
   - Recommendation: Use expo-image-manipulator for photo resize (already in Expo ecosystem, handles rotation/orientation). Convert resized JPEG to raw pixel buffer for model input.

4. **Three-model pipeline latency budget**
   - What we know: YOLO-N inference is ~50-80ms per stage on CPU, ~5-8ms with NPU. Three stages = 150-240ms CPU, 15-24ms NPU. Target is <2 seconds on mid-range devices.
   - What's unclear: Image preprocessing overhead (resize, buffer conversion) and JS-native bridge overhead.
   - Recommendation: Budget: 200ms preprocessing + 300ms inference (3 stages) + 100ms post-processing + 200ms UI render = ~800ms. Well within 2s target. Profile on actual mid-range device early.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | jest-expo (jest + React Native) |
| Config file | apps/mobile/jest.config.js |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern='detection' --no-coverage` |
| Full suite command | `cd apps/mobile && npx jest --no-coverage` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DET-01 | YOLO output decoding produces valid bounding boxes | unit | `cd apps/mobile && npx jest --testPathPattern='postProcess' -x` | Wave 0 |
| DET-01 | Inference router runs three-stage pipeline | unit | `cd apps/mobile && npx jest --testPathPattern='inferenceRouter' -x` | Wave 0 |
| DET-01 | Model loader resolves pack paths and loads models | unit | `cd apps/mobile && npx jest --testPathPattern='modelLoader' -x` | Wave 0 |
| DET-05 | Confidence color mapping matches thresholds | unit | `cd apps/mobile && npx jest --testPathPattern='confidence' -x` | Wave 0 |
| DET-05 | Correction store persists and retrieves correction history | unit | `cd apps/mobile && npx jest --testPathPattern='correctionStore' -x` | Wave 0 |
| DET-06 | Portion estimator (TS port) matches Python reference outputs | unit | `cd apps/mobile && npx jest --testPathPattern='portionBridge' -x` | Wave 0 |
| DET-06 | Portion slider updates macros in real-time | unit | `cd apps/mobile && npx jest --testPathPattern='PortionSlider' -x` | Wave 0 |
| DET-01 | YOLO training scripts complete without error | integration | `cd training && python train_detect.py --prepare-data --epochs 1 --device cpu` | Existing |
| DET-01 | YOLO export produces valid .tflite file | integration | `python -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='tflite', imgsz=640)"` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --testPathPattern='detection' --no-coverage`
- **Per wave merge:** `cd apps/mobile && npx jest --no-coverage`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `apps/mobile/src/services/detection/__tests__/postProcess.test.ts` -- covers DET-01 YOLO output decoding + NMS
- [ ] `apps/mobile/src/services/detection/__tests__/inferenceRouter.test.ts` -- covers DET-01 pipeline orchestration
- [ ] `apps/mobile/src/services/detection/__tests__/modelLoader.test.ts` -- covers DET-01 model loading
- [ ] `apps/mobile/src/services/detection/__tests__/portionBridge.test.ts` -- covers DET-06 TS port validation
- [ ] `apps/mobile/src/services/detection/__tests__/correctionStore.test.ts` -- covers DET-05 correction history
- [ ] `training/export_mobile.py` -- export script for TFLite models
- [ ] `apps/mobile/__mocks__/react-native-fast-tflite.ts` -- mock for jest testing

## Sources

### Primary (HIGH confidence)
- [Ultralytics YOLO export docs](https://docs.ultralytics.com/modes/export/) - Export format table, API parameters, CoreML and TFLite specifics
- [Ultralytics CoreML integration](https://docs.ultralytics.com/integrations/coreml/) - CoreML export command, macOS requirement, FP16/INT8 options
- [Ultralytics TFLite integration](https://docs.ultralytics.com/integrations/tflite/) - TFLite export command, quantization options, deployment targets
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite) - v2.0.0 API, CoreML delegate, GPU delegate, Expo config plugin
- [react-native-fast-tflite npm](https://www.npmjs.com/package/react-native-fast-tflite) - v2.0.0 release, latest version confirmation
- Existing codebase: training/train_detect.py, train_binary.py, train_classify.py, portion_estimator.py, packManager.ts

### Secondary (MEDIUM confidence)
- [Ultralytics YOLO26 release](https://docs.ultralytics.com/models/yolo26/) - YOLO26 architecture improvements, Jan 2026 release
- [Julius Hietala CoreML Expo guide](https://hietalajulius.medium.com/building-a-react-native-coreml-image-classification-app-with-expo-and-yolov8-a083c7866e85) - Expo module + CoreML pattern
- ADR-003 (YOLO decision) and ADR-005 (local-first architecture) - Project architecture constraints

### Tertiary (LOW confidence)
- [GitHub Issue #8361](https://github.com/ultralytics/ultralytics/issues/8361) - YOLO TFLite output handling in React Native (community experience)
- [GitHub Issue #96](https://github.com/mrousavy/react-native-fast-tflite/issues/96) - YOLO latency concerns (developer report)
- [YOLOv6+ INT8 paper](https://link.springer.com/article/10.1007/s11760-025-04234-0) - INT8 quantization accuracy degradation research

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - react-native-fast-tflite v2.0.0 confirmed on npm, Ultralytics export API verified via official docs, all existing libraries already installed in project
- Architecture: HIGH - Three-stage pipeline matches existing training scripts (train_binary.py, train_detect.py, train_classify.py), PackManager already supports model type packs, project conventions (Zustand, op-sqlite, soft-delete) well-documented
- Pitfalls: HIGH - YOLO TFLite output format issues documented in multiple GitHub issues, CoreML delegate limitations confirmed in official docs, INT8 degradation validated by peer-reviewed research
- Portion estimator port: MEDIUM - Python implementation is comprehensive and well-tested, but TS port accuracy needs validation against reference outputs

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable domain -- YOLO export and react-native-fast-tflite APIs unlikely to change within 30 days)
