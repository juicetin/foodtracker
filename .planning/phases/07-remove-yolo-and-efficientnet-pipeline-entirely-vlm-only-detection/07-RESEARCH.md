# Phase 7: Remove YOLO and EfficientNet pipeline entirely -- VLM-only detection - Research

**Researched:** 2026-03-15
**Domain:** ML pipeline refactoring, code removal, UX shimmer animation, VLM failure fallback
**Confidence:** HIGH

## Summary

Phase 7 is primarily a removal and simplification phase. The EfficientNet classification stage (4.9MB model, 905-class labels, ImageNet preprocessing, all training scripts) is fully deleted. YOLO is retained but stripped to bounding-box-only duty -- all class label output is ignored, every detection is treated as a generic "food region." The VLM becomes the sole source of food identification. The inference router is rewritten from a two-stage detect+classify pipeline to a single-stage bbox-only pipeline. The UI transitions from the "Refining..." badge to shimmer/skeleton animations inside bounding boxes and detection list items while VLM is processing. A VLM failure fallback provides user text input with KG-powered dish lookup.

The codebase is in excellent shape for this phase. All classification-related code is cleanly separated: `classify.tflite` and `labels_classify.json` as bundled assets, `CLASSIFY_CLASS_NAMES`/`CLASSIFY_INPUT_SIZE`/`IMAGENET_MEAN`/`IMAGENET_STD` constants, `classifyBuffer` preprocessing in DetectionScreen, the classify stage in `inferenceRouter.ts`, and the `classify` field in `ModelSet`. The training scripts (`train_classify.py`, `export_mobile.py`, `train_binary.py`) and evaluation script (`eval_classification.py`) are self-contained. Removal is surgical and well-scoped.

**Primary recommendation:** Execute in three waves: (1) EfficientNet removal + inferenceRouter simplification + modelLoader simplification, (2) shimmer/skeleton UX replacing RefiningBadge, (3) VLM failure fallback with text prompt and KG bridge.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Keep current GGCD YOLOv8n (241 food classes, 3.6MB) for bounding box spatial detection only
- Ignore all class label output from YOLO -- treat every detection as "food region"
- Do NOT swap to a different/smaller model -- current model works and is already tested
- YOLO training scripts (train_detect.py, export_ggcd_detect.py) are kept in the repo
- Delete classify.tflite (4.9MB) and labels_classify.json from bundled assets
- Remove all EfficientNet inference code, ImageNet normalization constants, classification preprocessing
- Remove Python training scripts: train_classify.py, export_mobile.py, eval_classification.py
- Remove dataset merge/acquisition scripts related to classifier training
- Remove all classification-related tests
- Clean break -- git history preserves everything if ever needed
- YOLO boxes appear instantly (50-80ms) with shimmer/skeleton animation inside each box where the label will appear
- Detection list below photo also shows shimmer placeholder items (one per YOLO box) -- user sees item count immediately
- When VLM results arrive, shimmer fades out and label fades in (animated transition, consistent with existing Phase 2.6 refining animation)
- On VLM failure: silent retry once behind the shimmer (user doesn't see the retry)
- If retry also fails: show "Describe your meal" text prompt
- Text prompt tells user that suggested ingredients will be populated from KG
- Map user-typed dishes to YOLO boxes by count (1:1 assignment by box size order -- largest box = first dish)
- KG fuzzy search matches dish names -> recipe decomposition -> nutrition
- The existing "Refining..." badge from Phase 2.6 should be replaced by the shimmer pattern
- When VLM fails and user types dish names, KG should surface suggested ingredients (not just accept the dish name silently)

### Claude's Discretion
- Exact shimmer animation implementation (library choice, timing)
- How to restructure inferenceRouter (YOLO-boxes-only stage + VLM identification stage)
- VLM retry timeout duration and backoff strategy
- How to handle count mismatch (more/fewer dishes typed than boxes detected)
- Whether to simplify model_manifest.json or keep it for YOLO entry only

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| react-native-reanimated | ~4.1.1 | Shimmer animations, fade transitions | Already installed, drives RefiningBadge, DetectionListItem swipe |
| react-native-fast-tflite | (existing) | YOLO TFLite model loading | Already used for YOLO detect, stays for bbox detection |
| llama.rn | 0.11.4 | VLM inference via SmolVLM GGUF | Already integrated in Phase 2.6, primary food identifier |
| zustand | (existing) | Ephemeral detection state | Already used for useDetectionStore |

### No New Dependencies Required
The shimmer animation can be built with pure `react-native-reanimated` (already installed) using `withRepeat(withTiming(...))` for opacity pulsing and `FadeIn`/`FadeOut` for transitions. No additional shimmer/skeleton library is needed.

**Rationale:** The existing `RefiningBadge.tsx` already demonstrates the pattern -- `useSharedValue` + `withRepeat(withTiming(...))` for pulsing animation. The shimmer for bounding boxes and list items follows the exact same pattern with minor visual differences (background color pulse vs. opacity pulse).

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom Reanimated shimmer | react-native-reanimated-skeleton | Extra dependency for ~30 lines of animation code. Not worth it. |
| Custom Reanimated shimmer | moti + @motify/skeleton | Heavy dependency chain (moti requires separate setup). Overkill. |
| Opacity pulse shimmer | LinearGradient sweep shimmer | Requires expo-linear-gradient or react-native-linear-gradient. Prettier but adds native dependency for minor UX gain. |

**Decision: Use custom Reanimated shimmer.** The pulsing opacity pattern (same as RefiningBadge) is lightweight, consistent with existing code, and requires zero new dependencies.

## Architecture Patterns

### Recommended Project Structure (Changes Only)
```
apps/mobile/src/
  services/detection/
    inferenceRouter.ts        # REWRITE: bbox-only pipeline, remove classify stage
    modelLoader.ts            # SIMPLIFY: detect-only, remove classify model loading
    constants.ts              # SIMPLIFY: remove CLASSIFY_*, IMAGENET_* exports
    imagePreprocess.ts        # SIMPLIFY: remove 'imagenet' normalization mode
    types.ts                  # SIMPLIFY: ModelSet -> detect only, PipelineStage drop 'classify'
    postProcess.ts            # KEEP AS-IS (already bbox-focused)
    portionBridge.ts          # KEEP AS-IS (not tied to food labels)
  services/vlm/
    vlmPipeline.ts            # REWRITE: from "refinement" to "primary identification"
    vlmService.ts             # KEEP AS-IS
    vlmPrompts.ts             # KEEP AS-IS
    vlmTypes.ts               # KEEP AS-IS
  components/detection/
    BoundingBoxOverlay.tsx     # MODIFY: shimmer inside boxes until VLM label arrives
    RefiningBadge.tsx          # DELETE: replaced by shimmer pattern
    DetectionList.tsx          # KEEP AS-IS (already uses displayLabel)
    DetectionListItem.tsx      # MODIFY: shimmer placeholder when isRefining
    ShimmerPlaceholder.tsx     # NEW: reusable shimmer component
    index.ts                  # UPDATE: remove RefiningBadge export, add ShimmerPlaceholder
  screens/
    DetectionScreen.tsx        # REWRITE: new flow (bbox -> shimmer -> VLM -> results)
  store/
    useDetectionStore.ts       # MODIFY: displayLabel and flow state adjustments
assets/models/
    classify.tflite            # DELETE (4.9MB saved)
    labels_classify.json       # DELETE (16KB saved)
    model_manifest.json        # SIMPLIFY: remove efficientnet-classify-v2 entry
    detect.tflite              # KEEP
    labels_detect.json         # KEEP (used for YOLO output tensor decoding only)
training/
    train_classify.py          # DELETE
    export_mobile.py           # DELETE
    train_binary.py            # DELETE
    evaluate/eval_classification.py  # DELETE
    datasets/scripts/merge_datasets.py  # DELETE (classifier-specific)
```

### Pattern 1: Bbox-Only Inference Router
**What:** Strip the inferenceRouter to a single-stage pipeline that runs YOLO and returns generic "food region" items with bounding boxes only.
**When to use:** Every detection flow entry point.
**Example:**
```typescript
// inferenceRouter.ts -- simplified
export async function runBboxDetection(
  detectBuffer: Float32Array,
  imageWidth: number,
  imageHeight: number,
  classNames: string[],  // Still needed for YOLO tensor decoding
): Promise<InferenceResult> {
  const models = getModelSet();
  if (!models) throw new Error('Model not loaded.');

  const detectStart = performance.now();
  const detectOutput = await models.detect.run([detectBuffer]);
  const detectTimeMs = performance.now() - detectStart;

  const detectTensor = detectOutput[0] instanceof Float32Array
    ? detectOutput[0]
    : new Float32Array(detectOutput[0] as ArrayBuffer);

  const numClasses = classNames.length;
  const stride = 4 + numClasses;
  const numPredictions = stride > 0 ? Math.floor(detectTensor.length / stride) : 0;

  const rawDetections = decodeYoloOutput(detectTensor, numClasses, numPredictions, classNames);

  // Every detection is "Food Region" -- YOLO labels ignored
  const items: DetectedItem[] = rawDetections.map((det) => ({
    id: generateDetectionId(),
    className: 'Food Region',  // Generic, not YOLO label
    confidence: det.confidence,
    bbox: { x: det.x, y: det.y, w: det.w, h: det.h },
    portionEstimate: defaultPortionEstimate(),
    portionMultiplier: 1,
    isRemoved: false,
    isRefining: true,  // Start in shimmer state
  }));

  return { items, inferenceTimeMs: performance.now() - detectStart, pipelineStages: [{ stage: 'detect', timeMs: detectTimeMs }] };
}
```

### Pattern 2: Shimmer Component with Reanimated
**What:** Reusable shimmer placeholder using `useSharedValue` + `withRepeat(withTiming(...))`.
**When to use:** Inside bounding box labels and detection list items while `isRefining === true`.
**Example:**
```typescript
// ShimmerPlaceholder.tsx
import React, { useEffect } from 'react';
import { StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  FadeIn,
  FadeOut,
} from 'react-native-reanimated';

interface ShimmerPlaceholderProps {
  width: number | string;
  height: number;
  style?: ViewStyle;
}

export function ShimmerPlaceholder({ width, height, style }: ShimmerPlaceholderProps) {
  const opacity = useSharedValue(1);

  useEffect(() => {
    opacity.value = withRepeat(
      withTiming(0.4, { duration: 800 }),
      -1,
      true,
    );
  }, [opacity]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  return (
    <Animated.View
      entering={FadeIn.duration(200)}
      exiting={FadeOut.duration(200)}
      style={[
        { width, height, borderRadius: 6, backgroundColor: '#E0E0E0' },
        animatedStyle,
        style,
      ]}
    />
  );
}
```

### Pattern 3: VLM Failure Fallback with Retry
**What:** Silent retry once on VLM failure, then fall back to text prompt with KG suggestions.
**When to use:** VLM identify() call fails.
**Example:**
```typescript
// In DetectionScreen or vlmPipeline
async function identifyWithRetry(
  photoUri: string,
  userText?: string,
  retryTimeoutMs: number = 30000,
): Promise<VlmFoodResult> {
  try {
    return await vlmService.identify(photoUri, userText);
  } catch (firstError) {
    // Silent retry once (user sees shimmer, not error)
    try {
      return await vlmService.identify(photoUri, userText);
    } catch (retryError) {
      // Both attempts failed -- return empty to trigger text fallback
      return { dishes: [] };
    }
  }
}
```

### Pattern 4: Text Fallback with Box-Size Assignment
**What:** When VLM fails, user types dish names. Each dish maps to a YOLO box by size order.
**When to use:** VLM returns empty dishes or both retry attempts fail.
**Example:**
```typescript
// Map user-typed dish names to YOLO boxes by area (largest first)
function assignDishesToBoxes(
  items: DetectedItem[],
  dishNames: string[],
): Map<string, string> {
  const sorted = [...items]
    .filter(i => !i.isRemoved)
    .sort((a, b) => (b.bbox.w * b.bbox.h) - (a.bbox.w * a.bbox.h));

  const assignments = new Map<string, string>();
  for (let i = 0; i < Math.min(sorted.length, dishNames.length); i++) {
    assignments.set(sorted[i].id, dishNames[i].trim());
  }
  return assignments;
}
```

### Anti-Patterns to Avoid
- **Keeping YOLO labels anywhere in the UI:** The whole point is VLM-only identification. Do not display `className` from YOLO (except possibly in `__DEV__` logs for debugging).
- **Running EfficientNet "just in case":** The model is removed. No fallback to classification.
- **Blocking UI on VLM failure:** VLM retry must be silent behind shimmer. Never show error dialogs for VLM timeouts.
- **Hardcoding "Food Region" in display:** The `displayLabel` function in the store should show shimmer (via `isRefining`) or the VLM label, never the raw className.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Shimmer animation | Custom requestAnimationFrame loop | Reanimated `withRepeat(withTiming())` | Runs on UI thread, auto-cleanup, same pattern as existing RefiningBadge |
| Animated transitions | Manual opacity state + setTimeout | Reanimated `FadeIn`/`FadeOut` entering/exiting | Declarative, hardware-accelerated, already used in RefiningBadge |
| KG fuzzy search for text input | New fuzzy matching | Existing `KnowledgeGraphService.searchDish()` | Already has SymSpell + FTS5 + alias chain, tested in Phase 2.5 |
| VLM-to-box matching | New matching algorithm | Existing `matchVlmToYolo()` in vlmPipeline.ts | Already handles substring, word overlap, and positional fallback |

**Key insight:** Almost all building blocks exist. This phase is 70% removal and 30% rewiring/adding shimmer UX.

## Common Pitfalls

### Pitfall 1: YOLO Class Names Still Needed for Tensor Decoding
**What goes wrong:** Removing `DETECT_CLASS_NAMES` and `labels_detect.json` because "we don't use YOLO labels anymore."
**Why it happens:** Confusion between "ignore YOLO labels in UI" and "don't need class names at all."
**How to avoid:** `DETECT_CLASS_NAMES` is still required for `decodeYoloOutput()` to properly parse the YOLO output tensor shape (stride = 4 + numClasses = 245). The class names determine how many class probability columns to read. Without them, tensor decoding breaks entirely.
**Warning signs:** Tests fail with "numClasses is 0" or NaN bounding boxes.

### Pitfall 2: Stale Installed Packs After Classify Model Removal
**What goes wrong:** Users who had the classify model installed (via PackManager) see errors because `modelLoader.ts` still finds an `efficientnet-classify-*` entry in `installed_packs`.
**Why it happens:** Database migration removes code but not data.
**How to avoid:** Add a migration or startup cleanup that removes `efficientnet-classify-*` and `yolo-classify-*` entries from `installed_packs`. Alternatively, since the classify model is bundled (not downloaded), just removing the bundled file and the code that references it is sufficient. The `installed_packs` table only tracks downloaded packs, not bundled ones.
**Warning signs:** "Model pack not installed" errors on clean install.

### Pitfall 3: classifyBuffer Preprocessing Left Behind
**What goes wrong:** `imagePreprocess.ts` still supports the `'imagenet'` normalization mode and `IMAGENET_MEAN`/`IMAGENET_STD` constants after EfficientNet is removed. Dead code sits in the codebase.
**Why it happens:** Incomplete cleanup -- removing the caller but not the callee.
**How to avoid:** Remove the `'imagenet'` normalization path from `preprocessImageForModel()`, the `IMAGENET_MEAN`/`IMAGENET_STD` exports from `constants.ts`, and the `CLASSIFY_INPUT_SIZE` constant. The function signature simplifies to `preprocessImageForModel(uri, size)` without the normalization parameter.
**Warning signs:** Dead code detection tools flag unused exports.

### Pitfall 4: VLM Pipeline Still Expects YOLO Labels for Matching
**What goes wrong:** `matchVlmToYolo()` in `vlmPipeline.ts` compares VLM dish names against `item.className`. After this phase, `className` is "Food Region" for all items -- so substring/word-overlap matching always fails, and everything falls through to positional (area-based) assignment.
**Why it happens:** The matching algorithm was designed for meaningful YOLO labels.
**How to avoid:** This is actually fine! Positional fallback (largest bbox = first VLM dish) is the intended behavior per CONTEXT.md. The matching function should be simplified to skip the substring/word-overlap strategies and go straight to positional assignment. Or remove the matching function entirely and use purely positional assignment.
**Warning signs:** None (it would work via fallback), but the extra matching code is dead weight.

### Pitfall 5: displayLabel Shows "Food Region" During Shimmer
**What goes wrong:** While VLM is processing, `displayLabel(item)` returns `item.vlmLabel ?? 'Identifying...'`. But after this phase, items start with `isRefining: true`. If the shimmer component only renders when `isRefining` is true but `displayLabel` is still called for the label chip text, users may briefly see "Identifying..." before shimmer renders.
**Why it happens:** Race between React render and Reanimated animation start.
**How to avoid:** The shimmer component should fully replace the text content (not overlay it). When `isRefining === true`, render `<ShimmerPlaceholder>` instead of `<Text>{displayLabel(item)}</Text>`.
**Warning signs:** Flash of "Identifying..." text before shimmer appears.

### Pitfall 6: APK Size Regression from Forgetting Build Artifacts
**What goes wrong:** `classify.tflite` is deleted from source but the Android build's `intermediates/` directory retains the old copy.
**Why it happens:** Gradle incremental builds cache previous assets.
**How to avoid:** Run `cd apps/mobile/android && ./gradlew clean` after removing bundled assets. Verify with `find android/app/build -name "classify*"` after rebuild.
**Warning signs:** APK is still the same size as before despite removing 4.9MB model.

## Code Examples

### Current Flow (Before Phase 7)
```
Photo -> preprocessImageForModel(uri, 640)        -> detectBuffer
      -> preprocessImageForModel(uri, 224, 'imagenet') -> classifyBuffer
      -> runDetectionPipeline(detectBuffer, classifyBuffer, 640, 640, DETECT_CLASS_NAMES)
          Stage 1: YOLO detect -> raw boxes with class names
          Stage 2: EfficientNet classify -> secondary label (logged, not used)
      -> enriched items with YOLO className as primary label
      -> VLM refinement (async): vlmLabel replaces className in display
```

### Target Flow (After Phase 7)
```
Photo -> preprocessImageForModel(uri, 640)  -> detectBuffer
      -> runBboxDetection(detectBuffer, 640, 640, DETECT_CLASS_NAMES)
          Stage 1 only: YOLO detect -> bbox coordinates, no meaningful labels
      -> items with className='Food Region', isRefining=true
      -> UI: shimmer in bbox labels + shimmer in detection list
      -> VLM identification (primary, not "refinement"):
          Success: vlmLabel populates, shimmer fades out, label fades in
          Failure: retry once silently
          Failure x2: show "Describe your meal" text input
              -> user types dish names
              -> KG fuzzy search -> suggested ingredients shown
              -> dishes assigned to boxes by area order
```

### Files to Delete (Complete List)
```
# Bundled assets
apps/mobile/assets/models/classify.tflite           (4.9MB)
apps/mobile/assets/models/labels_classify.json       (16KB)

# Python training scripts
training/train_classify.py
training/export_mobile.py
training/train_binary.py
training/evaluate/eval_classification.py

# Dataset scripts (classifier-specific)
training/datasets/scripts/merge_datasets.py
training/datasets/scripts/auto_label.py
training/datasets/scripts/download_datasets.py
training/configs/food-classify.yaml
```

### Files to Modify (Complete List)
```
# Core pipeline simplification
apps/mobile/src/services/detection/inferenceRouter.ts     # Remove classify stage, rename function
apps/mobile/src/services/detection/modelLoader.ts         # Remove classify model loading
apps/mobile/src/services/detection/constants.ts           # Remove CLASSIFY_*, IMAGENET_* exports
apps/mobile/src/services/detection/imagePreprocess.ts     # Remove 'imagenet' normalization mode
apps/mobile/src/services/detection/types.ts               # Simplify ModelSet, PipelineStage
apps/mobile/assets/models/model_manifest.json             # Remove efficientnet-classify-v2 entry

# VLM pipeline rewrite
apps/mobile/src/services/vlm/vlmPipeline.ts              # From "refinement" to "primary identification"

# UI changes
apps/mobile/src/screens/DetectionScreen.tsx               # New flow, remove classify preprocessing
apps/mobile/src/components/detection/BoundingBoxOverlay.tsx  # Shimmer in box labels
apps/mobile/src/components/detection/DetectionListItem.tsx   # Shimmer placeholder row
apps/mobile/src/components/detection/index.ts             # Remove RefiningBadge, add ShimmerPlaceholder
apps/mobile/src/store/useDetectionStore.ts                # Adjust displayLabel for new flow

# Tests to rewrite
apps/mobile/src/services/detection/__tests__/inferenceRouter.test.ts   # Single-stage tests
apps/mobile/src/services/detection/__tests__/modelLoader.test.ts       # Detect-only loading
apps/mobile/src/services/detection/__tests__/modelBootstrap.test.ts    # Detect-only bundled
apps/mobile/src/services/detection/__tests__/imagePreprocess.test.ts   # Remove imagenet mode tests
apps/mobile/src/services/vlm/__tests__/vlmPipeline.test.ts            # Update for primary identification
```

### Files to Create
```
apps/mobile/src/components/detection/ShimmerPlaceholder.tsx   # Reusable shimmer component
```

### Files to Delete (Components)
```
apps/mobile/src/components/detection/RefiningBadge.tsx        # Replaced by ShimmerPlaceholder
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| YOLO detect + EfficientNet classify (two-stage) | YOLO bbox + VLM identify (two-stage, different) | Phase 7 (now) | 4.9MB APK size reduction, simpler pipeline |
| YOLO class labels as primary food name | VLM as sole food identifier | Phase 2.6 (previous) | Already effective, this phase formalizes it |
| "Refining..." badge for VLM processing | Shimmer/skeleton in bounding boxes and list | Phase 7 (now) | Better perceived performance -- user sees structure immediately |
| Hard failure on VLM unavailable | Text fallback with KG ingredient suggestions | Phase 7 (now) | Graceful degradation -- app always produces results |

**Deprecated/outdated:**
- EfficientNet-Lite0 classifier: Fully removed. VLM provides better food identification with contextual understanding.
- ImageNet normalization: No longer needed without EfficientNet. Preprocessing simplifies to zero-one normalization only.
- Binary food gate (train_binary.py): Already dead code since Phase 2.2. Formally removed now.
- `classifyBuffer` dual-preprocessing: Single buffer (640x640) for YOLO only.

## Open Questions

1. **Should `decodeYoloOutput()` still return `className` from YOLO labels?**
   - What we know: The function currently extracts `className` from the highest-confidence class column. This is needed for proper tensor parsing.
   - What's unclear: Whether to keep the `className` field populated with the YOLO label (ignored in UI) or replace it with a generic string.
   - Recommendation: Keep returning the YOLO className from tensor decoding (it's free and useful for `__DEV__` debugging), but set `className: 'Food Region'` on the DetectedItem after inference. This preserves debugging capability without leaking YOLO labels to UI.

2. **Should `portionBridge.ts` still receive `className` for density lookups?**
   - What we know: `estimatePortion()` uses `className` to look up food-specific density. After Phase 7, className is "Food Region" until VLM identifies the dish.
   - What's unclear: Whether to defer portion estimation until VLM label arrives, or use a generic fallback density.
   - Recommendation: Use generic fallback density initially, then re-estimate when VLM label arrives. This keeps the portion estimate visible immediately (shimmer for label, but grams shown).

3. **What about the `training/datasets/scripts/` cleanup scope?**
   - What we know: `merge_datasets.py`, `auto_label.py`, `download_datasets.py` are used for building the classifier training set. `audit_cuisines.py` references classify but may also be used for YOLO audit.
   - What's unclear: Whether `audit_cuisines.py` should stay (it's general-purpose) or go.
   - Recommendation: Keep `audit_cuisines.py` (it's useful for YOLO class auditing too). Delete the other three.

4. **Should `model_manifest.json` be simplified or kept as-is?**
   - What we know: Currently has two entries (detect + classify). After removal, only detect remains.
   - What's unclear: Whether the manifest format should change.
   - Recommendation: Keep the manifest with one entry. Remove the classify entry. The manifest structure is fine for a single model and will be useful if Scale OCR (Phase 5) adds another TFLite model later.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | jest-expo (Jest with Expo preset) |
| Config file | `apps/mobile/jest.config.js` |
| Quick run command | `cd apps/mobile && npx jest --testPathPattern="detection\|vlm\|store" --no-coverage` |
| Full suite command | `cd apps/mobile && npx jest --no-coverage` |

### Phase Requirements -> Test Map

Phase 7 does not have formal requirement IDs (TBD in REQUIREMENTS.md). Requirements are derived from CONTEXT.md decisions:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| P7-01 | inferenceRouter runs bbox-only (no classify stage) | unit | `cd apps/mobile && npx jest inferenceRouter.test.ts -x` | Rewrite needed |
| P7-02 | modelLoader loads detect-only (no classify model) | unit | `cd apps/mobile && npx jest modelLoader.test.ts -x` | Rewrite needed |
| P7-03 | ModelSet type has no classify field | unit | `cd apps/mobile && npx jest modelLoader.test.ts -x` | Rewrite needed |
| P7-04 | imagePreprocess has no imagenet normalization mode | unit | `cd apps/mobile && npx jest imagePreprocess.test.ts -x` | Rewrite needed |
| P7-05 | vlmPipeline is primary identification (not refinement) | unit | `cd apps/mobile && npx jest vlmPipeline.test.ts -x` | Rewrite needed |
| P7-06 | VLM retry on failure (once, silent) | unit | `cd apps/mobile && npx jest vlmPipeline.test.ts -x` | New test |
| P7-07 | Text fallback assigns dishes to boxes by area | unit | `cd apps/mobile && npx jest vlmPipeline.test.ts -x` | New test |
| P7-08 | ShimmerPlaceholder renders with animation | unit | `cd apps/mobile && npx jest vlmComponents.test.tsx -x` | New test |
| P7-09 | displayLabel shows shimmer state correctly | unit | `cd apps/mobile && npx jest useDetectionStore.test.ts -x` | Update needed |
| P7-10 | classify.tflite and labels_classify.json deleted | manual-only | Verify file absence after deletion | N/A |
| P7-11 | APK size reduced by ~4.9MB | manual-only | Build APK and compare size | N/A |

### Sampling Rate
- **Per task commit:** `cd apps/mobile && npx jest --testPathPattern="detection|vlm|store" --no-coverage`
- **Per wave merge:** `cd apps/mobile && npx jest --no-coverage`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Existing tests for inferenceRouter, modelLoader, modelBootstrap reference classify model -- must be rewritten before or during implementation
- [ ] No test for VLM retry behavior exists
- [ ] No test for text fallback box assignment exists
- [ ] No test for ShimmerPlaceholder component exists

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis of all 20+ files in detection, vlm, and component modules
- `apps/mobile/src/services/detection/inferenceRouter.ts` -- current two-stage pipeline
- `apps/mobile/src/services/detection/modelLoader.ts` -- current two-model loading
- `apps/mobile/src/services/vlm/vlmPipeline.ts` -- current VLM refinement flow
- `apps/mobile/src/screens/DetectionScreen.tsx` -- current detection orchestration
- `apps/mobile/src/components/detection/RefiningBadge.tsx` -- existing Reanimated animation pattern
- `apps/mobile/src/store/useDetectionStore.ts` -- displayLabel and state management
- `.planning/phases/07-*/07-CONTEXT.md` -- user decisions

### Secondary (MEDIUM confidence)
- [Callstack blog on shimmer effects](https://www.callstack.com/blog/performant-and-cross-platform-shimmers-in-react-native-apps) -- shared animated value pattern for multiple shimmers
- [react-native-reanimated-skeleton](https://github.com/marcuzgabriel/react-native-reanimated-skeleton) -- reference implementation for Reanimated skeleton patterns

### Tertiary (LOW confidence)
- None -- all findings are from direct codebase analysis and verified sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already installed, no new dependencies
- Architecture: HIGH - direct code analysis of every affected file, clear change scope
- Pitfalls: HIGH - identified from actual code patterns and data flow analysis
- Shimmer implementation: MEDIUM - based on existing RefiningBadge pattern plus verified web sources
- VLM retry strategy: MEDIUM - straightforward pattern but timeout values need tuning on-device

**Research date:** 2026-03-15
**Valid until:** Indefinitely (codebase-specific findings, no external dependency changes)
