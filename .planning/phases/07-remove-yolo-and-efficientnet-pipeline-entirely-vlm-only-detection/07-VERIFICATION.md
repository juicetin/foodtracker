---
phase: 07-remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection
verified: 2026-03-15T03:30:00Z
status: human_needed
score: 11/11 must-haves verified
re_verification: false
human_verification:
  - test: "Verify shimmer animation renders correctly in bounding box label chips during VLM processing"
    expected: "Animated gray rectangle pulses (opacity 1.0->0.4, 800ms) in label chip; no text visible while isRefining=true; food name fades in with 200ms FadeIn when VLM result arrives"
    why_human: "Reanimated animation cannot be verified programmatically in test environment; requires visual inspection on device or emulator"
  - test: "Verify shimmer renders in DetectionListItem for name and portion rows during VLM processing"
    expected: "Two shimmer rectangles (120x14 for name, 60x10 for portion) appear in list items while isRefining=true; replaced by actual text when VLM results arrive"
    why_human: "Component render with animation state cannot be asserted in Jest; requires visual device test"
  - test: "Verify VLM failure banner and text fallback flow end-to-end"
    expected: "When VLM fails twice, orange banner appears ('AI identification unavailable. Describe your meal to get nutrition info.'); typing 'rice, curry' in text input assigns 'rice' to largest bbox and 'curry' to next largest; items in list update with typed names"
    why_human: "Requires actual VLM failure condition and interactive text input — not simulatable in unit tests"
  - test: "Verify APK size reduction"
    expected: "APK is approximately 4.9MB smaller than pre-phase-7 build due to removal of classify.tflite"
    why_human: "Requires building the APK and comparing against a baseline build"
---

# Phase 07: VLM-Only Detection Verification Report

**Phase Goal:** Strip the EfficientNet classification stage entirely and reduce YOLO to bounding-box-only duty, making VLM the sole source of food identification with shimmer UX during processing and graceful text fallback on VLM failure

**Verified:** 2026-03-15T03:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | EfficientNet classify.tflite, labels_classify.json, and all classification training scripts are deleted | VERIFIED | `assets/models/` contains only `detect.tflite`, `labels_detect.json`, `model_manifest.json`; 8 training scripts confirmed absent via `ls` check |
| 2 | YOLO outputs bounding boxes only — every detection is a generic "Food Region" until VLM identifies it | VERIFIED | `inferenceRouter.ts` line 133: `className: 'Food Region'`; `isRefining: true` set on all items; no classify stage |
| 3 | inferenceRouter is single-stage (bbox-only), modelLoader loads detect-only, no ImageNet normalization in preprocessing | VERIFIED | `runBboxDetection` has 4 params, no classifyBuffer; `ModelSet` has `detect` field only; `imagePreprocess.ts` is 2-param with zero_one normalization only |
| 4 | VLM is the primary food identifier with one silent retry on failure | VERIFIED | `vlmPipeline.ts` exports `identifyWithRetry` with double try-catch pattern returning `{ dishes: [] }` on double failure; `runVlmIdentification` calls `identifyWithRetry` |
| 5 | Shimmer animation appears in bounding box labels and detection list items while VLM processes | VERIFIED (automated) / NEEDS HUMAN (visual) | `BoundingBoxOverlay.tsx` line 87-89: `{item.isRefining ? <ShimmerPlaceholder width={80} height={14} /> : ...}`; `DetectionListItem.tsx` lines 103-107: two shimmer placeholders for name and portion |
| 6 | When VLM fails, user sees "Describe your meal" text input; typed dish names are assigned to boxes by size order with KG nutrition lookup | VERIFIED (automated) / NEEDS HUMAN (visual) | `DetectionScreen.tsx` lines 763-769: `vlmFailed` state triggers orange banner; lines 320-353: `assignDishesToBoxes` called on comma/newline-split text; KG `searchDish` called in `__DEV__` mode |

**Score:** 6/6 truths verified (4 human-confirmable checks remain for visual/interactive behaviors)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `apps/mobile/src/services/detection/inferenceRouter.ts` | Bbox-only inference pipeline | VERIFIED | Exports `runBboxDetection` (4 params), `formatFoodLabel`; single detect stage; Food Region labels; isRefining=true |
| `apps/mobile/src/services/detection/modelLoader.ts` | Detect-only model loading | VERIFIED | Loads single detect model; `ModelSet` has `detect` field only; no classify pack lookup |
| `apps/mobile/src/services/detection/types.ts` | Simplified ModelSet with detect field only | VERIFIED | `interface ModelSet { detect: TFLiteModel; }`; `PipelineStage.stage` is `'detect' | 'vlm'` |
| `apps/mobile/src/services/detection/constants.ts` | DETECT_CLASS_NAMES and DETECT_INPUT_SIZE only | VERIFIED | Only two exports; no CLASSIFY_*, IMAGENET_* |
| `apps/mobile/assets/models/model_manifest.json` | Single-model manifest (detect only) | VERIFIED | `"pipeline": "single-stage"`; one model entry: `ggcd-detect-v1`; no EfficientNet entry |
| `apps/mobile/src/services/vlm/vlmPipeline.ts` | Primary VLM identification with retry and text fallback | VERIFIED | Exports `runVlmIdentification`, `identifyWithRetry`, `assignDishesToBoxes`; positional matching only; no `computeWordOverlap` |
| `apps/mobile/src/store/useDetectionStore.ts` | Detection store with VLM-primary displayLabel logic | VERIFIED | `displayLabel` returns `vlmLabel` > `''` (isRefining shimmer) > `'Unknown food'` |
| `apps/mobile/src/components/detection/ShimmerPlaceholder.tsx` | Reusable shimmer placeholder component | VERIFIED | Reanimated opacity pulse (1.0->0.4, 800ms repeat), FadeIn/FadeOut 200ms, props: width/height/style/borderRadius |
| `apps/mobile/src/screens/DetectionScreen.tsx` | VLM-primary detection flow with text fallback | VERIFIED | Uses `runBboxDetection` (4 params, single buffer); `runVlmIdentification` as primary; `assignDishesToBoxes` on VLM failure; `vlmFailed` state triggers banner |
| `apps/mobile/src/components/detection/BoundingBoxOverlay.tsx` | Bounding boxes with shimmer in label chips | VERIFIED | Imports `ShimmerPlaceholder`; renders shimmer when `item.isRefining`, text with `FadeIn` when resolved |
| `apps/mobile/src/components/detection/DetectionListItem.tsx` | Detection list item with shimmer placeholder rows | VERIFIED | Imports `ShimmerPlaceholder`; renders two shimmer bars (name 120x14, portion 60x10) when `item.isRefining` |
| `apps/mobile/src/components/detection/index.ts` | Exports ShimmerPlaceholder, not RefiningBadge | VERIFIED | `export { ShimmerPlaceholder } from './ShimmerPlaceholder'`; no RefiningBadge export present |
| `apps/mobile/src/components/detection/RefiningBadge.tsx` | DELETED | VERIFIED | File does not exist |
| `training/train_classify.py` | DELETED | VERIFIED | All 8 classification training scripts absent |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `inferenceRouter.ts` | `modelLoader.ts` | `getModelSet()` returns detect-only ModelSet | WIRED | `getModelSet` called at line 81; `models.detect.run()` at line 94 |
| `inferenceRouter.ts` | `postProcess.ts` | `decodeYoloOutput` for bbox extraction | WIRED | `decodeYoloOutput` imported and called at line 111 |
| `vlmPipeline.ts` | `vlmService.ts` | `vlmService.identify()` for VLM inference | WIRED | `vlmService.identify(photoUri, userText)` called twice in `identifyWithRetry` (try/catch pattern) |
| `vlmPipeline.ts` | `knowledge-graph` | `getKnowledgeGraphService` + `searchDish` | WIRED | `getKnowledgeGraphService()` called at line 104; `kgService.searchDish(vlmDish.name)` at line 115 |
| `useDetectionStore.ts` | `detection/types.ts` | `DetectedItem.isRefining` and `vlmLabel` fields | WIRED | `displayLabel` reads `item.vlmLabel` and `item.isRefining`; `setRefining` mutates all item `isRefining` flags |
| `BoundingBoxOverlay.tsx` | `ShimmerPlaceholder.tsx` | ShimmerPlaceholder in bbox label chip when isRefining | WIRED | Imported at line 10; rendered at line 88 inside `item.isRefining` conditional |
| `DetectionListItem.tsx` | `ShimmerPlaceholder.tsx` | ShimmerPlaceholder in name area when isRefining | WIRED | Imported at line 17; two instances rendered at lines 105-106 |
| `DetectionScreen.tsx` | `inferenceRouter.ts` | `runBboxDetection` (single buffer, no classifyBuffer) | WIRED | Imported at line 17; called at lines 471-476 with 4 params |
| `DetectionScreen.tsx` | `vlmPipeline.ts` | `runVlmIdentification` + `assignDishesToBoxes` | WIRED | Imported at line 35; `runVlmIdentification` called at lines 247-251; `assignDishesToBoxes` called at line 331 |

---

### Requirements Coverage

The plans declare requirements P7-01 through P7-11. REQUIREMENTS.md uses a separate ID scheme (DET-*, DAT-*, ML-*, etc.) and does not yet contain P7-* entries — these are phase-internal requirement IDs defined within the plan files themselves, not cross-referenced to the global requirements document.

| Plan Req ID | Declared In | Topic | Status | Evidence |
|------------|-------------|-------|--------|----------|
| P7-01 | 07-01-PLAN | Delete classify.tflite and labels_classify.json | SATISFIED | Assets directory verified: only detect.tflite, labels_detect.json, model_manifest.json |
| P7-02 | 07-01-PLAN | Delete classification training scripts | SATISFIED | 8 scripts confirmed absent: train_classify.py, export_mobile.py, train_binary.py, eval_classification.py, merge_datasets.py, auto_label.py, download_datasets.py, food-classify.yaml, food-binary.yaml |
| P7-03 | 07-01-PLAN | inferenceRouter bbox-only (no classify stage) | SATISFIED | `runBboxDetection` is 4-param; no classify stage in production code |
| P7-04 | 07-01-PLAN | modelLoader loads detect-only ModelSet | SATISFIED | Single `loadTensorflowModel` call; `ModelSet.classify` is absent |
| P7-05 | 07-02-PLAN | VLM pipeline is primary food identifier | SATISFIED | `runVlmIdentification` is primary; YOLO provides bboxes only |
| P7-06 | 07-02-PLAN | One silent retry on VLM failure | SATISFIED | `identifyWithRetry` double try-catch; returns `{ dishes: [] }` on double failure |
| P7-07 | 07-02-PLAN | Positional dish assignment (area descending) | SATISFIED | `assignDishesToBoxes` and `matchVlmToItems` both sort by `bbox.w * bbox.h` descending |
| P7-08 | 07-03-PLAN | Shimmer in bbox and list items during VLM | SATISFIED (code) / NEEDS HUMAN (visual) | Both components conditionally render ShimmerPlaceholder on `isRefining` |
| P7-09 | 07-02-PLAN | displayLabel shimmer-first (empty string during isRefining) | SATISFIED | `displayLabel` returns `''` when `item.isRefining` |
| P7-10 | 07-01-PLAN | No CLASSIFY_*, IMAGENET_* in constants | SATISFIED | constants.ts exports only `DETECT_CLASS_NAMES` and `DETECT_INPUT_SIZE` |
| P7-11 | 07-03-PLAN | VLM failure -> text input -> KG nutrition | SATISFIED (code) / NEEDS HUMAN (visual) | `vlmFailed` state, banner, `assignDishesToBoxes`, KG `searchDish` all wired |

**Note on orphaned requirements:** No P7-* IDs appear in REQUIREMENTS.md traceability table. This is expected — Phase 7 is a pipeline simplification/refactoring phase, not one that adds new v1 product requirements. The P7-* IDs are scoped to the phase plans only.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `apps/mobile/src/services/detection/imagePreprocess.ts` | 119 | `// TODO: validate pixel extraction on-device with real model packs` | Info | Code comment noting a validation gap in the PNG decoder path. The todo is in the test-only fallback path (raw RGBA mock data), not production PNG handling. No functional impact. |

---

### Human Verification Required

#### 1. Shimmer animation in bounding boxes

**Test:** Build the app and open DetectionScreen. Take a food photo. Observe the bounding box label chips immediately after YOLO detection (before VLM completes).
**Expected:** Each bounding box label chip shows a gray animated rectangle (80x14px) that pulses between full opacity and 40% opacity on an 800ms cycle. No text visible. When VLM identifies the food, text fades in over 200ms and shimmer disappears.
**Why human:** Reanimated animation lifecycle cannot be verified in Jest; `entering={FadeIn.duration(200)}` and `withRepeat(withTiming(0.4, { duration: 800 }), -1, true)` require a running RN runtime to observe.

#### 2. Shimmer animation in detection list items

**Test:** Same photo session as above. Observe the detection list below the photo during VLM processing.
**Expected:** Each list item shows two gray shimmer bars — one wide (120x14px) where the food name will appear, one shorter (60x10px) where the portion estimate will appear. Confidence dot and percentage are always visible. When VLM resolves, bars are replaced by actual name and portion text.
**Why human:** Same reason — animation state requires device runtime.

#### 3. VLM failure banner and text fallback

**Test:** Force a VLM failure scenario (e.g., test with VLM model unloaded or mock failure). Observe DetectionScreen behavior.
**Expected:** Orange banner appears at the top of results with text "AI identification unavailable. Describe your meal to get nutrition info." The MealTextInput becomes the primary input. Typing "rice, curry" should assign "rice" to the largest detected box and "curry" to the second largest. List items should update to show the typed names.
**Why human:** Requires inducing VLM failure condition and interactive text entry; cannot be simulated in unit tests.

#### 4. APK size reduction

**Test:** Build a release APK and compare size against the last pre-phase-7 build.
**Expected:** APK is approximately 4.9MB smaller (the size of the removed `classify.tflite`). The asset bundle should not contain `classify.tflite` or `labels_classify.json`.
**Why human:** Requires APK build and size comparison; cannot be verified from source code alone.

---

### Test Suite Results

All 60 automated tests across 6 suites pass:

```
PASS src/services/detection/__tests__/inferenceRouter.test.ts
PASS src/services/detection/__tests__/imagePreprocess.test.ts
PASS src/store/__tests__/useDetectionStore.test.ts
PASS src/services/detection/__tests__/modelLoader.test.ts
PASS src/services/detection/__tests__/modelBootstrap.test.ts
PASS src/services/vlm/__tests__/vlmPipeline.test.ts

Tests: 60 passed, 60 total
Time:  0.44s
```

---

### Summary

Phase 07 is code-complete and all automated verifications pass. Every major structural change has been implemented and wired:

- EfficientNet classifier is fully removed (assets, training scripts, inference code, type contracts)
- YOLO is reduced to bbox-only duty; all 241 class names remain only for tensor decoding, not user display
- VLM is the sole food identification source, with retry logic and a graceful text fallback
- Shimmer UX is fully wired in BoundingBoxOverlay and DetectionListItem via ShimmerPlaceholder
- DetectionScreen orchestrates the new flow: single-buffer preprocessing -> runBboxDetection -> shimmer -> runVlmIdentification -> label fade-in, with assignDishesToBoxes fallback on failure

The four remaining human verification items are visual/interactive behaviors that require a running device or emulator to confirm. Automated evidence strongly indicates they will pass (code paths are correctly wired and component conditionals are correct), but the animation quality and fallback flow feel require eyes-on confirmation.

---

_Verified: 2026-03-15T03:30:00Z_
_Verifier: Claude (gsd-verifier)_
