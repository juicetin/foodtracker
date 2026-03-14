# Phase 7: Remove YOLO and EfficientNet pipeline entirely — VLM-only detection - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Strip the EfficientNet classification stage and all food-labeling logic from the YOLO pipeline. YOLO is retained purely as a spatial bounding box detector (no food-level data). VLM becomes the sole source of food identification. The inference router is rewritten from a two-stage YOLO+EfficientNet pipeline to a YOLO-boxes + VLM-identification architecture. EfficientNet model, labels, training scripts, and all classification code are fully removed.

</domain>

<decisions>
## Implementation Decisions

### YOLO model disposition
- Keep current GGCD YOLOv8n (241 food classes, 3.6MB) for bounding box spatial detection only
- Ignore all class label output from YOLO — treat every detection as "food region"
- Do NOT swap to a different/smaller model — current model works and is already tested
- YOLO training scripts (train_detect.py, export_ggcd_detect.py) are kept in the repo

### EfficientNet removal — full removal
- Delete classify.tflite (4.9MB) and labels_classify.json from bundled assets
- Remove all EfficientNet inference code, ImageNet normalization constants, classification preprocessing
- Remove Python training scripts: train_classify.py, export_mobile.py, eval_classification.py
- Remove dataset merge/acquisition scripts related to classifier training
- Remove all classification-related tests
- Clean break — git history preserves everything if ever needed

### Pre-VLM bounding box UX
- YOLO boxes appear instantly (50-80ms) with shimmer/skeleton animation inside each box where the label will appear
- Detection list below photo also shows shimmer placeholder items (one per YOLO box) — user sees item count immediately
- When VLM results arrive, shimmer fades out and label fades in (animated transition, consistent with existing Phase 2.6 refining animation)

### VLM failure fallback
- On VLM failure: silent retry once behind the shimmer (user doesn't see the retry)
- If retry also fails: show "Describe your meal" text prompt
- Text prompt tells user that suggested ingredients will be populated from KG
- Map user-typed dishes to YOLO boxes by count (1:1 assignment by box size order — largest box = first dish)
- KG fuzzy search matches dish names → recipe decomposition → nutrition

### Claude's Discretion
- Exact shimmer animation implementation (library choice, timing)
- How to restructure inferenceRouter (YOLO-boxes-only stage + VLM identification stage)
- VLM retry timeout duration and backoff strategy
- How to handle count mismatch (more/fewer dishes typed than boxes detected)
- Whether to simplify model_manifest.json or keep it for YOLO entry only

</decisions>

<specifics>
## Specific Ideas

- User explicitly noted: "we can use YOLO JUST for bounding boxes, ensuring we don't use it for any FOOD level data" — this is the core insight driving the phase
- The existing "Refining..." badge from Phase 2.6 should be replaced by the shimmer pattern — cleaner since VLM is no longer "refining" YOLO labels but is the primary identifier
- When VLM fails and user types dish names, KG should surface suggested ingredients (not just accept the dish name silently)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `vlmService.ts`: Singleton VLM service with init/identify/release — becomes the primary detection engine
- `vlmPipeline.ts`: VLM refinement pipeline — needs rewrite from "refinement" to "primary identification"
- `vlmPrompts.ts`: Prompt engineering for food identification — reusable as-is
- `BoundingBoxOverlay.tsx`: Renders YOLO bbox overlays on photo — keep, just remove label rendering until VLM
- `RefiningBadge.tsx`: Animated badge — replace with shimmer pattern
- `postProcess.ts`: `decodeYoloOutput()` and `nonMaxSuppression()` — keep for bbox extraction, remove class name lookups
- `portionBridge.ts`: Portion estimation from bbox geometry — fully reusable, not tied to food labels
- `KnowledgeGraphService`: Nutrition lookup by dish name — works with any name source (VLM or user text)

### Established Patterns
- Zustand detection store with ephemeral state (in-memory only until Log Meal)
- Detection flow state machine: idle → picking → detecting → results
- PackManager for model downloads with streaming (Phase 2.6 upgrade)
- `react-native-fast-tflite` for YOLO model loading — stays for YOLO bboxes and future Scale OCR (Phase 5)

### Integration Points
- `DetectionScreen.tsx`: Orchestrates the full flow — needs significant rewrite
- `inferenceRouter.ts`: Two-stage pipeline → rewrite to bbox-only + VLM
- `modelLoader.ts`: Currently loads ModelSet {detect, classify} → simplify to detect-only
- `constants.ts`: DETECT_CLASS_NAMES stays (but unused for labeling), CLASSIFY_CLASS_NAMES removed
- `types.ts`: `ModelSet` type simplified, `DetectedItem` keeps VLM fields as primary

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection*
*Context gathered: 2026-03-15*
