---
phase: 07-remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection
plan: 01
subsystem: detection
tags: [tflite, yolo, efficientnet, pipeline, bbox, vlm]

# Dependency graph
requires:
  - phase: 02.6-on-device-vlm-integration
    provides: VLM identification pipeline that replaces classifier
provides:
  - Single-stage bbox-only detection pipeline (runBboxDetection)
  - Detect-only ModelSet type (no classify field)
  - Simplified constants (DETECT_CLASS_NAMES, DETECT_INPUT_SIZE only)
  - Zero_one-only image preprocessing (no ImageNet normalization)
  - 4.9MB APK size reduction from removed classify.tflite
affects: [07-02, 07-03, phase-3]

# Tech tracking
tech-stack:
  added: []
  patterns: ["bbox-only detection with Food Region placeholder labels", "isRefining shimmer state for VLM handoff"]

key-files:
  created: []
  modified:
    - apps/mobile/src/services/detection/inferenceRouter.ts
    - apps/mobile/src/services/detection/modelLoader.ts
    - apps/mobile/src/services/detection/types.ts
    - apps/mobile/src/services/detection/constants.ts
    - apps/mobile/src/services/detection/imagePreprocess.ts
    - apps/mobile/src/screens/DetectionScreen.tsx
    - apps/mobile/assets/models/model_manifest.json

key-decisions:
  - "All detection items labelled 'Food Region' with isRefining=true -- VLM provides actual food names"
  - "YOLO labels logged in __DEV__ mode only for debugging, not displayed to user"
  - "DetectionScreen caller updated to single-buffer API (Rule 3 auto-fix)"

patterns-established:
  - "Bbox-only pipeline: YOLO detects regions, VLM identifies food names asynchronously"
  - "isRefining=true shimmer state pattern for all new detections pending VLM"

requirements-completed: [P7-01, P7-02, P7-03, P7-04, P7-10]

# Metrics
duration: 9min
completed: 2026-03-15
---

# Phase 07 Plan 01: Remove EfficientNet Classification Summary

**Bbox-only YOLO detection with "Food Region" placeholder labels and isRefining shimmer state for VLM handoff**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-15T01:27:29Z
- **Completed:** 2026-03-15T01:36:11Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Removed EfficientNet classification stage entirely (classify.tflite, labels_classify.json, 9 training scripts, 2 configs)
- Simplified inferenceRouter from 5-param two-stage to 4-param single-stage (runBboxDetection)
- Reduced ModelSet to detect-only (single loadTensorflowModel call vs Promise.all with 2)
- Removed IMAGENET_MEAN/STD, CLASSIFY_CLASS_NAMES, CLASSIFY_INPUT_SIZE exports
- Removed imagenet normalization mode from imagePreprocess (2-param signature)
- All 278 tests pass (31 detection + 65 VLM + 182 others)

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete classification assets and training scripts** - `fa8f4015` (chore) -- assets/scripts already removed in prior 07-02 TDD commit
2. **Task 2 RED: Failing tests for bbox-only pipeline** - `2048ef9f` (test)
3. **Task 2 GREEN: Implement bbox-only pipeline** - `e85ffc9b` (feat)

**Plan metadata:** [pending final commit]

_Note: Task 1 deletions were already committed in fa8f4015 (07-02 TDD RED commit that included asset cleanup). Task 2 followed TDD (RED then GREEN)._

## Files Created/Modified
- `apps/mobile/src/services/detection/inferenceRouter.ts` - Renamed to runBboxDetection, removed classify stage, Food Region labels, isRefining=true
- `apps/mobile/src/services/detection/modelLoader.ts` - Detect-only ModelSet, single model loading
- `apps/mobile/src/services/detection/types.ts` - ModelSet.classify removed, PipelineStage.stage is detect|vlm
- `apps/mobile/src/services/detection/constants.ts` - Only DETECT_CLASS_NAMES and DETECT_INPUT_SIZE
- `apps/mobile/src/services/detection/imagePreprocess.ts` - Removed imagenet normalization, 2-param signature
- `apps/mobile/src/screens/DetectionScreen.tsx` - Updated to runBboxDetection with single buffer
- `apps/mobile/assets/models/model_manifest.json` - Single-stage detect-only manifest
- `apps/mobile/src/services/detection/__tests__/inferenceRouter.test.ts` - Rewritten for bbox-only API
- `apps/mobile/src/services/detection/__tests__/modelLoader.test.ts` - Rewritten for detect-only ModelSet
- `apps/mobile/src/services/detection/__tests__/modelBootstrap.test.ts` - Rewritten for single model fallback
- `apps/mobile/src/services/detection/__tests__/imagePreprocess.test.ts` - Updated for 2-param signature

## Decisions Made
- All detection items use generic "Food Region" label instead of YOLO class names (VLM provides actual names)
- YOLO class names still logged in `__DEV__` mode for debugging purposes
- isRefining=true on all items creates shimmer state until VLM identifies food
- DetectionScreen.tsx updated as Rule 3 deviation (caller must match new API)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated DetectionScreen.tsx caller to match new API**
- **Found during:** Task 2 (pipeline simplification)
- **Issue:** DetectionScreen.tsx imports runDetectionPipeline with 5 params and CLASSIFY_INPUT_SIZE -- would break at runtime
- **Fix:** Updated import to runBboxDetection, removed classifyBuffer/CLASSIFY_INPUT_SIZE, single preprocessImageForModel call
- **Files modified:** apps/mobile/src/screens/DetectionScreen.tsx
- **Verification:** All 278 tests pass, no compile errors
- **Committed in:** e85ffc9b (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for correctness -- caller must match the new API signature. No scope creep.

## Issues Encountered
- Task 1 classification asset deletions were already committed in a prior 07-02 TDD RED commit (fa8f4015). The classify.tflite and training scripts were not git-tracked (gitignored), so the only tracked change was model_manifest.json which was also already updated. No new commit needed for Task 1.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Detection pipeline is now single-stage bbox-only
- VLM pipeline (07-02) can identify food items from "Food Region" bounding boxes
- Ready for 07-02 (VLM primary identification) and 07-03 (remaining cleanup)

## Self-Check: PASSED

- All 8 key files verified present
- Commits 2048ef9f and e85ffc9b verified in git log
- 278/278 tests pass across 22 test suites
- No classify references in production code

---
*Phase: 07-remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection*
*Completed: 2026-03-15*
