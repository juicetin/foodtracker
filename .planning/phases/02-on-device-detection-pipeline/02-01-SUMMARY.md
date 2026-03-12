---
phase: 02-on-device-detection-pipeline
plan: 01
subsystem: detection
tags: [tflite, yolo, react-native-fast-tflite, expo-plugin, typescript, python]

# Dependency graph
requires:
  - phase: 01-food-detection-foundation
    provides: "Pack system types and DB schema with bounding box columns"
provides:
  - "Detection pipeline type contracts (DetectedItem, BoundingBox, InferenceResult, etc.)"
  - "react-native-fast-tflite configured as Expo plugin with CoreML/GPU delegates"
  - "Metro bundler tflite asset support"
  - "YOLO-to-TFLite FP16 export script for three-stage pipeline"
  - "Jest mock for react-native-fast-tflite"
affects: [02-02, 02-03, 02-04, 02-05]

# Tech tracking
tech-stack:
  added: [react-native-fast-tflite, ultralytics (Python)]
  patterns: [three-stage-pipeline-types, normalised-bbox-coordinates, confidence-level-buckets]

key-files:
  created:
    - apps/mobile/src/services/detection/types.ts
    - apps/mobile/__mocks__/react-native-fast-tflite.ts
    - training/export_mobile.py
  modified:
    - apps/mobile/app.json
    - apps/mobile/metro.config.js
    - apps/mobile/package.json

key-decisions:
  - "TFLiteModel interface uses ArrayBufferLike[] for both sync/async run methods, matching react-native-fast-tflite's TensorflowModel shape"
  - "FP16 quantisation only (no INT8) -- avoids calibration dataset requirement and accuracy loss on food colour subtleties"
  - "NMS performed in JavaScript, not baked into TFLite model -- cross-platform portability per RESEARCH.md"

patterns-established:
  - "Detection types in services/detection/types.ts: all detection modules import from this canonical location"
  - "Confidence level buckets: high >= 0.80, medium >= 0.50, low < 0.50 with colour constants"
  - "Three-stage pipeline model (binary -> detect -> classify) typed via ModelSet interface"

requirements-completed: [DET-01]

# Metrics
duration: 3min
completed: 2026-03-12
---

# Phase 2 Plan 1: Detection Pipeline Contracts Summary

**Detection pipeline type contracts with react-native-fast-tflite plugin config and YOLO FP16 TFLite export script**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-12T12:14:17Z
- **Completed:** 2026-03-12T12:17:32Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- All detection pipeline type contracts defined (DetectedItem, BoundingBox, PortionEstimate, InferenceResult, RawDetection, ModelSet, TFLiteModel, MealType, CorrectionRecord, DetectionSessionState)
- react-native-fast-tflite configured as Expo plugin with CoreML and GPU delegate support
- Metro bundler configured to recognise .tflite files as assets
- YOLO-to-TFLite export script with three-stage pipeline support, FP16 quantisation, manifest generation, and validation
- Jest mock for unit testing inference-dependent modules

## Task Commits

Each task was committed atomically:

1. **Task 1: Create detection type contracts and configure mobile build tooling** - `7b039ac7` (feat)
2. **Task 2: Create YOLO-to-TFLite export script** - `8007787e` (feat)

## Files Created/Modified
- `apps/mobile/src/services/detection/types.ts` - All detection pipeline type contracts and helper functions
- `apps/mobile/app.json` - Added react-native-fast-tflite plugin with CoreML/GPU delegates
- `apps/mobile/metro.config.js` - Added tflite to resolver asset extensions
- `apps/mobile/package.json` - Added react-native-fast-tflite dependency
- `apps/mobile/__mocks__/react-native-fast-tflite.ts` - Jest mock for TFLite model loading
- `training/export_mobile.py` - YOLO-to-TFLite FP16 export script with CLI, manifest generation, and validation

## Decisions Made
- TFLiteModel interface uses ArrayBufferLike[] matching react-native-fast-tflite's actual API shape
- FP16 quantisation only (no INT8) to avoid calibration dataset requirement and preserve food colour accuracy
- NMS is not baked into TFLite models; performed in JavaScript for cross-platform portability
- Export script generates model_manifest.json for downstream pack system integration

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Pre-existing TypeScript errors in `src/services/nutrition/regionalResolver.ts` (Promise/Uint8Array type mismatch) -- not related to this plan's changes, out of scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Type contracts ready for Plans 02-03 through 02-05 to import
- react-native-fast-tflite plugin ready for native builds
- Export script ready to convert trained YOLO models once training runs complete
- Jest mock enables unit testing without native TFLite runtime

## Self-Check: PASSED

All files exist, all commits verified, all content checks passed.

---
*Phase: 02-on-device-detection-pipeline*
*Completed: 2026-03-12*
