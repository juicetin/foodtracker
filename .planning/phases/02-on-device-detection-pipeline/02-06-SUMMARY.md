---
phase: 02-on-device-detection-pipeline
plan: 06
subsystem: detection
tags: [expo-image-manipulator, png-decoder, tflite, preprocessing, typescript]

# Dependency graph
requires:
  - phase: 02-on-device-detection-pipeline
    provides: "inferenceRouter pipeline, DetectionScreen orchestration, model types"
provides:
  - "preprocessImageForModel: photo URI to 640x640 Float32Array normalised 0-1"
  - "TS-clean inferenceRouter with no compile errors"
  - "DetectionScreen wired to real photo pixel data (no placeholder buffer)"
affects: [03-nutrition-resolution-diary, model-pack-validation]

# Tech tracking
tech-stack:
  added: [expo-image-manipulator@14.0.8]
  patterns: [pure-js-png-decoder, base64-pixel-extraction, arraybuffer-cast-pattern]

key-files:
  created:
    - apps/mobile/src/services/detection/imagePreprocess.ts
    - apps/mobile/src/services/detection/__tests__/imagePreprocess.test.ts
  modified:
    - apps/mobile/src/services/detection/inferenceRouter.ts
    - apps/mobile/src/screens/DetectionScreen.tsx
    - apps/mobile/package.json

key-decisions:
  - "Pure-JS PNG decoder for pixel extraction -- no additional native dependencies"
  - "manipulateAsync legacy API for simplicity over new context-based API"
  - "Direct ArrayBuffer cast (as ArrayBuffer) instead of instanceof Float32Array checks"
  - "PNG format for base64 output (lossless) to preserve pixel accuracy for model input"

patterns-established:
  - "ArrayBuffer cast pattern: new Float32Array(output[0] as ArrayBuffer) for TFLite outputs"
  - "Image preprocessing pipeline: resize -> base64 PNG -> decode -> normalise -> Float32Array"

requirements-completed: [DET-01, DET-05, DET-06]

# Metrics
duration: 4min
completed: 2026-03-13
---

# Phase 2 Plan 06: Gap Closure Summary

**Image-to-tensor bridge via expo-image-manipulator with pure-JS PNG decoder, plus TS-clean inferenceRouter**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-12T19:26:48Z
- **Completed:** 2026-03-12T19:30:57Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Created imagePreprocess.ts with preprocessImageForModel that resizes photos to 640x640 and extracts normalised RGB pixel data as Float32Array
- Fixed two TS2339 compile errors in inferenceRouter.ts by replacing instanceof Float32Array with direct ArrayBuffer cast
- Wired DetectionScreen to use real photo pixel data instead of zeroed placeholder buffer
- 5 new imagePreprocess tests + all 74 existing detection tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing imagePreprocess tests** - `35e58ec2` (test)
2. **Task 1 GREEN: Fix inferenceRouter TS + create imagePreprocess** - `6d4a05c6` (feat)
3. **Task 2: Wire imagePreprocess into DetectionScreen** - `15cb04ee` (feat)

_TDD task 1 had separate RED and GREEN commits._

## Files Created/Modified
- `apps/mobile/src/services/detection/imagePreprocess.ts` - Image-to-tensor bridge: resize + PNG decode + RGB normalisation
- `apps/mobile/src/services/detection/__tests__/imagePreprocess.test.ts` - Unit tests for preprocessing contract (5 tests)
- `apps/mobile/src/services/detection/inferenceRouter.ts` - Fixed TS2339 errors with ArrayBuffer cast pattern
- `apps/mobile/src/screens/DetectionScreen.tsx` - Wired preprocessImageForModel, removed placeholder buffer
- `apps/mobile/package.json` - Added expo-image-manipulator ~14.0.8
- `apps/mobile/package-lock.json` - Updated lockfile

## Decisions Made
- Used manipulateAsync (legacy API) instead of the new context-based API -- simpler, returns base64 directly, well-tested
- Implemented a pure-JS PNG decoder rather than adding another native dependency -- handles filter types 0-4 per PNG spec
- Used PNG format (lossless) for base64 output to preserve pixel accuracy for model input
- Direct `as ArrayBuffer` cast for TFLite outputs -- safe because react-native-fast-tflite returns ArrayBuffer at runtime

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 2 detection pipeline is now fully wired: photo -> preprocess -> inference -> results
- On-device pixel extraction should be validated with real model packs (TODO comments in code)
- Ready to proceed to Phase 3 (Nutrition Resolution + Diary)

## Self-Check: PASSED

All files created exist. All commit hashes verified in git log.

---
*Phase: 02-on-device-detection-pipeline*
*Completed: 2026-03-13*
