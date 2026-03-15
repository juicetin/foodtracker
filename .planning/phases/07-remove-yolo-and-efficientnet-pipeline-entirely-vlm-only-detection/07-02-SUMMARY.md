---
phase: 07-remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection
plan: 02
subsystem: detection
tags: [vlm, pipeline, retry, positional-matching, shimmer, zustand]

# Dependency graph
requires:
  - phase: 02.6-on-device-vlm-integration
    provides: "VLM service singleton (vlmService) and pipeline foundation (vlmPipeline.ts)"
provides:
  - "runVlmIdentification: primary VLM food identification with retry logic"
  - "identifyWithRetry: VLM inference with one silent retry, empty fallback"
  - "assignDishesToBoxes: positional dish-to-bbox mapping by area descending"
  - "displayLabel shimmer-first logic (empty string during isRefining)"
affects: [07-03-detection-screen-rewrite]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Positional matching: bbox area descending replaces substring/word-overlap strategies"
    - "Silent retry: one retry on VLM failure, empty result on double fail (no throw)"
    - "Shimmer-first displayLabel: empty string signals shimmer to UI during identification"

key-files:
  created: []
  modified:
    - "apps/mobile/src/services/vlm/vlmPipeline.ts"
    - "apps/mobile/src/services/vlm/__tests__/vlmPipeline.test.ts"
    - "apps/mobile/src/store/useDetectionStore.ts"
    - "apps/mobile/src/store/__tests__/useDetectionStore.test.ts"

key-decisions:
  - "Positional matching only (no substring/word-overlap) since className is always 'Food Region'"
  - "identifyWithRetry returns { dishes: [] } on double failure instead of throwing"
  - "displayLabel returns empty string during isRefining (shimmer) instead of 'Identifying...'"
  - "displayLabel returns 'Unknown food' as final fallback instead of className"

patterns-established:
  - "Positional assignment: VLM dishes assigned to bounding boxes sorted by area descending"
  - "Silent retry pattern: try-catch wrapping with fallback to empty result"

requirements-completed: [P7-05, P7-06, P7-07, P7-09]

# Metrics
duration: 4min
completed: 2026-03-15
---

# Phase 07 Plan 02: VLM Pipeline Summary

**VLM pipeline rewritten as primary identification engine with retry logic, positional bbox assignment, and shimmer-first displayLabel**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-15T01:27:34Z
- **Completed:** 2026-03-15T01:31:06Z
- **Tasks:** 2 (Task 1 TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Rewritten vlmPipeline from refinement layer to primary VLM identification with identifyWithRetry
- Simplified matching to pure positional assignment (area-based) -- substring/word-overlap removed as dead code
- New assignDishesToBoxes export for text fallback path used by DetectionScreen
- displayLabel updated with shimmer-first flow: empty string during isRefining, 'Unknown food' fallback

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `fa8f4015` (test)
2. **Task 1 GREEN: VLM pipeline implementation** - `54248a54` (feat)
3. **Task 2: displayLabel shimmer-first update** - `c338f4c7` (feat)

## Files Created/Modified
- `apps/mobile/src/services/vlm/vlmPipeline.ts` - Primary VLM identification with retry, positional matching, assignDishesToBoxes
- `apps/mobile/src/services/vlm/__tests__/vlmPipeline.test.ts` - 15 tests: retry logic, positional assignment, KG lookup, error handling
- `apps/mobile/src/store/useDetectionStore.ts` - displayLabel shimmer-first logic
- `apps/mobile/src/store/__tests__/useDetectionStore.test.ts` - Updated displayLabel tests for shimmer and unknown fallback

## Decisions Made
- Positional matching only (no substring/word-overlap) since className is always 'Food Region' in the new flow
- identifyWithRetry returns { dishes: [] } on double failure instead of propagating errors (fail-safe)
- displayLabel returns empty string during isRefining (shimmer UI state) instead of 'Identifying...'
- displayLabel uses 'Unknown food' as final fallback instead of className (className is meaningless 'Food Region')

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated useDetectionStore test for new displayLabel behavior**
- **Found during:** Task 2
- **Issue:** Existing test asserted displayLabel falls back to className ('Curry'), but new logic returns 'Unknown food' since className is now meaningless
- **Fix:** Replaced className fallback test with shimmer state test and 'Unknown food' fallback test
- **Files modified:** apps/mobile/src/store/__tests__/useDetectionStore.test.ts
- **Verification:** All 29 tests pass across both test suites
- **Committed in:** c338f4c7 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in existing test)
**Impact on plan:** Test update was necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- vlmPipeline exports runVlmIdentification, identifyWithRetry, and assignDishesToBoxes for Plan 03
- DetectionScreen.tsx still imports runVlmRefinement (Plan 03 handles updating the consumer)
- displayLabel is ready for shimmer UI components in Plan 03

## Self-Check: PASSED

All files exist. All commits verified (fa8f4015, 54248a54, c338f4c7).

---
*Phase: 07-remove-yolo-and-efficientnet-pipeline-entirely-vlm-only-detection*
*Completed: 2026-03-15*
