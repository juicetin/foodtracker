---
phase: 04-gallery-scanning-deduplication
plan: 02
subsystem: gallery-scanning
tags: [expo-background-task, expo-task-manager, zustand, gallery-ui, workmanager, gemini-nano-foreground]

# Dependency graph
requires:
  - phase: 04-01
    provides: galleryScanService, foodClassifier, mealGrouper, photoImporter service layer
provides:
  - Background scheduler for periodic gallery photo discovery (4hr WorkManager)
  - Foreground drain trigger for Gemini Nano classification on app active
  - Zustand store for gallery scan state and UI progress tracking
  - GalleryScanScreen with manual trigger, auto-scan toggle, permission handling
  - Navigation wiring from ProfileScreen to GalleryScanScreen
affects: [deduplication, gallery-review-ui, settings]

# Tech tracking
tech-stack:
  added: []
  patterns: [side-effect-import-defineTask, appstate-foreground-drain, zustand-persist-partialize]

key-files:
  created:
    - apps/mobile/src/services/gallery/galleryScanScheduler.ts
    - apps/mobile/src/services/gallery/__tests__/galleryScanScheduler.test.ts
    - apps/mobile/src/store/useGalleryScanStore.ts
    - apps/mobile/src/screens/GalleryScanScreen.tsx
  modified:
    - apps/mobile/app.json
    - apps/mobile/App.tsx
    - apps/mobile/src/types/index.ts
    - apps/mobile/src/screens/index.ts
    - apps/mobile/src/screens/ProfileScreen.tsx
    - apps/mobile/src/navigation/RootNavigator.tsx

key-decisions:
  - "require() instead of dynamic import() in scheduler for Jest compatibility (no --experimental-vm-modules)"
  - "AppState listener in App.tsx fires triggerForegroundDrain on every foreground event (silent, fire-and-forget)"
  - "Zustand store partialize persists only scanEnabled and lastScanResult (not transient isScanning/progress)"

patterns-established:
  - "AppState foreground drain pattern: side-effect import + useRef tracking for app state transitions"
  - "Gallery scan store: Zustand + AsyncStorage persist with partialize for selective persistence"

requirements-completed: [GAL-01, GAL-02, GAL-05]

# Metrics
duration: 5min
completed: 2026-03-20
---

# Phase 04 Plan 02: Gallery Scan UI and Scheduling Summary

**Background WorkManager gallery discovery (4hr), foreground Gemini Nano drain on app active, GalleryScanScreen with manual trigger/auto-scan toggle/progress -- 8 scheduler tests passing**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-20T16:55:29Z
- **Completed:** 2026-03-20T17:00:30Z
- **Tasks:** 3 (2 auto + 1 checkpoint auto-approved)
- **Files modified:** 10

## Accomplishments
- Background task registered via TaskManager.defineTask at module scope, 4-hour minimum interval
- Foreground drain triggers on AppState 'active' transition (fire-and-forget) + manual via GalleryScanScreen
- GalleryScanScreen with scan status, progress indicator, permission gate, auto-scan toggle, error display
- Android permissions: READ_MEDIA_IMAGES + ACCESS_MEDIA_LOCATION for gallery + EXIF GPS access
- 8 unit tests covering scheduler lifecycle (defineTask, background task, foreground drain, permission denial)

## Task Commits

Each task was committed atomically:

1. **Task 1: Background scheduler + foreground drain + store + permissions** - `eea27f9d` (feat)
2. **Task 2: GalleryScanScreen UI + navigation wiring** - `47b67a2c` (feat)
3. **Task 3: Verify gallery scan flow on emulator** - auto-approved (checkpoint)

## Files Created/Modified
- `apps/mobile/src/services/gallery/galleryScanScheduler.ts` - Background task + foreground drain trigger
- `apps/mobile/src/services/gallery/__tests__/galleryScanScheduler.test.ts` - 8 tests for scheduler
- `apps/mobile/src/store/useGalleryScanStore.ts` - Zustand store with scan state, progress, auto-scan toggle
- `apps/mobile/src/screens/GalleryScanScreen.tsx` - Gallery scan UI with manual trigger and settings
- `apps/mobile/app.json` - READ_MEDIA_IMAGES + ACCESS_MEDIA_LOCATION permissions
- `apps/mobile/App.tsx` - Side-effect import + AppState foreground drain
- `apps/mobile/src/types/index.ts` - GalleryScan added to RootStackParamList
- `apps/mobile/src/screens/index.ts` - GalleryScanScreen barrel export
- `apps/mobile/src/screens/ProfileScreen.tsx` - Gallery Scan navigation row
- `apps/mobile/src/navigation/RootNavigator.tsx` - GalleryScan route

## Decisions Made
- Used require() instead of dynamic import() in scheduler/foreground drain for Jest compatibility (avoids --experimental-vm-modules requirement)
- AppState listener fires triggerForegroundDrain on every foreground transition (silent fire-and-forget, non-blocking)
- Zustand store uses partialize to persist only scanEnabled and lastScanResult, not transient scanning state

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Jest dynamic import() requires --experimental-vm-modules; switched to require() for scheduler internals (no functional difference at runtime)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Complete gallery scanning pipeline wired end-to-end: background discovery + foreground classification + UI
- 36 total gallery tests passing (28 service layer + 8 scheduler)
- Ready for deduplication layer or gallery review UI enhancements

---
*Phase: 04-gallery-scanning-deduplication*
*Completed: 2026-03-20*
