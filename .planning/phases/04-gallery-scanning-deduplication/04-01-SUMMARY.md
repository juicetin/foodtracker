---
phase: 04-gallery-scanning-deduplication
plan: 01
subsystem: gallery-scanning
tags: [expo-media-library, gemini-nano, haversine, exif, meal-grouping, image-downscale]

# Dependency graph
requires:
  - phase: 02.7-gemini-nano
    provides: geminiNanoService.identify() for food classification
provides:
  - Gallery scan service layer (discovery, classification, meal grouping, photo import)
  - Type contracts for gallery scanning pipeline (ScanQueueItem, ClassifiedPhoto, MealGroup)
  - Extended scan_queue schema with EXIF + classification columns
affects: [04-02-gallery-ui, gallery-scheduling, deduplication]

# Tech tracking
tech-stack:
  added: []
  patterns: [cursor-pagination, insert-or-ignore-dedup, haversine-clustering, 500ms-pacing]

key-files:
  created:
    - apps/mobile/src/services/gallery/types.ts
    - apps/mobile/src/services/gallery/galleryScanService.ts
    - apps/mobile/src/services/gallery/foodClassifier.ts
    - apps/mobile/src/services/gallery/mealGrouper.ts
    - apps/mobile/src/services/gallery/photoImporter.ts
    - apps/mobile/src/services/gallery/index.ts
    - apps/mobile/src/services/gallery/__tests__/mealGrouper.test.ts
    - apps/mobile/src/services/gallery/__tests__/photoImporter.test.ts
    - apps/mobile/src/services/gallery/__tests__/galleryScanService.test.ts
    - apps/mobile/src/services/gallery/__tests__/foodClassifier.test.ts
  modified:
    - apps/mobile/db/schema.ts
    - apps/mobile/db/client.ts

key-decisions:
  - "opsqlite raw SQL (not drizzle userDb) for gallery scan service -- consistent with historyService/backupService pattern"
  - "haversine test expectation corrected to actual computed distance for Sydney coords (1665m not 1860m)"
  - "importPhoto takes explicit dimensions param rather than reading from asset metadata -- avoids extra async call"

patterns-established:
  - "Gallery service barrel index pattern: all public APIs re-exported from gallery/index.ts"
  - "500ms pacing between Gemini Nano classify calls to avoid AICore BUSY"
  - "INSERT OR IGNORE on asset_id UNIQUE for idempotent gallery discovery"

requirements-completed: [GAL-01, GAL-03, GAL-04, GAL-05]

# Metrics
duration: 6min
completed: 2026-03-20
---

# Phase 04 Plan 01: Gallery Scan Service Layer Summary

**Four gallery scanning services (discovery, food classification, meal grouping, photo import) with cursor pagination, Gemini Nano 500ms pacing, haversine clustering, and 1024px downscale -- 28 tests passing**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-20T16:46:49Z
- **Completed:** 2026-03-20T16:53:00Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments
- Extended scan_queue schema with creationTime, lat/lon, isFood, mealGroupId + UNIQUE on assetId
- Built mealGrouper with haversine distance + 1hr/150m clustering (time-only fallback when GPS missing)
- Built photoImporter with 1024px longest-edge downscale and persistent app storage
- Built galleryScanService with cursor pagination via getAssetsAsync + INSERT OR IGNORE dedup
- Built foodClassifier wrapping Gemini Nano with 500ms pacing and exponential backoff
- drainScanQueue orchestrates full classify -> group -> import pipeline
- 28 unit tests covering all behavior cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Schema extension + type contracts + mealGrouper + photoImporter** - `4cfea997` (feat)
2. **Task 2: galleryScanService + foodClassifier + barrel index** - `a251d32c` (feat)

## Files Created/Modified
- `apps/mobile/db/schema.ts` - Extended scan_queue with EXIF + classification columns
- `apps/mobile/db/client.ts` - Added CREATE TABLE IF NOT EXISTS + idempotent ALTER TABLE for scan_queue
- `apps/mobile/src/services/gallery/types.ts` - ScanQueueItem, ClassifiedPhoto, MealGroup, GalleryScanPreferences, DEFAULT_SCAN_PREFS
- `apps/mobile/src/services/gallery/mealGrouper.ts` - haversineDistance + groupIntoMeals (time+GPS clustering)
- `apps/mobile/src/services/gallery/photoImporter.ts` - importPhoto (1024px downscale, JPEG 0.8, persistent storage)
- `apps/mobile/src/services/gallery/galleryScanService.ts` - discoverNewPhotos, getPendingScanItems, markScanItemDone
- `apps/mobile/src/services/gallery/foodClassifier.ts` - classifyPhoto, drainScanQueue (500ms pacing, backoff)
- `apps/mobile/src/services/gallery/index.ts` - Barrel re-export of all public APIs
- `apps/mobile/src/services/gallery/__tests__/mealGrouper.test.ts` - 10 tests for clustering
- `apps/mobile/src/services/gallery/__tests__/photoImporter.test.ts` - 3 tests for resize/import
- `apps/mobile/src/services/gallery/__tests__/galleryScanService.test.ts` - 8 tests for discovery/queue
- `apps/mobile/src/services/gallery/__tests__/foodClassifier.test.ts` - 7 tests for classification/drain

## Decisions Made
- Used opsqlite raw SQL (not drizzle userDb) for gallery scan service -- consistent with historyService and backupService patterns
- importPhoto takes explicit dimensions parameter rather than reading from asset metadata -- avoids extra async call and simplifies testing
- Haversine distance test corrected to actual computed distance for Sydney Opera House to Sydney Tower Eye coords

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 gallery service modules exist with full test coverage
- Ready for Plan 02 to wire into background scheduling and UI
- galleryScanService provides cursor pagination; foodClassifier provides foreground-only classification pipeline

---
*Phase: 04-gallery-scanning-deduplication*
*Completed: 2026-03-20*
