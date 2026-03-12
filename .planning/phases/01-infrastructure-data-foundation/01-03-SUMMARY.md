---
phase: 01-infrastructure-data-foundation
plan: 03
subsystem: database
tags: [regional-nutrition, afcd, cofid, ciqual, openpyxl, fts5, locale, multi-db, sqlite]

# Dependency graph
requires:
  - phase: 01-infrastructure-data-foundation
    provides: "NutritionService, PackManager, openNutritionDb, published nutrition pack schema"
provides:
  - "Regional build pipeline converting AFCD/CoFID/CIQUAL Excel to nutrition pack SQLite"
  - "Locale detector mapping device locale to regional pack suggestion"
  - "Multi-database RegionalResolver with priority ordering (regional > USDA > custom)"
  - "Schema validation for custom nutrition pack imports"
affects: [02-detection-pipeline, 03-nutrition-data, 06-sync-distribution]

# Tech tracking
tech-stack:
  added: ["expo-localization", "openpyxl"]
  patterns: ["multi-DB priority resolver: regional first, USDA fallback", "locale prefix matching for language families (fr-* -> ciqual)", "synthetic fdc_id generation for non-USDA sources"]

key-files:
  created:
    - "knowledge-graph/build_regional_db.py"
    - "knowledge-graph/tests/test_regional_build.py"
    - "apps/mobile/src/services/packs/localeDetector.ts"
    - "apps/mobile/src/services/packs/__tests__/localeDetector.test.ts"
    - "apps/mobile/src/services/nutrition/regionalResolver.ts"
    - "apps/mobile/src/services/nutrition/__tests__/regionalResolver.test.ts"
  modified:
    - "knowledge-graph/requirements.txt"
    - "apps/mobile/package.json"

key-decisions:
  - "Used openpyxl for Excel parsing rather than pandas -- lighter dependency for simple row iteration"
  - "CoFID and CIQUAL use synthetic sequential fdc_id (row index) since they have non-numeric food codes"
  - "Standard USDA FDC nutrient IDs (1008=Energy, 1003=Protein, etc.) used across all regional databases for cross-DB nutrient lookup compatibility"
  - "Locale prefix matching for language families: fr-CA, fr-BE, fr-CH all map to CIQUAL"
  - "RegionalResolver priority: regional packs (1) > usda-core (2) > usda-branded (3) > custom (4)"

patterns-established:
  - "Regional build pipeline: source-specific Excel loader -> common schema insert -> FTS5 index -> VACUUM"
  - "Locale detection via expo-localization getLocales() with lowercase normalization"
  - "Multi-DB resolver: each NutritionService instance keyed by pack ID, searched in priority order"
  - "ResolvedFood extends NutritionFood with source field for cross-DB routing"

requirements-completed: [DAT-03]

# Metrics
duration: 7min
completed: 2026-03-12
---

# Phase 01 Plan 03: Regional Nutrition Databases Summary

**AFCD/CoFID/CIQUAL build pipeline with locale-based auto-suggestion and multi-DB resolver prioritizing regional results over USDA**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-12T06:43:34Z
- **Completed:** 2026-03-12T06:50:37Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Built Python pipeline converting AFCD (Australia), CoFID (UK), and CIQUAL (France) Excel data to published nutrition pack SQLite schema with FTS5 search
- Created locale detector that auto-suggests regional packs based on device locale (en-AU->afcd, en-GB->cofid, fr-*->ciqual)
- Implemented RegionalResolver that manages multiple NutritionService instances with regional databases taking query priority over USDA
- Added schema validation for custom pack imports, checking table/column conformance
- All 24 new tests pass (13 Python + 11 TypeScript), 79 total TypeScript tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Regional DB build pipeline and locale detector** - `e9779140` (test, RED) then `c772120f` (feat, GREEN)
2. **Task 2: Multi-database regional resolver with priority ordering** - `61e46d9c` (test, RED) then `7a12b35d` (feat, GREEN)

_Note: TDD tasks have separate test and implementation commits._

## Files Created/Modified
- `knowledge-graph/build_regional_db.py` - Python script converting AFCD/CoFID/CIQUAL Excel to nutrition pack SQLite
- `knowledge-graph/tests/test_regional_build.py` - 13 pytest tests with Excel fixture data
- `knowledge-graph/requirements.txt` - Added openpyxl dependency
- `apps/mobile/src/services/packs/localeDetector.ts` - Device locale detection and regional pack suggestion
- `apps/mobile/src/services/packs/__tests__/localeDetector.test.ts` - 13 locale detector tests
- `apps/mobile/src/services/nutrition/regionalResolver.ts` - Multi-DB query resolver with priority ordering
- `apps/mobile/src/services/nutrition/__tests__/regionalResolver.test.ts` - 11 resolver tests
- `apps/mobile/package.json` - Added expo-localization dependency

## Decisions Made
- **openpyxl over pandas:** Used openpyxl for Excel parsing since it's lighter than pandas for simple row iteration. pandas is already in requirements.txt but not needed for this pipeline.
- **Synthetic fdc_id for non-USDA sources:** CoFID uses food codes like "C001" and CIQUAL uses "CQ001" -- these are non-numeric. The build pipeline assigns sequential integer IDs (row index) to maintain schema compatibility with the fdc_id INTEGER PRIMARY KEY.
- **Standard nutrient IDs across databases:** All regional databases use USDA FDC nutrient numbering (1008=Energy, 1003=Protein, etc.) even though the source data uses different codes. This enables cross-DB nutrient lookup without translation.
- **Locale prefix matching:** French locales use language prefix matching (fr-*) rather than exact locale matching, so fr-CA, fr-BE, and fr-CH all map to CIQUAL.
- **Priority ordering:** RegionalResolver sorts databases by type: regional (1) > usda-core (2) > usda-branded (3) > custom (4). Within each tier, databases appear in order of installation.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 data foundation is complete: all 3 plans (local DB, nutrition pipeline, regional resolver) are done
- RegionalResolver is ready for integration with food detection pipeline (Phase 2)
- Locale detector is ready for onboarding UI (Phase 3)
- Custom pack import validation is ready for Settings UI (Phase 3)
- Regional build pipeline can process real AFCD/CoFID/CIQUAL data when available

## Self-Check: PASSED

All 6 created files verified present on disk. All 4 task commits (e9779140, c772120f, 61e46d9c, 7a12b35d) verified in git log. 24 new tests passing (13 Python + 11 TypeScript), 79 total TypeScript tests passing. TypeScript compiles clean.

---
*Phase: 01-infrastructure-data-foundation*
*Completed: 2026-03-12*
