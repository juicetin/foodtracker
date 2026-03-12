# Roadmap: FoodTracker v1.0 (Local-First Reset)

## Overview

This roadmap delivers a fully functional local-first AI food tracker from on-device infrastructure through distribution. The critical path runs: local data foundation -> on-device detection pipeline -> nutrition resolution + diary UI -> gallery scanning -> enhanced detection + scale OCR -> sync + model delivery. Phases 1-3 produce a usable MVP (photo -> detection -> nutrition -> diary). Phase 4 adds the primary differentiator (passive gallery scanning). Phases 5-6 layer accuracy improvements, UX refinements, and cloud sync. Three prior plans (dataset acquisition 01-01, knowledge graph 01-02, VLM benchmark 01-04) are carried forward as validated work; three others (YOLO training 01-03, model export 01-05, mobile ML integration 01-06) are incorporated into Phases 2-3.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Infrastructure + Data Foundation** - Local SQLite storage, dev build workflow, bundled nutrition DB, schema migrations
- [ ] **Phase 2: On-Device Detection Pipeline** - YOLO training completion, model export, mobile ML integration, inference router
- [ ] **Phase 3: Nutrition Resolution + Diary** - Ingredient-to-nutrient lookup, portion estimation, diary UI, manual search, meal editing, recipes
- [ ] **Phase 4: Gallery Scanning + Deduplication** - Photo discovery, EXIF extraction, temporal clustering, batch processing within platform constraints
- [ ] **Phase 5: Enhanced Detection + Scale OCR** - VLM integration, hidden ingredient inference, scale reading, container weights, UX modes, notifications
- [ ] **Phase 6: Sync + Distribution** - Google Drive and iCloud sync, Play for On-Device AI, iOS On-Demand Resources, Gemini Nano adapter

## Phase Details

### Phase 1: Infrastructure + Data Foundation
**Goal**: All local data infrastructure is in place so every subsequent module has a reliable storage and query layer
**Depends on**: Nothing (first phase)
**Requirements**: DAT-01, DAT-02, DAT-03
**Success Criteria** (what must be TRUE):
  1. App builds and runs on both iOS and Android as custom dev builds (no Expo Go) with all native config plugins compiling together
  2. User data (food entries, recipes, preferences) persists across app restarts in local op-sqlite database with versioned schema migrations
  3. Bundled USDA FDC nutrition database is available on first launch (or fast-follow asset pack) and returns results for common food queries
  4. Optional regional nutrition databases (AFCD, CoFID, CIQUAL) can be downloaded and queried alongside USDA data
**Plans:** 4 plans

Plans:
- [ ] 01-01-PLAN.md -- Local data foundation: op-sqlite + drizzle schema, migrations, store refactor, legacy cleanup
- [ ] 01-02-PLAN.md -- USDA nutrition DB: FDC build pipeline, pack manager, nutrition query service
- [ ] 01-03-PLAN.md -- Regional DBs: AFCD/CoFID/CIQUAL build pipelines, locale detection, multi-DB resolver
- [ ] 01-04-PLAN.md -- Gap closure: importCustomPack entry point for custom nutrition pack imports

### Phase 2: On-Device Detection Pipeline
**Goal**: Users can photograph food and receive on-device identification with bounding boxes and confidence indicators
**Depends on**: Phase 1
**Requirements**: DET-01, DET-05, DET-06
**Success Criteria** (what must be TRUE):
  1. User photographs food and sees identified food items with bounding boxes drawn on the image within 2 seconds on mid-range devices
  2. Each detected item shows a confidence indicator (green/yellow/red) and the user can manually correct low-confidence results
  3. Detected items include portion size estimates based on visual cues (plate size, reference objects, density tables)
  4. Detection pipeline runs entirely on-device via CoreML (iOS) and LiteRT (Android) with no network dependency
**Plans:** 5 plans

Plans:
- [ ] 02-01-PLAN.md -- Detection types, build config (react-native-fast-tflite plugin, metro .tflite), YOLO export script
- [ ] 02-02-PLAN.md -- ML service layer: YOLO post-processing (tensor decode + NMS), model loader, inference router
- [ ] 02-03-PLAN.md -- Portion estimator TS port (from Python), correction store with SQLite history
- [ ] 02-04-PLAN.md -- Detection store + UI components: annotated photo, bounding boxes, summary bar, list, FAB, undo toast
- [ ] 02-05-PLAN.md -- Detail sheet + portion slider, DetectionScreen orchestration, navigation wiring

**Carried forward work incorporated:**
- 01-03 (YOLO training scripts) -> continues as training completion within this phase
- 01-05 (model export) -> CoreML/LiteRT export pipeline
- 01-06 (mobile ML integration) -> react-native-fast-tflite integration + inference router

### Phase 3: Nutrition Resolution + Diary
**Goal**: Users can view detected food as actionable nutrition data in a daily diary, with full manual editing and recipe management
**Depends on**: Phase 2
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07, UI-08
**Success Criteria** (what must be TRUE):
  1. User views a daily food diary organized by meal (breakfast/lunch/dinner/snacks) showing per-meal and daily macro totals
  2. User can search the bundled USDA database and manually add a food item in under 7 taps
  3. User can edit any logged meal (change ingredients, adjust portions, modify quantities) after initial logging
  4. User can save a corrected meal as a recipe, reuse it in one tap, and create nested recipes (recipes containing other recipes) with expand/collapse and edit-in-context-or-globally prompts
  5. User can view the linked photo(s) for any logged meal and switch between UX modes (zero-effort, confirm-only, guided-edit)
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

### Phase 4: Gallery Scanning + Deduplication
**Goal**: Users no longer need to manually trigger photo analysis -- the app discovers food photos from the gallery automatically
**Depends on**: Phase 2, Phase 3
**Requirements**: GAL-01, GAL-02, GAL-03, GAL-04, GAL-05
**Success Criteria** (what must be TRUE):
  1. User can manually trigger a gallery scan and see newly discovered food photos queued for processing
  2. App performs periodic background scanning that surfaces new food photos without user intervention, operating within platform constraints (iOS 30-second BGTask, Android WorkManager) using chunked processing blocks
  3. Multiple photos of the same meal (taken within 5-minute window with GPS proximity) are grouped into a single meal event instead of creating duplicates
  4. Each discovered photo displays EXIF-derived context (timestamp as meal time, location as meal venue)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Enhanced Detection + Scale OCR
**Goal**: Users get higher accuracy through VLM for complex dishes, hidden ingredient inference, and kitchen scale weight reading
**Depends on**: Phase 2, Phase 4
**Requirements**: DET-02, DET-03, DET-04, SCL-01, SCL-02, SCL-03, NTF-01, NTF-02
**Success Criteria** (what must be TRUE):
  1. Device automatically selects and downloads the appropriate VLM tier (SmolVLM-256M / Moondream 0.5B / Gemma 3n) based on device capability, and on supported devices (Pixel 8+, Galaxy S24+) Gemini Nano provides opportunistic inference
  2. User sees inferred hidden ingredients for identified dishes (e.g., "carbonara" shows egg, pancetta, parmesan) via knowledge graph lookup
  3. When a kitchen scale is visible in a food photo, the app reads the displayed weight via 7-segment OCR and user can manage container tare weights (save, auto-subtract, and the app learns frequently used containers over time)
  4. User receives a configurable end-of-day push notification summarizing daily macros, which can also serve as a trigger to bring the app to foreground for gallery processing
  5. User can import weight data from Apple Health / Google Fit and view a smoothed weight trend
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD

### Phase 6: Sync + Distribution
**Goal**: Users can back up data to the cloud and receive ML models through platform-optimized delivery channels
**Depends on**: Phase 1, Phase 5
**Requirements**: DAT-04, DAT-05, DAT-06, MDL-01, MDL-02
**Success Criteria** (what must be TRUE):
  1. User can opt into Google Drive backup/sync via app data folder, with data accessible cross-platform (iOS and Android)
  2. User on iOS can opt into iCloud backup/sync as an alternative to Google Drive
  3. Sync conflicts between devices are resolved via last-write-wins with timestamps, and full edit history is retained locally
  4. Android app delivers ML models via Play for On-Device AI with device targeting by RAM and chipset; iOS app delivers optional models via On-Demand Resources or Background Assets API
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure + Data Foundation | 4/4 | Complete | 2026-03-12 |
| 2. On-Device Detection Pipeline | 0/5 | Not started | - |
| 3. Nutrition Resolution + Diary | 0/3 | Not started | - |
| 4. Gallery Scanning + Deduplication | 0/2 | Not started | - |
| 5. Enhanced Detection + Scale OCR | 0/3 | Not started | - |
| 6. Sync + Distribution | 0/2 | Not started | - |
