# Roadmap: FoodTracker v1.0 (Local-First Reset)

## Overview

This roadmap delivers a fully functional local-first AI food tracker from on-device infrastructure through distribution. The critical path runs: local data foundation -> on-device detection pipeline -> nutrition resolution + diary UI -> gallery scanning -> enhanced detection + scale OCR -> sync + model delivery. Phases 1-3 produce a usable MVP (photo -> detection -> nutrition -> diary). Phase 4 adds the primary differentiator (passive gallery scanning). Phases 5-6 layer accuracy improvements, UX refinements, and cloud sync. Three prior plans (dataset acquisition 01-01, knowledge graph 01-02, VLM benchmark 01-04) are carried forward as validated work; three others (YOLO training 01-03, model export 01-05, mobile ML integration 01-06) are incorporated into Phases 2-3.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Infrastructure + Data Foundation** - Local SQLite storage, dev build workflow, bundled nutrition DB, schema migrations
- [ ] **Phase 2: On-Device Detection Pipeline** - YOLO training completion, model export, mobile ML integration, inference router (gap closure in progress)
- [ ] **Phase 2.1: Pre-trained Model Acquisition** - AIY Food V1, YOLO26n COCO, TFLite pipeline wiring
- [ ] **Phase 2.2: Deploy Custom 335-Class Classifier** - Replace AIY with trained EfficientNet-Lite0 (335 classes, 3.9MB)
- [ ] **Phase 2.3: Food-Specific YOLO Detection** - Replace COCO YOLO with GGCD YOLOv8n (241 food classes)
- [ ] **Phase 2.4: Global Cuisine Training Expansion** - Merge datasets, retrain to 700+ classes, deploy
- [ ] **Phase 2.5: Nutrition & Metadata Enrichment** - Open Food Facts, Nutrition5k, WorldCuisines multilingual labels
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
**Plans:** 6 plans (5 complete + 1 gap closure)

Plans:
- [x] 02-01-PLAN.md -- Detection types, build config (react-native-fast-tflite plugin, metro .tflite), YOLO export script
- [x] 02-02-PLAN.md -- ML service layer: YOLO post-processing (tensor decode + NMS), model loader, inference router
- [x] 02-03-PLAN.md -- Portion estimator TS port (from Python), correction store with SQLite history
- [x] 02-04-PLAN.md -- Detection store + UI components: annotated photo, bounding boxes, summary bar, list, FAB, undo toast
- [x] 02-05-PLAN.md -- Detail sheet + portion slider, DetectionScreen orchestration, navigation wiring
- [ ] 02-06-PLAN.md -- Gap closure: image preprocessing bridge + inferenceRouter TS fixes

**Carried forward work incorporated:**
- 01-03 (YOLO training scripts) -> continues as training completion within this phase
- 01-05 (model export) -> CoreML/LiteRT export pipeline
- 01-06 (mobile ML integration) -> react-native-fast-tflite integration + inference router

### Phase 02.1: Pre-trained model acquisition and TFLite integration (INSERTED)

**Goal:** Acquire pre-trained ML models (Google AIY Food V1 + YOLO26n COCO) and wire them into the existing three-stage detection pipeline, producing a testable APK with on-device inference
**Requirements**: DET-01
**Depends on:** Phase 2
**Success Criteria** (what must be TRUE):
  1. Three .tflite model files (binary gate, detection, classification) are bundled in the app
  2. Pipeline loads bundled models via require() fallback when no downloaded packs exist
  3. Binary gate correctly interprets AIY Food V1 multi-class output (max-confidence approach)
  4. Detection uses YOLO26n COCO with proper 80-class names and food-class filtering
  5. APK builds successfully and detection pipeline runs end-to-end on a real device
**Plans:** 2/3 plans executed

Plans:
- [ ] 02.1-01-PLAN.md -- Python acquisition script: download AIY Food V1, export YOLO26n to TFLite, validate, copy to assets
- [ ] 02.1-02-PLAN.md -- TypeScript pipeline wiring: bundled model fallback, binary gate fix, COCO constants, dual input sizes
- [ ] 02.1-03-PLAN.md -- EAS build config, APK build, on-device human verification of detection pipeline

### Phase 2.2: Deploy Custom 335-Class Classifier (INSERTED)

**Goal:** Replace AIY Food V1 (2024 generic classes) with our trained EfficientNet-Lite0 (335 food-specific classes, 3.9MB INT8) in the app, producing a testable APK with dramatically improved food classification
**Requirements**: ML-01
**Depends on:** Phase 2.1
**Success Criteria** (what must be TRUE):
  1. EfficientNet-Lite0 INT8 TFLite (3.9MB) replaces AIY Food V1 as the primary classifier in the bundled model set
  2. Preprocessing updated from 192x192 uint8 to 224x224 float32 with ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  3. Labels file updated to 335 merged classes (Food-101 + UEC-256) with correct index mapping
  4. Food-101 fallback classifier removed (no longer needed — its 101 classes are a subset of the new 335)
  5. APK builds and classifier correctly identifies ramen, pad thai, bibimbap, and other previously-misclassified foods on-device
**Plans:** 2 plans

Plans:
- [ ] 02.2-01-PLAN.md -- Deploy model assets and update type contracts: swap classify.tflite, deploy 335-class labels, simplify ModelSet to 2-stage
- [ ] 02.2-02-PLAN.md -- Pipeline wiring: rewrite inferenceRouter as 2-stage (detect+classify), add ImageNet normalization, update tests

### Phase 2.3: Food-Specific YOLO Detection (INSERTED)

**Goal:** Replace COCO YOLO11n (10 food classes out of 80) with GGCD YOLOv8n (241 food-specific classes) for dramatically better multi-dish detection and separation
**Requirements**: ML-02
**Depends on:** Phase 2.2
**Success Criteria** (what must be TRUE):
  1. GGCD YOLOv8n (6.7MB) is converted to TFLite INT8 format suitable for react-native-fast-tflite
  2. Detection pipeline uses food-specific YOLO with 241 class labels instead of COCO 80-class with food filtering
  3. Post-processing (NMS, bounding box decode) updated for YOLOv8 output format if different from YOLO11n
  4. Multi-dish photos (e.g., rice + curry + salad) produce separate bounding boxes per dish instead of one large box
  5. APK builds and detection correctly separates multiple food items in test photos on-device
**Plans:** 2 plans

Plans:
- [ ] 02.3-01-PLAN.md -- Export GGCD YOLOv8n to TFLite, deploy model and labels, update manifest and constants
- [ ] 02.3-02-PLAN.md -- Pipeline wiring: remove COCO filtering, use 241 GGCD food classes, per-box YOLO labels, update tests

### Phase 2.4: Global Cuisine Training Expansion (INSERTED)

**Goal:** Expand the food classifier from 335 to 700+ classes covering all major global cuisines by merging additional datasets, retraining, and deploying
**Requirements**: ML-03
**Depends on:** Phase 2.2
**Success Criteria** (what must be TRUE):
  1. Additional datasets downloaded and merged: Indian (bharat-raghunathan 15 classes, rajistics 20 classes, Khana 80 classes), Chinese (CNFOOD-241), Ethiopian (Tinsae 11 classes), Thai (THFOOD-50), Vietnamese (VietFood67), Taiwanese (Taiwanese Food 101), and others from HuggingFace/Kaggle research
  2. EfficientNet-Lite0 retrained on merged dataset with 700+ unique classes, achieving >70% top-1 validation accuracy
  3. New model exported to INT8 TFLite (<6MB) and deployed to app replacing the 335-class model
  4. Labels and nutrition mappings updated for all new classes
  5. Previously-uncovered cuisines (Indian beyond samosa/curry, African, regional Chinese) are correctly classified on-device
**Plans:** 3 plans

Plans:
- [ ] 02.4-01-PLAN.md -- Dataset acquisition and merge: download 14 datasets (CNFOOD-241, Khana, THFOOD-50, etc.), deduplicate classes, create merged_v2 with 700+ classes
- [ ] 02.4-02-PLAN.md -- Retrain and export: EfficientNet-Lite0 on merged_v2, 30 epochs on RX 7900 XT, export INT8 TFLite
- [ ] 02.4-03-PLAN.md -- Deploy to app: swap classify.tflite and labels, update manifest and constants, verify tests pass

### Phase 2.5: Nutrition & Metadata Enrichment (INSERTED)

**Goal:** Enrich the nutrition database with Open Food Facts (4.4M products), Nutrition5k calorie data, and WorldCuisines multilingual metadata so every classified food returns accurate nutrition info and cultural context
**Requirements**: ML-04, ML-05
**Depends on:** Phase 2.4, Phase 1
**Success Criteria** (what must be TRUE):
  1. Open Food Facts product database filtered and imported for food nutrition lookup (packaged foods with barcodes + generic food items)
  2. Nutrition5k per-dish calorie/macro data integrated for ground-truth nutrition estimates on common dishes
  3. WorldCuisines food-kb used to add multilingual labels (30+ languages), cuisine tags, and country associations to classified foods
  4. Every class in the 700+ classifier has a nutrition mapping (direct match, category fallback, or USDA proxy)
  5. App displays cuisine context (e.g., "Thai", "Ethiopian") alongside food names in the detection results UI
**Plans:** 3 plans

Plans:
- [ ] 02.5-01-PLAN.md -- OFF + Nutrition5k build pipelines: CSV/HuggingFace-to-SQLite matching nutrition pack schema
- [ ] 02.5-02-PLAN.md -- WorldCuisines metadata pipeline: multilingual labels, cuisine tags, country associations
- [ ] 02.5-03-PLAN.md -- Class-to-nutrition mapping + app integration: ClassNutritionMapper service, cuisine context in detection results

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
Phases execute in numeric order: 1 -> 2 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure + Data Foundation | 4/4 | Complete | 2026-03-12 |
| 2. On-Device Detection Pipeline | 5/6 | Gap closure | - |
| 2.1. Pre-trained Model Acquisition | 2/3 | In progress | - |
| 2.2. Deploy Custom 335-Class Classifier | 0/2 | Not started | - |
| 2.3. Food-Specific YOLO Detection | 0/2 | Not started | - |
| 2.4. Global Cuisine Training Expansion | 0/3 | Not started | - |
| 2.5. Nutrition & Metadata Enrichment | 0/3 | Not started | - |
| 3. Nutrition Resolution + Diary | 0/3 | Not started | - |
| 4. Gallery Scanning + Deduplication | 0/2 | Not started | - |
| 5. Enhanced Detection + Scale OCR | 0/3 | Not started | - |
| 6. Sync + Distribution | 0/2 | Not started | - |
