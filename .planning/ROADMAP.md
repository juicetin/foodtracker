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
- [x] **Phase 2.3: Food-Specific YOLO Detection** - Replace COCO YOLO with GGCD YOLOv8n (241 food classes)
- [x] **Phase 2.4: Global Cuisine Training Expansion** - Merge datasets, retrain to 700+ classes, deploy
- [x] **Phase 2.5: Food Knowledge Graph** - Recipe-based nutrition decomposition, dish taxonomy, multilingual aliases, SymSpell fuzzy search, KG-to-detection bridge
- [ ] **Phase 2.6: On-Device VLM Integration** - SmolVLM via llama.rn, progressive YOLO->VLM refinement, text+image fusion, KG-grounded nutrition
- [ ] **Phase 3: Nutrition Resolution + Diary** - Ingredient-to-nutrient lookup, portion estimation, diary UI, manual search, meal editing, recipes
- [ ] **Phase 4: Gallery Scanning + Deduplication** - Photo discovery, EXIF extraction, temporal clustering, batch processing within platform constraints
- [ ] **Phase 5: Scale OCR + Notifications + Health Data** - Kitchen scale reading, container weights, daily macro notifications, Apple Health/Google Fit
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
  4. Food-101 fallback classifier removed (no longer needed -- its 101 classes are a subset of the new 335)
  5. APK builds and classifier correctly identifies ramen, pad thai, bibimbap, and other previously-misclassified foods on-device
**Plans:** 2 plans

Plans:
- [x] 02.2-01-PLAN.md -- Deploy model assets and update type contracts: swap classify.tflite, deploy 335-class labels, simplify ModelSet to 2-stage
- [x] 02.2-02-PLAN.md -- Pipeline wiring: rewrite inferenceRouter as 2-stage (detect+classify), add ImageNet normalization, update tests

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
**Plans:** 3 plans

Plans:
- [x] 02.3-01-PLAN.md -- Export GGCD YOLOv8n to TFLite, deploy model and labels, update manifest and constants
- [x] 02.3-02-PLAN.md -- Pipeline wiring: remove COCO filtering, use 241 GGCD food classes, per-box YOLO labels, update tests
- [x] 02.3-03-PLAN.md -- APK build and emulator smoke-test of multi-dish detection

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
- [x] 02.4-01-PLAN.md -- Dataset acquisition and merge: download 14 datasets (CNFOOD-241, Khana, THFOOD-50, etc.), deduplicate classes, create merged_v2 with 700+ classes
- [x] 02.4-02-PLAN.md -- Retrain and export: EfficientNet-Lite0 on merged_v2, 30 epochs on RX 7900 XT, export INT8 TFLite
- [x] 02.4-03-PLAN.md -- Deploy to app: swap classify.tflite and labels, update manifest and constants, verify tests pass

### Phase 2.5: Food Knowledge Graph (INSERTED -- replaces prior Nutrition & Metadata Enrichment)

**Goal:** Build a compact on-device food knowledge graph (SQLite, <70MB) that maps dishes to canonical recipes with ingredient-level USDA nutrition, enabling recipe-based nutrition decomposition to replace the hardcoded 1.5 kcal/g proxy in the detection pipeline
**Requirements**: ML-04, ML-05, NUT-01
**Depends on:** Phase 2.4, Phase 1
**Success Criteria** (what must be TRUE):
  1. Knowledge graph schema implemented: cuisine -> dish_category -> dish -> dish_alias -> recipe -> recipe_ingredient -> usda_food, with FTS5 indexes on dish names and aliases
  2. KG seeded with 5K-10K dishes across 50+ cuisines from RecipeDB/RecipeNLG/curated data, each with canonical recipe(s) linking ingredients to USDA nutrition entries
  3. Multilingual dish aliases (transliterations, colloquial names, translations) populated from WorldCuisines food-kb (30+ languages)
  4. TypeScript KnowledgeGraphService on mobile provides: searchDish (FTS5 + SymSpell fuzzy matching), getCanonicalRecipe, calculateDishNutrition (recipe -> per-ingredient USDA lookup -> aggregated macros)
  5. Detection pipeline uses KG-derived nutrition instead of hardcoded proxies: YOLO className -> KG dish lookup -> recipe decomposition -> portion-scaled macros
  6. KG exported as a downloadable pack via PackManager (new 'knowledge-graph' pack type) or bundled in APK if under 50MB
**Plans:** 6 plans (3 complete + 3 gap closure)

Plans:
- [x] 02.5-01-PLAN.md -- Python KG pipeline: hierarchical schema, RecipeNLG + generate_dishes seeding, USDA SR Legacy embedding, SymSpell pre-computation
- [x] 02.5-02-PLAN.md -- TypeScript KnowledgeGraphService + SymSpellIndex: FTS5 + fuzzy search, recipe decomposition, pack type extension
- [x] 02.5-03-PLAN.md -- Detection pipeline wiring: KG nutrition replaces flat-rate proxy, three-tier fallback chain, human verification
- [ ] 02.5-04-PLAN.md -- Gap closure: ingredient quantity parsing + full dataset loading (all 4 parquet files, no dish cap)
- [ ] 02.5-05-PLAN.md -- Gap closure: full USDA SR Legacy (7,793 foods), Levenshtein fuzzy matching, classifier/detector label coverage, micronutrient schema
- [ ] 02.5-06-PLAN.md -- Gap closure: full KG rebuild, export to mobile, verify data quality metrics

### Phase 2.6: On-Device VLM Integration (INSERTED)

**Goal:** Integrate an on-device vision-language model (SmolVLM family via llama.rn) that refines YOLO food identification using multimodal image+text understanding, with optional free-form text input, producing structured food identification grounded in the knowledge graph
**Requirements**: DET-02
**Depends on:** Phase 2.5
**Success Criteria** (what must be TRUE):
  1. llama.rn integrated with Expo dev client build; VlmService loads SmolVLM GGUF + mmproj and produces grammar-constrained JSON food identifications from image + optional text input
  2. Device RAM detected via expo-device; appropriate VLM tier auto-selected: SmolVLM-256M (4GB devices, ~280MB), SmolVLM-500M (6GB, ~500MB), SmolVLM2-2.2B Q4 (8GB+, ~1.2GB)
  3. PackManager upgraded for large file streaming downloads (no OOM on 300MB+ files), paired file support (model + mmproj GGUF), and 'vlm' pack type with download progress UI
  4. Progressive refinement pipeline: YOLO gives instant bounding boxes (50-80ms) -> VLM refines identification asynchronously (1-3s) -> labels update in-place with animation; UI shows "Refining..." badge during VLM processing
  5. Optional "Describe your meal" text input on DetectionScreen; user text injected into VLM prompt alongside image for text-guided disambiguation (e.g., "massaman" + curry photo -> massaman curry)
  6. VLM output (dish names, ingredients, modifiers) fed into KG for recipe-based nutrition lookup; fallback chain: VLM+KG -> YOLO+KG -> YOLO+flat-rate proxy
**Plans:** 6 plans

Plans:
- [ ] 02.6-01-PLAN.md -- Foundation: install llama.rn + expo-device, VLM type contracts, RAM-based tier selection, Jest mock
- [ ] 02.6-02-PLAN.md -- PackManager streaming overhaul: large file downloads, VLM paired files (model + mmproj), DB schema extension
- [ ] 02.6-03-PLAN.md -- VlmService singleton: llama.rn lifecycle management, prompt engineering, grammar-constrained output
- [ ] 02.6-04-PLAN.md -- Detection types + store extension: VLM refinement fields (vlmLabel, isRefining), store actions
- [ ] 02.6-05-PLAN.md -- Progressive refinement pipeline: YOLO->VLM->KG wiring, VLM-to-YOLO matching, DetectionScreen UI (text input, refining badge)
- [ ] 02.6-06-PLAN.md -- VLM download screen with tier auto-selection, end-to-end on-device verification

### Phase 3: Nutrition Resolution + Diary
**Goal**: Users can view detected food as actionable nutrition data in a daily diary, with full manual editing and recipe management
**Depends on**: Phase 2.5, Phase 2.6
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

### Phase 5: Scale OCR + Notifications + Health Data
**Goal**: Users get precise portion weights via kitchen scale OCR, daily macro summaries via push notifications, and weight trend tracking via health platform integration
**Depends on**: Phase 2.6, Phase 4
**Requirements**: DET-04, SCL-01, SCL-02, SCL-03, NTF-01, NTF-02
**Success Criteria** (what must be TRUE):
  1. When a kitchen scale is visible in a food photo, the app reads the displayed weight via 7-segment OCR and user can manage container tare weights (save, auto-subtract, and the app learns frequently used containers over time)
  2. User receives a configurable end-of-day push notification summarizing daily macros, which can also serve as a trigger to bring the app to foreground for gallery processing
  3. User can import weight data from Apple Health / Google Fit and view a smoothed weight trend
  4. On supported devices (Pixel 8+, Galaxy S24+) Gemini Nano provides opportunistic inference enhancement via AICore
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
Phases execute in numeric order: 1 -> 2 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.6 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure + Data Foundation | 4/4 | Complete | 2026-03-12 |
| 2. On-Device Detection Pipeline | 5/6 | Gap closure | - |
| 2.1. Pre-trained Model Acquisition | 2/3 | In progress | - |
| 2.2. Deploy Custom 335-Class Classifier | 2/2 | Complete | 2026-03-13 |
| 2.3. Food-Specific YOLO Detection | 3/3 | Complete | 2026-03-14 |
| 2.4. Global Cuisine Training Expansion | 3/3 | Complete | 2026-03-14 |
| 2.5. Food Knowledge Graph | 6/6 | Complete | 2026-03-14 |
| 2.6. On-Device VLM Integration | 0/6 | Not started | - |
| 3. Nutrition Resolution + Diary | 0/3 | Not started | - |
| 4. Gallery Scanning + Deduplication | 0/2 | Not started | - |
| 5. Scale OCR + Notifications + Health Data | 0/3 | Not started | - |
| 6. Sync + Distribution | 0/2 | Not started | - |
