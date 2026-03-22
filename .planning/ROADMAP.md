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
- [x] **Phase 02.7: Gemini Nano System-Managed VLM Integration** - Tier 0 Gemini Nano via ML Kit GenAI Prompt API, quality spike + human gate, pipeline integration (completed 2026-03-20)
- [ ] **Phase 3.1: Daily Diary View + Macro Dashboard** - Chronological entry list, photo thumbnails, always-visible macro summary, entry expand/detail
- [ ] **Phase 3.2: Food Search + Manual Add + Quick Add** - KG+OFF search, personal history ranking, quick cal/macro entry, persistent search bar
- [ ] **Phase 3.3: Meal Editing + Portion Adjustment** - Post-logging ingredient editing, serving size selector, portion adjustment, re-run VLM
- [ ] **Phase 3.4: Recipe Management** - Save meal as recipe, recipe list/search, 1-tap re-log, edit recipe ingredients
- [x] **Phase 3.5: OFF Cache + Attribution** - SQLite cache for OFF API responses, stale-while-revalidate, offline fallback, adaptive rate limiting, ODbL attribution (completed 2026-03-19)
- [ ] **Phase 3.6: Incremental Backup System** - updateHook change journal, JSON changeset export, VACUUM INTO full backup, compaction, includes OFF cache + all user data
- [ ] **Phase 3.7: Google Drive Sync** - react-native-cloud-storage, Google OAuth, upload/download backups, auto-sync on background, restore on fresh install
- [ ] **Phase 4: Gallery Scanning + Deduplication** - Photo discovery, EXIF extraction, temporal clustering, batch processing within platform constraints
- [ ] **Phase 5: Scale OCR + Notifications + Health Data** - Kitchen scale reading, container weights, daily macro notifications, Apple Health/Google Fit
- [ ] **Phase 6: Sync + Distribution** - Google Drive and iCloud sync, Play for On-Device AI, iOS On-Demand Resources, Gemini Nano adapter
- [x] **Phase 7: Remove YOLO+EfficientNet pipeline -- VLM-only detection** - Strip EfficientNet classifier, YOLO bbox-only, shimmer UX, VLM failure fallback (completed 2026-03-15)
- [ ] **Phase 8: On-device vector search embedding via TFLite MiniLM** - MiniLM-L6-v2 TFLite export, pure-JS WordPiece tokenizer, semantic USDA food matching

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

### Phase 02.7: Gemini Nano System-Managed VLM Integration (INSERTED)

**Goal:** Integrate Gemini Nano (system-managed, via ML Kit GenAI Prompt API) as Tier 0 above SmolVLM in the food identification pipeline. Wave 1 is a quality spike — a minimal native module + test screen to evaluate Gemini Nano output on a Pixel 9 Pro before committing to full integration. Wave 2 (gated on spike quality) wires GeminiNanoService into runVlmIdentification() as the primary path, with SmolVLM as fallback on unsupported devices.
**Requirements**: DET-03
**Depends on:** Phase 02.6
**Success Criteria** (what must be TRUE):
  1. Native Expo module wraps ML Kit GenAI Prompt API with checkAvailability() and identifyFood(uri, prompt) — works on Pixel 9 Pro, returns graceful unavailable on Pixel 7 Pro
  2. GeminiNanoTestScreen (debug/settings) shows Gemini Nano food identification output side-by-side comparison for quality evaluation
  3. Human checkpoint passed: Gemini Nano quality judged sufficient on Pixel 9 Pro before Wave 2 proceeds
  4. GeminiNanoService produces VlmFoodResult (same shape as vlmService) from Prompt API JSON output
  5. runVlmIdentification() checks Gemini Nano availability first; falls back to SmolVLM if unavailable or fails; Pixel 7 Pro experience unchanged
  6. No model download required on Pixel 9 Pro / Galaxy S25+ — AICore provides the model system-managed
**Plans:** 2/2 plans complete

Plans:
- [ ] 02.7-01-PLAN.md -- Wave 1 spike: GeminiNano native module (Kotlin + TS bindings), GeminiNanoTestScreen, navigation wiring, unit tests, human checkpoint on Pixel 9 Pro
- [ ] 02.7-02-PLAN.md -- Wave 2 integration (GATED): wire geminiNanoService into vlmPipeline.ts as Tier 0, unit tests, APK verify

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
**Plans:** 3/3 plans complete

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

### Phase 7: Remove YOLO and EfficientNet pipeline entirely -- VLM-only detection

**Goal:** Strip the EfficientNet classification stage entirely and reduce YOLO to bounding-box-only duty, making VLM the sole source of food identification with shimmer UX during processing and graceful text fallback on VLM failure
**Requirements**: P7-01, P7-02, P7-03, P7-04, P7-05, P7-06, P7-07, P7-08, P7-09, P7-10, P7-11
**Depends on:** Phase 2.6
**Success Criteria** (what must be TRUE):
  1. EfficientNet classify.tflite (4.9MB), labels_classify.json, and all classification training scripts are deleted from the repo
  2. YOLO outputs bounding boxes only -- every detection is a generic "Food Region" until VLM identifies it
  3. inferenceRouter is single-stage (bbox-only), modelLoader loads detect-only, no ImageNet normalization in preprocessing
  4. VLM is the primary food identifier (not a refinement step), with one silent retry on failure
  5. Shimmer/skeleton animation appears in bounding box labels and detection list items while VLM processes
  6. When VLM fails, user sees "Describe your meal" text input; typed dish names are assigned to boxes by size order with KG nutrition lookup
**Plans:** 3/3 plans complete

Plans:
- [x] 07-01-PLAN.md -- EfficientNet removal + pipeline simplification: delete assets/scripts, simplify types/constants/modelLoader/inferenceRouter to bbox-only
- [x] 07-02-PLAN.md -- VLM pipeline rewrite: primary identification with retry, text fallback with box-size assignment, store displayLabel update
- [x] 07-03-PLAN.md -- Shimmer UX + DetectionScreen rewrite: ShimmerPlaceholder component, bbox/list shimmer, VLM-primary flow, text fallback UI

### Phase 3.1: Daily Diary View + Macro Dashboard (INSERTED -- replaces old Phase 3)
**Goal**: Users see a daily food diary with chronological entries, photo thumbnails, and an always-visible macro summary -- the core screen they live in every day
**Depends on**: Phase 2.6, Phase 7, Phase 3.6
**Requirements**: UI-01
**Success Criteria** (what must be TRUE):
  1. User views a daily food diary with entries grouped by time period (morning/afternoon/evening) showing per-period and daily macro totals (Cal/P/F/C)
  2. Each entry card shows photo thumbnail (our differentiator), food name(s) from VLM identification, Cal and P/F/C values, and time logged
  3. Always-visible daily macro summary header pinned to top of diary (consumed values, with Consumed/Remaining toggle)
  4. User can tap an entry to expand detail view, navigate to edit, or delete
  5. Date navigation (previous/next day) and week overview bar showing which days have entries
  6. Entries logged via DetectionScreen appear immediately in the diary with correct time and photo association
**Plans**: 2 plans

Plans:
- [ ] 03.1-01-PLAN.md -- Service layer + diary components: time-period logic, diary queries, preferences extension, StickyMacroHeader, WeekOverviewBar, TimePeriodSection, ExpandableEntryCard
- [ ] 03.1-02-PLAN.md -- DiaryScreen refactor: wire components, swipe navigation, time-period grouping, human verification

### Phase 3.2: Food Search + Manual Add + Quick Add (INSERTED)
**Goal**: Users can add food without using the camera -- via text search across KG + OFF, personal food history, or raw calorie/macro entry as an escape hatch
**Depends on**: Phase 3.1
**Requirements**: UI-02, UI-06
**Success Criteria** (what must be TRUE):
  1. Persistent search bar visible on diary view -- adding food is always 1 tap away
  2. Search queries KG dishes + OFF cache simultaneously, returning results with name, Cal, P/F/C, and serving description
  3. "From History" results appear first, ranked by personal logging frequency (foods user has eaten before)
  4. Quick Add screen allows entering just Cal + P/F/C values directly (escape hatch when AI fails or DB doesn't have the food)
  5. User can add a food item from search results to the diary in under 5 taps (search -> select -> confirm serving -> logged)
  6. Barcode scanner icon visible in search bar (stub -- wired in v1.1)
**Plans**: 2 plans

Plans:
- [ ] 03.2-01-PLAN.md -- History service, macro validation, QuickAdd screen, SearchBar component, navigation wiring
- [ ] 03.2-02-PLAN.md -- DiaryScreen search bar integration, FoodSearchScreen history+QuickAdd enhancement, human verification

### Phase 3.3: Meal Editing + Portion Adjustment (INSERTED)
**Goal**: Users can correct any logged meal after the fact -- change ingredients, swap items, adjust portions, or re-run VLM identification
**Depends on**: Phase 3.1
**Requirements**: UI-03, UI-07
**Success Criteria** (what must be TRUE):
  1. User can tap any diary entry to open a full edit view showing all ingredients with individual nutrition values
  2. Each ingredient has a serving size selector with options from KG (grams, cups, portions, "1 serving") and free-form gram input
  3. User can add, remove, or replace individual ingredients within a logged meal
  4. Editing an ingredient's portion recalculates the entry's total nutrition in real-time
  5. User can trigger VLM re-identification on a meal's photo(s) to get updated food names
  6. All edits are persisted to SQLite and reflected immediately in the diary view and daily macro summary
**Plans**: 2 plans

Plans:
- [ ] 03.3-01-PLAN.md -- Edit infrastructure: editSessionManager (command pattern undo/redo), ServingSizeSelector, IngredientSearchSheet, PhotoViewer, EntryDetailScreen wiring
- [ ] 03.3-02-PLAN.md -- Re-identification: reidentifyService (Gemini Nano re-scan + KG enrichment), ReidentifyMergeScreen (drag-and-drop diff/merge), human verification

### Phase 3.4: Recipe Management (INSERTED)
**Goal**: Users can save any logged meal as a reusable recipe and re-log it in 1-2 taps, building a personal recipe library over time
**Depends on**: Phase 3.3
**Requirements**: UI-04, UI-08
**Success Criteria** (what must be TRUE):
  1. User can save any logged meal as a named recipe with one tap ("Save as Recipe" action on entry detail)
  2. Recipe list screen shows all saved recipes with name, photo, Cal/P/F/C per serving, and times used
  3. User can re-log a saved recipe to the diary in 1-2 taps (recipe list -> confirm -> logged)
  4. User can edit recipe ingredients and portions (changes apply to the recipe template, not past entries)
  5. Recipe search integrated into the food search flow (Phase 3.2) -- recipes appear alongside KG/OFF results
  6. Existing custom_recipes and recipeIngredients tables are used (no new schema required)
**Plans:** 2 plans

Plans:
- [ ] 03.4-01-PLAN.md -- Service layer: schema migrations (servings, source_recipe_id), saveEntryAsRecipe, searchRecipes, updateRecipeWithVersioning, Gemini Nano recipe_name prompt, UX mode preference
- [ ] 03.4-02-PLAN.md -- UI wiring: Save as Recipe on EntryDetail, enhanced RecipeScreen (photo/macros/search/versioning), recipe search integration, UX mode selector, human verification

### Phase 3.5: OFF Cache + Attribution (INSERTED)
**Goal**: OFF API responses cached locally in SQLite for offline use, with adaptive rate limiting and proper ODbL attribution
**Depends on**: Phase 3
**Success Criteria** (what must be TRUE):
  1. OFF barcode lookups and text search results are cached in a local SQLite table with stale-while-revalidate pattern (serve cache first, refresh from network)
  2. When offline, cached OFF results are returned -- user sees previously looked-up foods without internet
  3. Adaptive rate limiter respects OFF limits (100 product/min, 10 search/min) with variable debounce that ramps up as rolling window approaches capacity
  4. ODbL attribution link displayed in Profile/About screen crediting Open Food Facts
  5. OFF cache is included in data export (CSV/JSON) and backup system
**Plans:** 2 plans

Plans:
- [ ] 03.5-01-PLAN.md -- SQLite cache layer: two cache tables, offCacheService, stale-while-revalidate wrapping of OFF API
- [ ] 03.5-02-PLAN.md -- Export integration + ODbL attribution: OFF cache in CSV/JSON exports, About section in ProfileScreen

### Phase 3.6: Incremental Backup System (INSERTED)
**Goal**: Users can back up all data (entries, recipes, favourites, OFF cache, settings) with incremental diffs and periodic full snapshots
**Depends on**: Phase 3.5
**Success Criteria** (what must be TRUE):
  1. op-sqlite updateHook writes to a `_change_journal` table recording table, rowid, operation, and timestamp for every INSERT/UPDATE/DELETE
  2. Manual or automatic backup trigger exports journal entries as a JSON changeset file (incremental diff since last backup)
  3. Periodic full backup via VACUUM INTO creates a clean, portable .db snapshot
  4. Compaction tool replays incremental JSON diffs onto the last full backup to produce a merged full backup
  5. Backup includes ALL user data: food entries, ingredients, dishes, photos, recipes, favourites, OFF cache, preferences
  6. Backup files stored on local device storage accessible via Files app
**Requirements**: BKP-01, BKP-02, BKP-03, BKP-04, BKP-05, BKP-06
**Plans:** 2 plans

Plans:
- [ ] 03.6-01-PLAN.md -- Change journal, backup types, backupService (incremental JSON, full VACUUM INTO, compaction, retention)
- [ ] 03.6-02-PLAN.md -- Background auto-backup scheduler, ProfileScreen backup card, iOS file visibility config

### Phase 3.7: Google Drive Sync (INSERTED)
**Goal**: Users can sync backups to Google Drive for cross-device restore and cloud safety
**Depends on**: Phase 3.6
**Success Criteria** (what must be TRUE):
  1. Google OAuth sign-in flow integrated via react-native-cloud-storage or equivalent
  2. Incremental backups auto-upload to Google Drive on app background
  3. Manual full backup upload available from Profile screen
  4. Fresh app install can discover and restore from Google Drive backup
  5. Restore applies full backup + incremental diffs in order to reconstruct complete database
**Requirements**: DAT-04, DAT-06
**Plans:** 3 plans

Plans:
- [x] 03.7-01-PLAN.md -- Service layer: install deps, Expo config plugin, sync types, driveAuth, driveSync, conflictResolver, useSyncStore + tests
- [x] 03.7-02-PLAN.md -- Scheduler + UI: syncScheduler Drive upload, restoreService, SyncSettingsScreen, ProfileScreen sync card, human verification
- [ ] 03.7-03-PLAN.md -- Gap closure: wire ConflictResolverModal to applyResolution, fix listRemoteBackups return type, remove hollow registerSyncTask

### Phase 4: Gallery Scanning + Deduplication
**Goal**: Users no longer need to manually trigger photo analysis -- the app discovers food photos from the gallery automatically
**Depends on**: Phase 2, Phase 3
**Requirements**: GAL-01, GAL-02, GAL-03, GAL-04, GAL-05
**Success Criteria** (what must be TRUE):
  1. User can manually trigger a gallery scan and see newly discovered food photos queued for resources
  2. App performs periodic background scanning that surfaces new food photos without user intervention, operating within platform constraints (iOS 30-second BGTask, Android WorkManager) using chunked processing blocks
  3. Multiple photos of the same meal (taken within 1-hour window with GPS proximity ~150m) are grouped into a single meal event instead of creating duplicates
  4. Each discovered photo displays EXIF-derived context (timestamp as meal time, location as meal venue)
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Schema extension, type contracts, gallery scan services (discovery, classification, meal grouping, photo import) with unit tests
- [ ] 04-02-PLAN.md -- Background scheduler, foreground drain, Zustand store, GalleryScanScreen UI, permissions, human verification

### Phase 5: Scale OCR + Notifications + Health Data
**Goal**: Users get precise portion weights via kitchen scale OCR, daily macro summaries via push notifications, and weight trend tracking via health platform integration
**Depends on**: Phase 2.6, Phase 4
**Requirements**: DET-04, SCL-01, SCL-02, SCL-03, NTF-01, NTF-02
**Success Criteria** (what must be TRUE):
  1. When a kitchen scale is visible in a food photo, the app reads the displayed weight via 7-segment OCR and user can manage container tare weights (save, auto-subtract, and the app learns frequently used containers over time)
  2. User receives a configurable end-of-day push notification summarizing daily macros, which can also serve as a trigger to bring the app to foreground for gallery processing
  3. User can import weight data from Google Health Connect and view a smoothed weight trend
  4. Detected dishes show inferred hidden ingredients from knowledge graph lookup
**Plans:** 4 plans

Plans:
- [ ] 05-01-PLAN.md -- Hidden ingredients KG enrichment service + daily macro notification service
- [ ] 05-02-PLAN.md -- Scale OCR service (Gemini Nano spike + ML Kit fallback) + container tare weight management
- [ ] 05-03-PLAN.md -- Google Health Connect weight import + EMA-smoothed weight trend service + weight store
- [ ] 05-04-PLAN.md -- UI screens (ScaleInput, WeightTrend, ProfileScreen settings), navigation wiring, human verification

### Phase 6: Sync + Distribution
**Goal**: FTP backup as alternative sync backend alongside Google Drive, and Play for On-Device AI model delivery on Android. iOS features (iCloud, ODR) deferred until Apple has comparable on-device AI.
**Depends on**: Phase 3.7, Phase 5
**Requirements**: DAT-04, DAT-05, DAT-06, MDL-01, MDL-02
**Success Criteria** (what must be TRUE):
  1. User can back up to FTP server alongside Google Drive (both run independently via Promise.allSettled)
  2. FTP credentials stored securely via expo-secure-store (not AsyncStorage)
  3. Android app includes Play for On-Device AI configuration with device targeting; packManager resolves AI pack path before R2 fallback
  4. DAT-04 (Drive sync) and DAT-06 (LWW conflicts) already complete from Phase 3.7
  5. DAT-05 (iCloud) and MDL-02 (iOS ODR) deferred per user decision
**Plans:** 2 plans

Plans:
- [ ] 06-01-PLAN.md -- FTP backup client: native module (Apache Commons Net), ftpClient/ftpSync services, syncScheduler multi-backend dispatch, SyncSettingsScreen FTP card
- [ ] 06-02-PLAN.md -- Play for On-Device AI: withAiPack config plugin, ai-pack-delivery native bridge module, packManager AI pack resolution

### Phase 7.1: Integration Wiring + Dead Code Cleanup (GAP CLOSURE)
**Goal**: Fix 2 broken E2E flows (gallery->diary, scale->portion), drop obsolete requirements (DET-05/DET-06), and remove orphaned code from YOLO->VLM-only pivot
**Depends on**: Phase 4, Phase 5, Phase 6
**Requirements**: DET-05, DET-06
**Gap Closure:** Closes gaps from v1.0 audit
**Success Criteria** (what must be TRUE):
  1. Gallery scan pipeline creates diary entries automatically from discovered meal groups via scanFood + logScanResult
  2. ScaleInputScreen returns confirmed netWeight to caller via navigation params callback; DetectionScreen applies weight proportionally
  3. DET-05 (confidence display) dropped -- Gemini Nano does not produce confidence scores
  4. DET-06 (portionBridge) dropped -- Gemini Nano provides gram estimates directly
  5. Orphaned SmolVLM code removed (VlmDownloadScreen, vlmService.ts, old detection components, portionBridge.ts)
**Plans:** 2 plans

Plans:
- [ ] 07.1-01-PLAN.md -- Gallery pipeline completion + scale weight return: wire drainScanQueue to diary, fix ScaleInput onResult, proportional weight redistribution
- [ ] 07.1-02-PLAN.md -- Dead code removal: delete 10 orphaned files, update barrel exports and navigation, verify TypeScript compiles

### Phase 8: On-device vector search embedding via TFLite MiniLM

**Goal:** Implement on-device query-time text embedding using MiniLM-L6-v2 TFLite with pure-JS WordPiece tokenizer, activating the existing vec search path (usda_embeddings + vec_distance_cosine) for semantic USDA food matching alongside BM25
**Requirements**: EMB-01, EMB-02, EMB-03, EMB-04, EMB-05
**Depends on:** Phase 7
**Success Criteria** (what must be TRUE):
  1. MiniLM-L6-v2 exported as TFLite INT8 (~11MB) with mean pooling + L2 normalization baked into the graph, producing 384-dim normalized float32 vectors matching build_kg.py embeddings
  2. WordPiece vocabulary (30522 tokens) bundled as JSON asset, pure-JS tokenizer handles lowercasing, punctuation, subword splitting, [CLS]/[SEP], attention mask
  3. EmbeddingService loads TFLite model via react-native-fast-tflite with lazy init on first detection flow, embed() returns Float32Array(384) after warmup
  4. Vec search path in vlmPipeline activates automatically when embedding service is ready -- no pipeline code changes needed
  5. Semantic USDA food matching works for non-exact food names (e.g., "tonkatsu" finds "pork, loin" via vector similarity)
**Plans:** 2 plans

Plans:
- [x] 08-01-PLAN.md -- Python export script: MiniLM ONNX export with pooling+norm, Docker onnx2tf INT8 conversion, vocab extraction, validation, asset deployment
- [x] 08-02-PLAN.md -- Pure-JS WordPiece tokenizer + EmbeddingService TFLite implementation, unit tests for both

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.6 -> 02.7 -> 7 -> 3.5 -> 3.6 -> 3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.7 -> 4 -> 5 -> 6 -> 7.1 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure + Data Foundation | 4/4 | Complete | 2026-03-12 |
| 2. On-Device Detection Pipeline | 5/6 | Gap closure | - |
| 2.1. Pre-trained Model Acquisition | 2/3 | In progress | - |
| 2.2. Deploy Custom 335-Class Classifier | 2/2 | Complete | 2026-03-13 |
| 2.3. Food-Specific YOLO Detection | 3/3 | Complete | 2026-03-14 |
| 2.4. Global Cuisine Training Expansion | 3/3 | Complete | 2026-03-14 |
| 2.5. Food Knowledge Graph | 6/6 | Complete | 2026-03-14 |
| 2.6. On-Device VLM Integration | 6/6 | Complete | 2026-03-14 |
| 02.7. Gemini Nano VLM Integration | 2/2 | Complete    | 2026-03-20 |
| 7. Remove YOLO+EfficientNet -- VLM-only | 3/3 | Complete | 2026-03-15 |
| 3.5. OFF Cache + Attribution | 2/2 | Complete | 2026-03-19 |
| 3.6. Incremental Backup System | 2/2 | Complete | 2026-03-19 |
| 3.1. Daily Diary View + Macro Dashboard | 2/2 | Complete | 2026-03-20 |
| 3.2. Food Search + Manual Add + Quick Add | 2/2 | Complete | 2026-03-20 |
| 3.3. Meal Editing + Portion Adjustment | 2/2 | Complete | 2026-03-21 |
| 3.4. Recipe Management | 2/2 | Complete | 2026-03-21 |
| 3.7. Google Drive Sync | 3/3 | Complete | 2026-03-21 |
| 4. Gallery Scanning + Deduplication | 2/2 | Complete | 2026-03-21 |
| 5. Scale OCR + Notifications + Health Data | 4/4 | Complete | 2026-03-21 |
| 6. Sync + Distribution | 2/2 | Complete | 2026-03-21 |
| 7.1. Integration Wiring + Cleanup | 0/2 | Not started | - |
| 8. On-device vector search embedding | 0/2 | Not started | - |

### Phase 9: UX redesign — diary-first home, add food with barcode/photo/voice/gallery, item detail bottom sheet, long-press context menu, copy/move meals

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 8
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 9 to break down)


### Phase 09.1: Dark mode theme with system preference detection and manual toggle (INSERTED)

**Goal:** [Urgent work - to be planned]
**Requirements**: TBD
**Depends on:** Phase 9
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 09.1 to break down)
