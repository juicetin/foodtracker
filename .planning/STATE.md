---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 03.6-02-PLAN.md
last_updated: "2026-03-19T06:05:06.435Z"
progress:
  total_phases: 17
  completed_phases: 10
  total_plans: 42
  completed_plans: 39
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Accurate, effortless food tracking from photos you already take -- no manual entry, no barcode scanning, no subscription, just eat, photograph, and review.
**Current focus:** Phase 03.6 — incremental-backup-system-inserted

## Current Position

Phase: 03.6 (incremental-backup-system-inserted) — EXECUTING
Plan: 2 of 2

## Performance Metrics

**Velocity:**

- Total plans completed: 34 (3 carried from pre-pivot + 4 new phase 1 + 6 phase 2 + 2 phase 02.1 + 2 phase 02.2 + 3 phase 02.3 + 3 phase 02.4 + 5 phase 02.5 + 6 phase 02.6 + 1 phase 03.5 - 1 phase 02.5 duplicate)
- Average duration: 10min
- Total execution time: ~3.1 hours

**Previous Phase 1 (carried forward):**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 6min | 2 tasks | 11 files |
| Phase 01 P02 | 45min | 3 tasks | 8 files |
| Phase 01 P04 | 13min | 2 tasks | 7 files |

| New Phase 01 P01 | 8min | 2 tasks | 16 files |
| New Phase 01 P02 | 19min | 2 tasks | 12 files |
| New Phase 01 P03 | 7min | 2 tasks | 8 files |
| New Phase 01 P04 | 4min | 1 task | 2 files |

**Recent Trend:**

- Last 3 plans: 3min, 25min, 25min
- Trend: Moderate (VLM integration plans include verification checkpoints and orchestrator fixes)

*Updated after each plan completion*
| Phase 02 P01 | 3min | 2 tasks | 7 files |
| Phase 02 P02 | 6min | 2 tasks | 6 files |
| Phase 02 P03 | 8min | 2 tasks | 6 files |
| Phase 02 P04 | 3min | 2 tasks | 8 files |
| Phase 02 P05 | 15min | 3 tasks | 10 files |
| Phase 02 P06 | 4min | 2 tasks | 6 files |

| Phase 02.1 P01 | 19min | 1 task | 6 files |
| Phase 02.1 P02 | 7min | 2 tasks | 8 files |

| Phase 02.2 P01 | 4min | 2 tasks | 9 files |
| Phase 02.2 P02 | 7min | 2 tasks | 7 files |

| Phase 02.3 P01 | 12min | 2 tasks | 5 files |
| Phase 02.3 P02 | 4min | 2 tasks | 3 files |
| Phase 02.3 P03 | 3min | 2 tasks | 0 files |

| Phase 02.4 P01 | 35min | 2 tasks | 5 files |
| Phase 02.4 P02 | 10min | 1 task | 5 files |
| Phase 02.4 P03 | 10min | 2 tasks | 6 files |

| Phase 02.5 P02 | 4min | 2 tasks | 6 files |
| Phase 02.5 P01 | 210min | 2 tasks | 6 files |
| Phase 02.5 P01 | 14min | 2 tasks | 6 files |
| Phase 02.5 P03 | 5min | 2 tasks | 8 files |
| Phase 02.5 P04 | 4min | 2 tasks | 3 files |
| Phase 02.5 P05 | 5min | 2 tasks | 4 files |
| Phase 02.5 P06 | 38min | 2 tasks | 2 files |

| Phase 02.6 P01 | 4min | 2 tasks | 9 files |
| Phase 02.6 P02 | 5min | 2 tasks | 4 files |
| Phase 02.6 P03 | 3min | 2 tasks | 5 files |
| Phase 02.6 P04 | 3min | 2 tasks | 3 files |
| Phase 02.6 P05 | 25min | 3 tasks | 11 files |
| Phase 02.6 P06 | 25min | 2 tasks | 13 files |
| Phase 07 P02 | 4min | 2 tasks | 4 files |
| Phase 07 P01 | 9min | 2 tasks | 11 files |
| Phase 07 P03 | 15min | 3 tasks | 7 files |

| Phase 03.5 P01 | 4min | 2 tasks | 6 files |
| Phase 03.5 P02 | 3min | 2 tasks | 2 files |
| Phase 03.6 P01 | 3min | 2 tasks | 4 files |
| Phase 03.6 P02 | 2min | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [ADR-005]: Local-first, no-subscription architecture -- all inference on-device, bundled nutrition data, optional cloud sync
- [ADR-005]: op-sqlite replaces PostgreSQL; bundled USDA replaces runtime API
- [ADR-005]: LWW conflict resolution for sync; CRDTs overkill for single-user food logs
- [Roadmap]: Prior plans 01-03, 01-05, 01-06 incorporated into new Phase 2 (detection pipeline)
- [Roadmap]: Prior plans 01-01, 01-02, 01-04 carried forward as validated work
- [01-01]: op-sqlite v15.x uses codegen autolinking, not Expo config plugin
- [01-01]: db/client.ts is canonical location for all DB connections (userDb + openNutritionDb)
- [01-01]: Write-first-then-refresh pattern for all Zustand store mutations
- [01-01]: Soft-delete via isDeleted flag instead of row removal
- [01-02]: expo-file-system v19 uses class-based API (File/Directory/Paths), not legacy functional API
- [01-02]: Nutrition queries use raw SQL via op-sqlite, not drizzle-orm (separate schema)
- [01-02]: R2 download with X-API-Key header for Phase 1; full attestation deferred to Phase 6
- [01-02]: Both platforms download from R2 for Phase 1; platform-native delivery deferred to Phase 6
- [01-02]: PackManager is generic -- same logic for nutrition DBs and ML model packs
- [01-03]: Regional DBs use standard USDA FDC nutrient IDs for cross-DB compatibility
- [01-03]: CoFID/CIQUAL use synthetic sequential fdc_id (non-numeric source codes)
- [01-03]: Locale prefix matching for language families (fr-* -> ciqual)
- [01-03]: RegionalResolver priority: regional (1) > usda-core (2) > usda-branded (3) > custom (4)
- [01-04]: importCustomPack composes existing primitives (validatePackSchema + file copy + DB insert + addDatabase) -- no new infrastructure
- [01-04]: Schema validation failure throws before any file copy or DB registration -- no partial state on error
- [Phase 01]: importCustomPack composes existing primitives (validatePackSchema + file copy + DB insert + addDatabase) -- no new infrastructure
- [02-01]: TFLiteModel interface uses ArrayBufferLike[] matching react-native-fast-tflite's TensorflowModel shape
- [02-01]: FP16 quantisation only (no INT8) -- avoids calibration dataset and preserves food colour accuracy
- [02-01]: NMS performed in JavaScript, not baked into TFLite model -- cross-platform portability
- [02-02]: inferenceRouter uses getModelSet() (not loadModelSet()) to enforce pre-loading pattern
- [02-02]: Portion estimates placeholder (method: pending) -- portionBridge fills in Plan 03
- [02-02]: Detection IDs use monotonic counter + timestamp for RN runtime compatibility
- [02-02]: Transposed YOLO access: output[row * numPredictions + col] is correct pattern
- [02-03]: Density table has 81 entries (not 55 as plan estimated) -- all ported faithfully from Python
- [02-03]: Standard servings 52 entries + separate category_defaults fallback layer
- [02-03]: Suggestion threshold of 3 corrections ensures pattern-based recommendations
- [02-03]: Uses crypto.randomUUID() for correction record IDs (matches useFoodLogStore convention)
- [02-04]: View-based absolute positioning for bounding boxes instead of react-native-svg (not installed)
- [02-04]: Detection store is ephemeral (in-memory only) -- no SQLite persistence until Log Meal
- [02-04]: Rough calorie/protein estimates (1.5 kcal/g, 0.1g protein/g) as Phase 2 proxy
- [02-05]: @react-native-community/slider for portion adjustment (reliable cross-platform vs custom gesture)
- [02-05]: DetectionScreen state machine pattern (idle/picking/detecting/results/logging) for clear flow control
- [02-05]: Bottom sheet snap points at 40%/70% for compact and expanded detail views
- [02-05]: Barrel index pattern for detection component module -- all components from single index.ts
- [02-06]: Pure-JS PNG decoder for pixel extraction -- no additional native dependencies
- [02-06]: manipulateAsync legacy API for simplicity over new context-based ImageManipulator API
- [02-06]: Direct ArrayBuffer cast (as ArrayBuffer) for TFLite outputs instead of instanceof checks
- [02-06]: PNG format for base64 output (lossless) to preserve pixel accuracy for model input
- [02.1-01]: AIY Food V1 actual properties: 192x192 uint8 quantized input (not 224x224 float32), 2024 classes (not 2023). Scale=0.0078125, zero_point=128.
- [02.1-01]: YOLO26n TFLite export fails (onnx2tf TopK error); YOLO11n succeeds as fallback with [1,84,8400] output shape
- [02.1-01]: Kaggle Models API replaces GCS for AIY download (GCS returns 403). Returns tar.gz archive.
- [02.1-01]: ai-edge-litert (v2.1.2) replaces tflite-runtime for Python 3.12+ TFLite validation
- [02.1-01]: AIY Food V1 English labels extracted from embedded probability-labels-en.txt metadata (2024 food names)
- [02.1-02]: BINARY_INPUT_SIZE=192 (not 224) and 2024 classes (not 2023) per actual AIY model dimensions from Plan 01
- [02.1-02]: Manual loop for binary gate max (not Math.max(...spread)) to avoid stack overflow on 2024-element array
- [02.1-02]: Dual-buffer pipeline: detectBuffer (640x640) for YOLO, classifyBuffer (192x192) for AIY binary gate + classify
- [02.1-02]: Food-only post-filter in DetectionScreen using COCO_FOOD_CLASS_IDS with __DEV__ debug logging
- [02.1-02]: tfliteAsset.js mock (returns numeric 1) for Jest .tflite moduleNameMapper
- [02.2-01]: EfficientNet-Lite0 INT8 (3.9MB) replaces AIY Food V1 (21MB) -- 5.4x smaller, food-only trained
- [02.2-01]: Binary gate removed -- EfficientNet-Lite0 is food-only so no food-vs-not-food step needed
- [02.2-01]: Food-101 fallback removed -- 101 classes are strict subset of new 335
- [02.2-01]: ImageNet normalization constants (IMAGENET_MEAN, IMAGENET_STD) exported for classify preprocessing
- [02.2-02]: CLASSIFY_CONFIDENCE_THRESHOLD=0.15 for fallback label -- items always returned, labeled 'Food Item' when low confidence
- [02.2-02]: formatClassLabel generalizes formatFood101Label for any snake_case class name to Title Case
- [02.2-02]: preprocessImageForModel gains 'imagenet' normalization mode parameter for classifier input
- [02.3-01]: Docker-based onnx2tf conversion because TensorFlow has no Python 3.14 wheels
- [02.3-01]: Dynamic range INT8 quantization (3.6MB) -- float32 IO compatible with existing pipeline
- [02.3-01]: DETECT_CLASS_NAMES loaded from labels_detect.json (same pattern as CLASSIFY_CLASS_NAMES)
- [02.3-02]: Per-item YOLO labels as primary className -- classifier serves as secondary confirmation only
- [02.3-02]: formatFoodLabel replaces formatClassLabel, handles hyphens/underscores/spaces for GGCD names
- [02.3-02]: Classifier result logged in __DEV__ mode for debugging, not applied to items
- [02.3-03]: Clean install required when swapping bundled models -- stale installed_packs DB entries cause old model to load
- [02.3-03]: modelLoader pack-path priority bug deferred to future phase (needs version check/migration)
- [02.3-03]: GGCD component-level detection for composite dishes is expected behavior, not a bug
- [02.4-01]: Conservative fuzzy matching (Levenshtein 1 for names >= 10 chars) to avoid false food merges
- [02.4-01]: Manual dedup map for cross-dataset duplicates (burger/hamburger, pakora/pakode, etc.)
- [02.4-01]: Symlink-based merge -- all merged_v2 images are os.symlink() to source datasets, zero duplication
- [02.4-01]: merged_v2 at 370 classes (expandable to 700+ after Kaggle auth and re-run)
- [02.4-02]: Docker-based onnx2tf conversion (tensorflow/tensorflow:2.18.0) because Python 3.14 lacks ai-edge-torch/onnx2tf
- [02.4-02]: Dynamic range INT8 quantization (weights-only, float32 I/O) matches existing pipeline
- [02.4-02]: Training exceeded expectations: 76.72% top-1 on 905 classes (vs 74.17% on 335 classes)
- [02.4-03]: CLASSIFY_CLASS_NAMES dynamically loaded from labels_classify.json -- model swap needs zero code changes
- [02.4-03]: 241 CNFOOD-241 classes have numeric names (000-240) -- classifier is secondary to YOLO labels anyway
- [Phase 02.5]: SymSpell implemented inline (~230 lines) instead of npm symspell-ex package (250 downloads/week, stale)
- [Phase 02.5]: FTS5 -> alias FTS5 -> SymSpell fallback chain covers exact, alias, and fuzzy matching in priority order
- [Phase 02.5]: Recipe decomposition: sum ingredient USDA per-100g values scaled by quantity, then scale to portion/recipe ratio
- [Phase 02.5]: corbt/all-recipes (537K Parquet) replaces RecipeNLG (manual download deprecated)
- [Phase 02.5]: 500 USDA SR Legacy foods embedded in build_kg.py for self-contained pipeline
- [Phase 02.5]: SymSpell edit distance 1 (not 2) for performance with 10K+ dish names
- [Phase 02.5]: Temporary index on recipe_ingredient(ingredient_name) during USDA linking -- 300x speedup (10min to 2s)
- [Phase 02.5]: corbt/all-recipes dataset (537K Parquet) replaces RecipeNLG (manual download, script API deprecated)
- [Phase 02.5]: 500 USDA SR Legacy foods embedded in build_kg.py for self-contained pipeline, no external CSV download
- [02.5-03]: food-knowledge.db bundled in APK via assets/data/ (under 70MB, avoids mandatory first-run download)
- [02.5-03]: expo-asset Asset.fromModule() resolves bundled .db to filesystem path for op-sqlite open()
- [02.5-03]: Three-tier nutrition fallback: KG recipe decomposition -> KG dish averages -> flat-rate proxy
- [02.5-03]: Lazy KG initialization on first detection flow, not at app boot (no startup cost)
- [02.5-04]: Ingredient-specific cup/tbsp overrides (flour=120g/cup, oil=218g/cup) instead of single water-default
- [02.5-04]: parse_quantity_grams returns None for unparseable; caller applies 50g fallback
- [02.5-04]: All dishes included (no cap, no minimum threshold); tiered confidence for 1-2 vs >=3 recipe counts
- [02.5-05]: Tiered USDA loading: full CSV (7,793 foods) -> auto-download -> embedded 500-food subset
- [02.5-05]: Levenshtein fuzzy matching (Strategy 6) as supplement to direct map, invoked only after 5 faster strategies fail
- [02.5-05]: Classifier/detector labels seeded before recipes; original labels registered as dish_alias type 'model_label'
- [Phase 02.5]: MIN_RECIPE_COUNT=10 threshold balances 15K+ dish coverage vs 70MB mobile bundling limit
- [Phase 02.5]: Top 30 ingredients per dish cap reduces DB size from 1.4M to 452K rows while preserving nutrition accuracy
- [02.6-01]: llama.rn v0.11.4 with enableEntitlements, forceCxx20, enableOpenCL Expo plugin config
- [02.6-01]: expo-device getter-based mock pattern for per-test totalMemory mutation in Jest
- [02.6-01]: SmolVLM family tier configs: budget 256M (365MB), mid 500M (546MB), high 2.2B (1.3GB)
- [Phase 02.6]: createDownloadResumable from expo-file-system/legacy replaces fetch().arrayBuffer() for streaming to disk -- avoids 300MB+ OOM
- [Phase 02.6]: All pack types use streaming download (not just VLM) for consistent OOM prevention
- [Phase 02.6]: VLM paired files: model deleted if mmproj download fails (atomic cleanup)
- [02.6-03]: Singleton object literal pattern (not class) for VlmService matching PackManager convention
- [02.6-03]: 60s inactivity timeout auto-releases VLM context to free RAM on constrained devices
- [02.6-03]: JSON.parse fallback returns { dishes: [] } on grammar constraint failure (defense-in-depth)
- [02.6-04]: VLM fields on DetectedItem are all optional for backward compatibility
- [02.6-04]: setRefining propagates isRefining to all items; displayLabel uses vlmLabel ?? className fallback
- [02.6-05]: Word overlap matching (substring + word ratio) for VLM-to-YOLO dish pairing, with positional fallback
- [02.6-05]: Debounced 500ms re-refinement on user text input changes
- [02.6-05]: patch-package fix for llama.rn v0.11.4 ESM/CJS config plugin incompatibility
- [02.6-05]: VLM init is lazy (on first detection results), not at app boot
- [02.6-06]: VLM download screen uses detectVlmTier() for automatic tier selection -- no manual tier picker
- [02.6-06]: Detection gated behind VLM availability -- YOLO-only fallback removed (YOLO labels unreliable, e.g. ramen->"egg")
- [02.6-06]: VlmPipeline throws if VLM not ready instead of silently returning bad labels -- fail-fast for correctness
- [02.6-06]: E2E verification on physical device deferred -- emulator insufficient RAM for VLM download
- [Phase 07]: Positional matching only (no substring/word-overlap) since className is always 'Food Region'
- [Phase 07]: identifyWithRetry returns { dishes: [] } on double failure instead of throwing
- [Phase 07]: displayLabel returns empty string during isRefining (shimmer state), 'Unknown food' as final fallback
- [Phase 07]: All detection items labelled 'Food Region' with isRefining=true -- VLM provides actual food names
- [Phase 07]: YOLO labels logged in __DEV__ mode only for debugging, not displayed to user
- [Phase 07]: RefiningBadge replaced by inline shimmer inside bbox and list items (not header)
- [Phase 03.5]: Stale-while-revalidate for OFF cache: products 7-day, searches 24-hour freshness
- [Phase 03.5]: INSERT OR REPLACE for cache upserts; normalized cache keys (trim+lowercase) for search
- [Phase 03.5]: loadExportOFFCache returns [] on error -- safe if off_product_cache table not yet created
- [Phase 03.5]: OFF cache CSV uses comment-style header matching existing Recipes/Favourites pattern
- [Phase 03.6]: expo-file-system v19 class API (Paths, File, Directory) for backup file operations
- [Phase 03.6]: Copy-and-replay compaction (not VACUUM INTO shortcut) per user decision
- [Phase 03.6]: Side-effect import in App.tsx ensures defineTask runs at module load before React renders

### Roadmap Evolution

- Phase 02.1 inserted after Phase 02: Pre-trained model acquisition and TFLite integration (URGENT) — no .tflite models exist in repo; pipeline untestable without real models. Uses Google AIY Food V1 (classification) + YOLO26n COCO (detection) as zero-training baseline.
- Phase 02.5 replaced: "Nutrition & Metadata Enrichment" (static JSON mapping) → "Food Knowledge Graph" (recipe-based nutrition decomposition via SQLite KG). Old plans archived to `_archived_02.5-nutrition-and-metadata-enrichment/`. KG subsumes the ClassNutritionMapper concept with live recipe→ingredient→USDA decomposition instead of static JSON.
- Phase 02.6 inserted: "On-Device VLM Integration" — pulled forward from Phase 5. SmolVLM family via llama.rn (revised from ADR-005 tiers: Moondream 0.5B and Gemma 3n E2B not viable for multimodal GGUF). Progressive YOLO→VLM refinement with text+image fusion.
- Phase 5 scope reduced: VLM + KG hidden ingredients moved to 2.5+2.6. Phase 5 retains Scale OCR, notifications, health data import.
- Phase 3 dependency updated: now depends on Phase 2.5 (KG) + Phase 2.6 (VLM) instead of just Phase 2.
- Phase 7 added: Remove YOLO and EfficientNet pipeline entirely — VLM-only detection
- Phase 02.7 inserted after Phase 02.6: Gemini Nano System-Managed VLM Integration — spike-first quality test of ML Kit GenAI APIs (Prompt API) on Pixel 9 Pro, then proper pipeline integration as Tier 0 above SmolVLM.

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: ~30-35% of active Android devices have <=4GB RAM -- tiered model delivery is critical
- [Research]: Thermal throttling at ~2.5min sustained inference -- batch processing needs bursty pattern
- [Research]: Gemini Nano foreground-only restriction blocks background gallery scanning inference
- [Research]: CoreML/LiteRT model conversion can fail silently -- validate on-device outputs early
- [Research]: Base APK must stay under 100MB (6MB = ~1% conversion drop)

## Session Continuity

Last session: 2026-03-19T05:58:58.660Z
Stopped at: Completed 03.6-02-PLAN.md
Resume file: None
