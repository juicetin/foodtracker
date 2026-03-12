# Architecture: Local-First Integration with Existing React Native + Expo Codebase

**Domain:** On-device ML food tracking -- local-first pivot integration
**Researched:** 2026-03-12
**Confidence:** MEDIUM-HIGH

## Executive Summary

The existing codebase is a React Native + Expo (SDK 54, New Architecture enabled) scaffold with navigation, Zustand state management, photo picker components, and TypeScript types. The backend (Express + PostgreSQL) and AI agent service (Google ADK) are being removed. The local-first pivot requires adding 6 new native module integrations (op-sqlite, react-native-fast-tflite, llama.rn, react-native-cloud-storage, custom 7-segment OCR, expo-media-library enhanced usage) and refactoring 3 existing layers (data layer from API to local SQLite, state management from in-memory to persistent, photo processing from cloud to on-device).

The critical constraint: **Expo CNG (Continuous Native Generation) via `npx expo prebuild` replaces the need for bare workflow eject.** All native dependencies (op-sqlite, react-native-fast-tflite, llama.rn) support Expo through config plugins or automatic autolinking. Expo Go will no longer work -- development shifts to custom dev client builds via `npx expo run:ios` / `npx expo run:android` or EAS Build.

## Recommended Architecture

```
Mobile App (React Native / Expo SDK 54 + CNG)
|
+-- Presentation Layer (KEEP + EXTEND)
|   +-- Navigation (react-navigation)           <- keep as-is
|   +-- Screens (Home, Diary, Profile, +new)     <- extend with 4-5 new screens
|   +-- Components (PhotoPicker, BatchGrid, +new) <- extend significantly
|   +-- Theme / Design system                     <- new
|
+-- State Layer (REFACTOR)
|   +-- Zustand stores                           <- refactor from in-memory to SQLite-backed
|   +-- usePreferencesStore                      <- keep, migrate to op-sqlite
|   +-- useFoodLogStore                          <- heavy refactor to SQLite queries
|   +-- useInferenceStore (new)                  <- ML pipeline state
|   +-- useSyncStore (new)                       <- sync status and queue
|
+-- Service Layer (NEW -- replaces API client + AI agents)
|   +-- InferenceService                         <- orchestrates ML pipeline
|   |   +-- YoloPipeline (react-native-fast-tflite)
|   |   +-- VlmPipeline (llama.rn)
|   |   +-- ScaleOcrPipeline (react-native-fast-tflite)
|   |   +-- GeminiNanoPipeline (ML Kit GenAI)
|   |   +-- InferenceRouter                      <- selects pipeline by device capability
|   +-- NutritionService                         <- queries bundled nutrition DB
|   +-- PortionService                           <- port of Python portion_estimator.py
|   +-- GalleryService                           <- expo-media-library scanning + dedup
|   +-- SyncService                              <- pluggable sync adapters
|
+-- Data Layer (REPLACE)
|   +-- op-sqlite                                <- user data (entries, recipes, history)
|   +-- Bundled knowledge graph (SQLite, read-only) <- already exists as food-knowledge.db
|   +-- Bundled USDA nutrition DB (SQLite, read-only) <- new, ~50-80MB
|   +-- Expo FileSystem                          <- photo storage (already in use)
|
+-- Native Bridge Layer (NEW)
    +-- react-native-fast-tflite                 <- YOLO + Scale OCR models
    +-- llama.rn                                 <- VLM inference (GGUF models)
    +-- @op-engineering/op-sqlite                 <- SQLite with JSI
    +-- react-native-cloud-storage               <- Google Drive + iCloud sync
    +-- expo-media-library                       <- gallery access (existing dep)
    +-- expo-file-system                         <- file I/O (existing dep)
```

## Component Inventory: Keep / Modify / Remove / Add

### KEEP (as-is or minimal changes)

| Component | Path | Rationale |
|-----------|------|-----------|
| App shell + GestureHandler | `App.tsx` | Root component unchanged |
| RootNavigator | `src/navigation/RootNavigator.tsx` | Stack navigator structure works, add screens |
| MainTabNavigator | `src/navigation/MainTabNavigator.tsx` | Tab structure works, add tabs |
| Navigation types | `src/navigation/types.ts` | Extend, don't replace |
| PhotoPicker component | `src/components/PhotoPicker.tsx` | Photo selection UI works as-is |
| BatchPhotoGrid component | `src/components/BatchPhotoGrid.tsx` | Grid display works as-is |
| expo-image-picker dep | `package.json` | Still needed for manual photo selection |
| expo-file-system dep | `package.json` | Still needed for local photo storage |
| expo-media-library dep | `package.json` | Needed, enhanced usage for gallery scanning |
| react-native-gesture-handler | `package.json` | UI dependency, keep |
| react-native-reanimated | `package.json` | UI dependency, keep |
| Zustand | `package.json` | State management, keep |
| Knowledge graph DB | `src/data/food-knowledge.db` | Already bundled, central to new arch |
| Training scripts | `training/` | Model development pipeline, untouched |
| Knowledge graph scripts | `knowledge-graph/` | DB seeding/export, untouched |

### MODIFY (significant refactoring)

| Component | Current State | Required Changes |
|-----------|---------------|------------------|
| `useFoodLogStore.ts` | In-memory array, no persistence | Refactor to read/write from op-sqlite; SQLite becomes source of truth, Zustand becomes reactive cache |
| `usePreferencesStore.ts` | AsyncStorage persistence | Migrate from AsyncStorage to op-sqlite for consistency; or keep AsyncStorage since prefs are small |
| `src/types/index.ts` | Has `gcsUrl`, `userId`, `APIResponse`, `databaseSource` types | Remove `gcsUrl`, `userId` (single-user, no auth), `APIResponse`; add sync metadata fields (`updatedAt`, `isSynced`, `isDeleted`); keep core `FoodEntry`, `Ingredient`, `Photo`, `DetectedItem`, `ScaleReading` types |
| `app.json` | Basic Expo config | Add config plugins for op-sqlite, react-native-fast-tflite (CoreML delegate, GPU libs), llama.rn, react-native-cloud-storage; add iCloud entitlements; add Android permissions |
| `package.json` | Has async-storage, no native ML deps | Add op-sqlite, react-native-fast-tflite, llama.rn, react-native-cloud-storage; remove async-storage (optional) |
| HomeScreen | Mock "process photos" with Alert | Replace with real on-device inference pipeline call |
| DiaryScreen | Empty state placeholder | Connect to op-sqlite queries for food entries |

### REMOVE (superseded by local-first)

| Component | Path | Why |
|-----------|------|-----|
| API client | `src/lib/api/client.ts` | No backend server; all data local |
| Food log API | `src/lib/api/foodLogApi.ts` | Replaced by local InferenceService + op-sqlite |
| API barrel export | `src/lib/api/index.ts` | No API layer needed |
| Backend directory | `backend/` | Express + PostgreSQL entirely superseded |
| AI agent service | `services/ai-agent/` | Google ADK agents entirely superseded |

### ADD (new modules)

| Module | Purpose | Native Dependency | Priority |
|--------|---------|-------------------|----------|
| **Database module** (`src/data/`) | op-sqlite setup, migrations, typed queries | @op-engineering/op-sqlite | P0 -- everything depends on this |
| **Schema + migrations** (`src/data/schema.ts`) | SQLite schema ported from PostgreSQL init.sql | op-sqlite | P0 |
| **YOLO inference module** (`src/services/inference/yolo.ts`) | Load and run YOLO TFLite/CoreML models | react-native-fast-tflite | P0 |
| **Inference router** (`src/services/inference/router.ts`) | Device capability detection, pipeline selection | Platform APIs | P0 |
| **Nutrition lookup** (`src/services/nutrition/`) | Query bundled USDA DB by FDC ID or name | op-sqlite (read-only) | P0 |
| **Portion estimator (TS port)** (`src/services/portion/`) | Port Python portion_estimator.py to TypeScript | None (pure logic) | P1 |
| **VLM inference module** (`src/services/inference/vlm.ts`) | Load and run VLM for complex dish ID | llama.rn | P1 |
| **Scale OCR module** (`src/services/inference/scaleOcr.ts`) | 7-segment display reading pipeline | react-native-fast-tflite | P1 |
| **Gallery scanner** (`src/services/gallery/`) | Background/periodic photo discovery + EXIF | expo-media-library | P1 |
| **Photo deduplication** (`src/services/gallery/dedup.ts`) | pHash-based duplicate detection | Image processing lib | P1 |
| **Sync service** (`src/services/sync/`) | Pluggable adapter pattern for cloud sync | react-native-cloud-storage | P2 |
| **Google Drive adapter** (`src/services/sync/googleDrive.ts`) | Google Drive appDataFolder sync | react-native-cloud-storage | P2 |
| **iCloud adapter** (`src/services/sync/icloud.ts`) | iCloud sync for iOS | react-native-cloud-storage | P2 |
| **Gemini Nano adapter** (`src/services/inference/geminiNano.ts`) | Opportunistic AICore inference on supported devices | ML Kit GenAI (Android) | P2 |
| **Model manager** (`src/services/models/`) | Download, cache, version ML models | expo-file-system | P1 |
| **Device capability detector** (`src/services/device/`) | RAM, chipset, NPU detection for tiered models | Platform APIs | P1 |

## Data Flow Diagrams

### Primary Flow: Photo to Nutrition Entry

```
User picks photo(s) via PhotoPicker
       |
       v
[GalleryService] -- reads EXIF metadata (timestamp, location)
       |
       v
[InferenceRouter] -- checks device capabilities (RAM, NPU, installed models)
       |
       +-- RAM >= 8GB + VLM installed --> VLM pipeline
       +-- RAM >= 4GB + YOLO only     --> YOLO pipeline
       +-- Gemini Nano available       --> try GeminiNano (foreground only)
       |
       v
[YoloPipeline] (primary path, always available)
       |
       +-- Stage 1: Binary food/not-food classifier (~2ms NPU, ~10ms CPU)
       |   +-- NOT food --> skip photo, mark as non-food
       |   +-- IS food --> continue
       |
       +-- Stage 2: Object detection with bounding boxes (~5-8ms NPU, ~50ms CPU)
       |   +-- returns: [{name, bbox, confidence}]
       |   +-- detects: food items, plates, bowls, scale displays
       |
       +-- Stage 3: Classification on each detected item (~5ms per item)
       |   +-- returns: specific dish name per bounding box
       |
       +-- [Optional] Stage 4: VLM for low-confidence items
           +-- only if VLM is loaded and item confidence < threshold
           +-- sends cropped image region to VLM
           +-- returns: refined dish identification
       |
       v
[ScaleOcrPipeline] (if scale detected in Stage 2)
       |
       +-- Crop scale display region from bounding box
       +-- Preprocess: threshold, perspective correction
       +-- Run 7-segment TFLite model on digit regions
       +-- Post-process: validate range, single decimal, whitelist
       +-- returns: ScaleReading {value, unit, confidence}
       |
       v
[PortionService] (TypeScript port of portion_estimator.py)
       |
       +-- If scale reading: use exact weight
       +-- Else if reference objects (plate, bowl): geometry estimation
       +-- Else if user history for this dish: history extrapolation
       +-- Else: USDA default serving size from knowledge graph
       +-- returns: PortionEstimate {weight_g, confidence, method}
       |
       v
[NutritionService]
       |
       +-- For each detected ingredient:
       |   +-- Query knowledge graph: dish -> ingredients -> weight percentages
       |   +-- Query USDA DB: ingredient -> nutrients per 100g
       |   +-- Calculate: (ingredient_weight_g / 100) * nutrients_per_100g
       +-- Aggregate: total calories, protein, carbs, fat per entry
       +-- returns: FoodEntry with ingredients and totals
       |
       v
[op-sqlite] -- persist FoodEntry, ingredients, photo references
       |
       v
[Zustand store] -- update reactive UI cache
       |
       v
UI renders entry in Diary with edit capability
```

### Gallery Scanning Flow

```
App comes to foreground / periodic timer / user triggers manual scan
       |
       v
[GalleryService.scanNewPhotos()]
       |
       +-- expo-media-library.getAssetsAsync({
       |       createdAfter: lastScanTimestamp,
       |       mediaType: 'photo',
       |       sortBy: 'creationTime'
       |   })
       |
       +-- For each new photo:
       |   +-- Extract EXIF (timestamp, GPS, camera info)
       |   +-- Compute perceptual hash (pHash) for dedup
       |   +-- Check pHash against existing entries
       |   +-- If duplicate within time window: group as meal event
       |   +-- If unique: add to scan queue
       |
       v
[Inference pipeline] -- process scan queue in bursty batches
       |                  (5 photos, pause 2-3s, next 5 -- thermal mgmt)
       |
       v
[Pending entries] -- created with confidence scores
       |
       +-- High confidence: auto-log, show in diary
       +-- Medium confidence: show confirmation prompt
       +-- Low confidence: show for user review with suggestions
       |
       v
[End-of-day notification] -- "We found 3 meals today. Review?"
```

### Sync Flow (Optional)

```
User enables sync in Settings
       |
       v
[SyncService.configure(provider: 'google-drive' | 'icloud')]
       |
       +-- google-drive: OAuth2 sign-in, request drive.appdata scope
       +-- icloud: native iCloud entitlement (automatic on iOS)
       |
       v
On data change (new entry, edit, delete):
       |
       v
[SyncService.enqueue(changeset)]
       |
       +-- Mark records: isSynced = false, updatedAt = now()
       +-- Add to outbox queue in op-sqlite
       |
       v
On connectivity + foreground (or user-triggered):
       |
       v
[SyncAdapter.push(outbox)]
       |
       +-- Export changed records as JSON
       +-- Upload to cloud storage (file-level for MVP)
       +-- On success: mark records isSynced = true
       |
       v
[SyncAdapter.pull()]
       |
       +-- Download remote state
       +-- Compare updatedAt timestamps (LWW resolution)
       +-- Apply remote changes to local op-sqlite
       +-- Update Zustand cache
```

## Component Boundaries

| Component | Responsibility | Communicates With | Interface |
|-----------|---------------|-------------------|-----------|
| **Presentation (screens/components)** | UI rendering, user interaction | State layer (Zustand) | React hooks, props |
| **State layer (Zustand stores)** | Reactive state cache, derived computations | Data layer, Service layer | `useStore()` hooks |
| **InferenceService** | ML model loading, pipeline orchestration | Native bridges (tflite, llama.rn) | `async processPhotos(photos): InferenceResult` |
| **NutritionService** | Nutrient lookup from bundled DBs | op-sqlite (read-only DBs) | `async lookupNutrients(ingredients): NutrientData[]` |
| **PortionService** | Weight estimation from visual cues | None (pure logic) | `estimate(bbox, imageSize, dish, refs): PortionEstimate` |
| **GalleryService** | Photo discovery, EXIF, dedup | expo-media-library, op-sqlite | `async scanNewPhotos(): Photo[]` |
| **SyncService** | Cloud backup/restore | react-native-cloud-storage, op-sqlite | `async push()`, `async pull()` |
| **Database (op-sqlite)** | Persistent storage, migrations | None (data store) | SQL queries via typed wrapper |
| **ModelManager** | ML model download, caching, versioning | expo-file-system, Platform APIs | `async ensureModel(modelId): string` (returns path) |

## Expo CNG vs Bare Workflow Decision

**Use Expo CNG with prebuild. Do NOT eject to bare workflow.**

| Factor | Expo CNG (prebuild) | Bare Workflow |
|--------|---------------------|---------------|
| Native module support | All needed libs have config plugins or autolink | Same, but manual Xcode/Gradle config |
| op-sqlite | Autolinks via prebuild, no plugin needed | Same |
| react-native-fast-tflite | Has config plugin for CoreML/GPU delegates | Manual podspec/gradle setup |
| llama.rn | Autolinks, no special config | Same |
| react-native-cloud-storage | Has config plugin for iCloud entitlements | Manual entitlement config |
| OTA updates | EAS Update supported | EAS Update supported |
| Maintenance burden | Regenerate native dirs on demand | Maintain ios/ and android/ manually |
| CI/CD | EAS Build handles native compilation | Must maintain build infra |

**Critical: Expo Go is no longer usable once native modules are added.** Development requires:
- `npx expo prebuild` to generate ios/ and android/ directories
- `npx expo run:ios` / `npx expo run:android` for local dev builds
- Or EAS Build for cloud-based dev client builds

The project already has `newArchEnabled: true` in app.json, which is required for JSI-based libraries (op-sqlite, react-native-fast-tflite, llama.rn).

## Schema Migration: PostgreSQL to SQLite

The existing PostgreSQL schema (`backend/db/init.sql`) maps cleanly to SQLite with these changes:

| PostgreSQL Feature | SQLite Equivalent |
|-------------------|-------------------|
| `UUID` primary keys | `TEXT` with `uuid()` generated in app (use `crypto.randomUUID()`) |
| `uuid_generate_v4()` | Generate in TypeScript before insert |
| `SERIAL` / `BIGSERIAL` | `INTEGER PRIMARY KEY AUTOINCREMENT` |
| `DECIMAL(10,2)` | `REAL` (SQLite has no fixed-point) |
| `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | `TEXT DEFAULT (datetime('now'))` |
| `VARCHAR(n)` | `TEXT` (SQLite ignores length) |
| `BOOLEAN` | `INTEGER` (0/1) |
| `REFERENCES ... ON DELETE CASCADE` | Supported in SQLite with `PRAGMA foreign_keys=ON` |
| Trigger-based `updated_at` | Same pattern works in SQLite |
| `CREATE EXTENSION` | Not applicable |

**Tables to migrate from PostgreSQL:**
- `food_entries` -- keep all columns, drop `user_id` FK (single-user), add sync metadata
- `ingredients` -- keep all columns, add `entry_id` FK
- `photos` -- keep, drop `gcs_url`, add `local_path`
- `modification_history` -- keep, drop `modified_by` FK
- `custom_recipes` -- keep, drop `user_id` FK
- `recipe_ingredients` -- keep as-is
- `recipe_photos` -- keep, replace `gcs_url` with `local_path`

**New tables for local-first:**
- `sync_outbox` -- pending sync operations (`id`, `table_name`, `record_id`, `operation`, `created_at`)
- `scan_queue` -- gallery scanner pending photos (`id`, `asset_id`, `uri`, `created_at`, `status`)
- `photo_hashes` -- perceptual hashes for dedup (`photo_id`, `phash`, `created_at`)
- `user_settings` -- replaces `users` + `nutrition_goals` tables (single-user, flattened)
- `container_weights` -- user's container tare weights for scale accuracy
- `model_cache` -- downloaded ML model metadata (`model_id`, `version`, `path`, `size_bytes`, `downloaded_at`)

**Tables that already exist and are kept:**
- Knowledge graph tables (`dishes`, `ingredients`, `dish_ingredients`, `dishes_fts`) -- already SQLite, bundled as `food-knowledge.db`

## Patterns to Follow

### Pattern 1: Zustand + SQLite Reactive Store

Zustand stores become thin reactive caches over op-sqlite. SQLite is the source of truth. This avoids the common mistake of dual state where the store and database drift.

```typescript
// src/data/db.ts
import { open } from '@op-engineering/op-sqlite';

export const db = open({ name: 'foodtracker.db' });

// src/store/useFoodLogStore.ts (refactored)
import { create } from 'zustand';
import { db } from '../data/db';

interface FoodLogState {
  entries: FoodEntry[];
  loadTodayEntries: () => Promise<void>;
  addEntry: (entry: Omit<FoodEntry, 'id'>) => Promise<string>;
}

export const useFoodLogStore = create<FoodLogState>((set, get) => ({
  entries: [],

  loadTodayEntries: async () => {
    const today = new Date().toISOString().split('T')[0];
    const result = await db.executeAsync(
      'SELECT * FROM food_entries WHERE entry_date = ? ORDER BY created_at DESC',
      [today]
    );
    set({ entries: result.rows._array });
  },

  addEntry: async (entry) => {
    const id = crypto.randomUUID();
    await db.executeAsync(
      `INSERT INTO food_entries (id, meal_type, entry_date, total_calories, ...)
       VALUES (?, ?, ?, ?, ...)`,
      [id, entry.mealType, entry.entryDate, entry.totalCalories, ...]
    );
    await get().loadTodayEntries(); // refresh cache
    return id;
  },
}));
```

### Pattern 2: Inference Router with Device Capability Detection

Route to the best available pipeline based on device capabilities detected at runtime.

```typescript
// src/services/inference/router.ts
import { Platform } from 'react-native';

interface DeviceCapabilities {
  totalRam: number;       // MB
  availableRam: number;   // MB
  hasNpu: boolean;
  npuTops: number;
  geminiNanoAvailable: boolean;
  installedModels: string[];
}

type InferenceTier = 'yolo-only' | 'yolo+vlm-small' | 'yolo+vlm-mid' | 'yolo+vlm-flagship';

function selectTier(caps: DeviceCapabilities): InferenceTier {
  if (caps.totalRam >= 8192 && caps.installedModels.includes('gemma-3n-e2b')) {
    return 'yolo+vlm-flagship';
  }
  if (caps.totalRam >= 6144 && caps.installedModels.includes('moondream-0.5b')) {
    return 'yolo+vlm-mid';
  }
  if (caps.totalRam >= 4096 && caps.installedModels.includes('smolvlm-256m')) {
    return 'yolo+vlm-small';
  }
  return 'yolo-only';
}
```

### Pattern 3: Bursty Batch Processing (Thermal Management)

Process photos in short bursts with cooldown to avoid thermal throttling.

```typescript
// src/services/inference/batchProcessor.ts
const BURST_SIZE = 5;
const COOLDOWN_MS = 2500;

async function processBatch(photos: Photo[], onProgress: (i: number) => void) {
  const results: InferenceResult[] = [];

  for (let i = 0; i < photos.length; i += BURST_SIZE) {
    const burst = photos.slice(i, i + BURST_SIZE);
    const burstResults = await Promise.all(
      burst.map(photo => inferenceService.processPhoto(photo))
    );
    results.push(...burstResults);
    onProgress(Math.min(i + BURST_SIZE, photos.length));

    // Cooldown between bursts (skip if last burst)
    if (i + BURST_SIZE < photos.length) {
      await new Promise(resolve => setTimeout(resolve, COOLDOWN_MS));
    }
  }

  return results;
}
```

### Pattern 4: Pluggable Sync Adapter

Abstract sync behind an interface so providers can be swapped without touching business logic.

```typescript
// src/services/sync/types.ts
interface SyncAdapter {
  readonly provider: 'google-drive' | 'icloud' | 'webdav';
  configure(): Promise<void>;
  push(changeset: SyncChangeset): Promise<SyncResult>;
  pull(): Promise<SyncChangeset>;
  isAvailable(): Promise<boolean>;
}

// src/services/sync/googleDrive.ts
class GoogleDriveSyncAdapter implements SyncAdapter {
  readonly provider = 'google-drive';
  // ... implementation using react-native-cloud-storage
}
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Dual State (Zustand + SQLite Out of Sync)

**What:** Updating Zustand store optimistically and then writing to SQLite, creating windows where UI shows data that the database does not reflect.
**Why bad:** Crashes, force-quits, or failed writes leave the store and DB inconsistent. On restart, the SQLite state wins and the user loses data.
**Instead:** Write to SQLite first, then refresh the Zustand cache from SQLite. Use `executeAsync` to keep the UI responsive. Accept the tiny latency cost (op-sqlite is 8-9x faster than alternatives; sub-millisecond for single writes).

### Anti-Pattern 2: Loading All ML Models at Startup

**What:** Eagerly loading YOLO + VLM + OCR models into memory on app launch.
**Why bad:** VLM alone can take 500MB-2GB RAM. Loading all models causes OOM on budget devices, slow cold start.
**Instead:** Lazy-load models on first use. YOLO loads on first photo process. VLM loads only when needed and confidence is low. OCR loads only when scale is detected. Release VLM after processing is complete.

### Anti-Pattern 3: Continuous Gallery Scanning

**What:** Running a background timer that constantly polls expo-media-library for new photos.
**Why bad:** Battery drain, thermal issues, iOS will kill background processes. expo-media-library has no push-based change notification API.
**Instead:** Scan on app foreground (check `lastScanTimestamp`), provide a manual "Scan Gallery" button, and optionally scan on a conservative interval (every 30 min when foregrounded). On iOS, there is no background photo access capability.

### Anti-Pattern 4: Using React Native Bridge for ML Inference Hot Path

**What:** Passing image data through the old React Native bridge (serialization) for each inference call.
**Why bad:** Bridge serialization adds 10-50ms per image. For batch processing of 20 photos, that is 200-1000ms of pure overhead.
**Instead:** All ML libraries chosen (react-native-fast-tflite, llama.rn, op-sqlite) use JSI for zero-copy direct native calls. New Architecture is already enabled. Ensure no bridge-based alternatives sneak in.

## Suggested Build Order with Dependency Rationale

The build order is driven by dependency chains: later modules depend on earlier ones being functional.

### Phase 1: Foundation (Must Be First)

**Objective:** Replace the data layer. Nothing else can be built until local storage works.

| Step | Module | Depends On | Rationale |
|------|--------|------------|-----------|
| 1.1 | op-sqlite setup + database wrapper | None | Every other module writes to or reads from SQLite |
| 1.2 | SQLite schema + migration system | 1.1 | Tables must exist before data can be stored |
| 1.3 | Refactor useFoodLogStore to SQLite | 1.1, 1.2 | Core data access pattern established |
| 1.4 | Remove API client layer | None (parallel) | Clean break from cloud architecture |
| 1.5 | Expo prebuild + config plugins setup | None (parallel) | Required for native module development |

**Exit criteria:** App boots, reads/writes food entries to local SQLite, no API calls.

### Phase 2: Core ML Pipeline

**Objective:** On-device food detection -- the primary value proposition.

| Step | Module | Depends On | Rationale |
|------|--------|------------|-----------|
| 2.1 | react-native-fast-tflite integration | 1.5 (prebuild) | Native module for YOLO |
| 2.2 | YOLO model loading + inference wrapper | 2.1 | Load .tflite models, run inference, parse output tensors |
| 2.3 | Inference Router (device capability detection) | 2.1 | Determine which models the device can run |
| 2.4 | YOLO pipeline (binary -> detect -> classify) | 2.2, 2.3 | Three-stage food detection |
| 2.5 | HomeScreen integration (real inference) | 2.4, 1.3 | Replace mock "process photos" with real pipeline |
| 2.6 | Model export scripts (Python -> CoreML/TFLite) | Training scripts (existing) | Export trained YOLO models for mobile |

**Exit criteria:** Pick a photo, YOLO detects food items with bounding boxes, results saved to SQLite and displayed in UI.

### Phase 3: Nutrition Resolution

**Objective:** Turn detected food items into macro/nutrient data.

| Step | Module | Depends On | Rationale |
|------|--------|------------|-----------|
| 3.1 | Bundle USDA FDC SQLite database | 1.1 (op-sqlite) | Nutrition data must be queryable |
| 3.2 | NutritionService (ingredient -> nutrients) | 3.1, knowledge graph | Query bundled DBs for nutrient values |
| 3.3 | Port portion_estimator.py to TypeScript | None (pure logic) | Weight estimation for detected items |
| 3.4 | End-to-end pipeline: photo -> detection -> portion -> nutrition | 2.4, 3.2, 3.3 | Complete flow from photo to logged macros |
| 3.5 | DiaryScreen with real data | 3.4, 1.3 | Display logged entries with macro breakdown |

**Exit criteria:** Photo produces a complete food entry with per-ingredient macros stored in SQLite and shown in diary.

### Phase 4: Enhanced Detection + Gallery

**Objective:** Add VLM for complex dishes, scale OCR, gallery scanning.

| Step | Module | Depends On | Rationale |
|------|--------|------------|-----------|
| 4.1 | llama.rn integration | 1.5 (prebuild) | Native module for VLM |
| 4.2 | VLM pipeline (load GGUF, multimodal inference) | 4.1 | SmolVLM-256M as initial model |
| 4.3 | Inference Router update (add VLM tier) | 4.2, 2.3 | Route to VLM when YOLO confidence is low |
| 4.4 | Scale OCR pipeline (7-segment TFLite model) | 2.1 | Reuses react-native-fast-tflite |
| 4.5 | Gallery scanner service | expo-media-library | Discover food photos from gallery |
| 4.6 | Photo deduplication (pHash) | 4.5 | Group duplicate angles into meal events |
| 4.7 | Model manager (download, cache, version) | expo-file-system | On-demand model delivery |
| 4.8 | Batch processing with thermal management | 2.4, 4.5 | Process gallery batch with bursty pattern |

**Exit criteria:** VLM refines low-confidence YOLO detections, scale reading works, gallery scanning discovers food photos.

### Phase 5: UX + Polish

**Objective:** Production-quality user experience.

| Step | Module | Depends On | Rationale |
|------|--------|------------|-----------|
| 5.1 | Editable meals (correct ingredients, portions) | 3.5, 1.3 | Users must be able to fix AI mistakes |
| 5.2 | Manual food search + add | 3.2 | Fallback when AI fails |
| 5.3 | Saveable recipes with reuse | 1.3 | Repeat meals without re-scanning |
| 5.4 | Configurable UX modes (zero-effort, confirm, guided) | 3.4 | Match user preference for AI autonomy |
| 5.5 | End-of-day summary notification | 4.5, 3.5 | Prompt review of auto-discovered meals |
| 5.6 | Container weight management | 4.4 | Tare weight for scale accuracy |
| 5.7 | Device capability display ("Basic/Enhanced" indicator) | 2.3 | Transparency about what the device supports |

### Phase 6: Sync + Distribution

**Objective:** Cloud backup, model delivery, distribution readiness.

| Step | Module | Depends On | Rationale |
|------|--------|------------|-----------|
| 6.1 | react-native-cloud-storage integration | 1.5 (prebuild) | Native module for sync |
| 6.2 | Google Drive sync adapter | 6.1, 1.3 | MVP sync: file-level SQLite backup |
| 6.3 | iCloud sync adapter | 6.1 | iOS users expect iCloud |
| 6.4 | Sync UI (settings, status, conflict display) | 6.2 | User controls sync |
| 6.5 | Play for On-Device AI integration | 4.7 | Android model delivery with device targeting |
| 6.6 | iOS On-Demand Resources | 4.7 | iOS equivalent of Play AI packs |
| 6.7 | Gemini Nano adapter (opportunistic) | Platform APIs | Zero-cost inference on supported devices |

## Dependency Graph (Critical Path)

```
op-sqlite setup (1.1) ──> Schema (1.2) ──> Store refactor (1.3)
                                                    |
                                                    v
expo prebuild (1.5) ──> tflite integration (2.1) ──> YOLO pipeline (2.4)
                    |                                       |
                    |                                       v
                    +──> llama.rn (4.1) ──> VLM (4.2)    Photo->Detect->Portion->Nutrition (3.4)
                    |                                       |
                    +──> cloud-storage (6.1)                v
                                                      Diary UI (3.5) ──> Edit/Search (5.x)
```

**Critical path:** 1.1 -> 1.2 -> 1.3 -> 2.4 -> 3.4 -> 3.5

The critical path is: database foundation -> YOLO pipeline -> nutrition resolution -> diary UI. This is 6 sequential dependencies that block meaningful user-facing functionality. Parallelize where possible: prebuild setup (1.5) runs in parallel with schema work; portion estimator port (3.3) runs in parallel with nutrition DB bundling (3.1).

## Scalability Considerations

| Concern | 100 entries | 10K entries | 100K entries |
|---------|-------------|-------------|--------------|
| SQLite query speed | Sub-ms | 1-5ms with indexes | 5-20ms with indexes; consider pagination |
| Photo storage | ~200MB | ~20GB | ~200GB (unlikely; device storage limit) |
| Gallery scan time | <1s | 5-10s | Paginate, scan incrementally |
| Sync payload | <1MB JSON | 10-50MB | Must chunk uploads; consider delta sync |
| App cold start | <500ms | <500ms (lazy queries) | <500ms (lazy queries) |
| Memory (Zustand cache) | Negligible | Load only current day/week | Must paginate; never load all entries |

## Sources

- [op-sqlite Installation Docs](https://op-engineering.github.io/op-sqlite/docs/installation/)
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite)
- [llama.rn GitHub](https://github.com/mybigday/llama.rn)
- [react-native-cloud-storage Docs](https://react-native-cloud-storage.oss.kuatsu.de/)
- [Expo CNG Documentation](https://docs.expo.dev/workflow/continuous-native-generation/)
- [Expo Custom Native Code](https://docs.expo.dev/workflow/customizing/)
- [Expo SDK 54 Changelog](https://expo.dev/changelog/sdk-54)
- [Expo MediaLibrary API](https://docs.expo.dev/versions/latest/sdk/media-library/)
- [llama.rn npm](https://www.npmjs.com/package/llama.rn) -- v0.11.2, multimodal vision support via mmproj
- [Run Gemma and VLMs on mobile with llama.cpp](https://farmaker47.medium.com/run-gemma-and-vlms-on-mobile-with-llama-cpp-dbb6e1b19a93)
- [LLM Inference on Edge (Hugging Face)](https://huggingface.co/blog/llm-inference-on-edge)
- [Callstack: LLM Inference On-Device in React Native](https://www.callstack.com/events/llm-inference-on-device-in-react-native-the-practical-aspects)
- ADR-005: Local-First Architecture (internal)
- Research docs 002-006 (internal)
