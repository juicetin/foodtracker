# Stack Research: Local-First Architecture Pivot

**Domain:** Local-first AI food tracking -- on-device ML, bundled data, optional cloud sync
**Researched:** 2026-03-12
**Confidence:** MEDIUM-HIGH (core libraries verified via npm/GitHub; some integration points unverified in combination)

**Scope:** This document covers ONLY the new libraries needed for the local-first pivot (ADR-005). It does not re-document the existing stack (React Native 0.81.5, Expo ~54.0.33, Zustand, React Navigation, etc.) which is carried forward unchanged.

---

## Recommended Stack

### 1. Local Database (replaces PostgreSQL)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| @op-engineering/op-sqlite | ^15.2.5 | All local structured data (user data, food entries, recipes, history) | 8-9x faster than expo-sqlite in batch operations per @craftzdog benchmarks. JSI-based with zero-copy. Synchronous and asynchronous APIs. WAL mode support. Used by PowerSync's React Native SDK internally. Expo compatible via `npx expo install`. Requires `npx expo prebuild --clean` (CNG). **HIGH confidence** -- verified via npm (15.2.5 published Feb 2026). |
| drizzle-orm | latest | Type-safe query builder and schema management over op-sqlite | Official op-sqlite integration (`drizzle-orm/op-sqlite`). Type-safe queries, migrations via `useMigrations` hook, schema-as-code. Avoids raw SQL string sprawl across the codebase. Zero runtime overhead -- compiles to SQL at build time. **HIGH confidence** -- verified via drizzle docs. |
| drizzle-kit | latest (dev) | Migration generation and schema diffing | Generates SQL migration files from TypeScript schema changes. `dialect: 'sqlite'`, `driver: 'expo'` in drizzle.config.ts. Dev-only dependency. **HIGH confidence** -- official companion to drizzle-orm. |

**Why not expo-sqlite:** expo-sqlite (v55.0.9) has closed the gap significantly and is simpler to set up. However, op-sqlite remains faster for batch operations and large dataset queries -- both critical for nutrition DB lookups (50-80MB database) and batch gallery scanning. op-sqlite also has better PowerSync integration if row-level sync is needed later. The performance difference matters when querying a bundled nutrition database with 300K+ food entries.

**Why not WatermelonDB:** WatermelonDB adds its own abstraction layer over SQLite with lazy loading and observable queries. This is valuable for apps with complex sync requirements, but adds unnecessary complexity for a single-user local-first app. op-sqlite + drizzle gives us direct SQLite control without the overhead.

### 2. On-Device ML Inference (YOLO Pipeline)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| react-native-fast-tflite | ^2.0.0 | Run YOLO TFLite/LiteRT models for food detection | JSI-based, zero-copy ArrayBuffers. v2.0.0 (Jan 2026) upgrades to LiteRT 1.4.0. GPU delegates: CoreML (iOS), OpenGL (Android). Expo config plugin for CoreML and GPU library configuration. New Architecture compatible (required). Loads `.tflite` files at runtime. **HIGH confidence** -- verified via npm and GitHub releases. |
| react-native-vision-camera | ^4.7.3 | Camera capture for scale OCR (live viewfinder) | Frame processor plugin system runs synchronous native ML inference per-frame. Pairs with react-native-fast-tflite for live camera detection (scale reading). Expo config plugin available. Same author (mrousavy) as fast-tflite -- tight integration. **HIGH confidence** -- verified via npm. |
| vision-camera-resize-plugin | latest | Frame preprocessing (resize, crop, pixel format conversion) | SIMD-accelerated frame resizing before TFLite inference. Required companion for VisionCamera + TFLite pipeline. Converts YUV camera frames to RGB tensors. **MEDIUM confidence** -- version not independently pinned. |

**Expo integration:** Both react-native-fast-tflite and react-native-vision-camera ship Expo config plugins. Add to `app.json` plugins array. Requires CNG (`npx expo prebuild --clean`) -- not compatible with Expo Go. Use Expo Dev Client for development.

**New Architecture:** The project already has `"newArchEnabled": true` in app.json. react-native-fast-tflite v2.0.0 requires New Architecture (Fabric + TurboModules). react-native-vision-camera v4.7.3 supports New Architecture. No conflicts.

**CoreML vs LiteRT strategy:** Export YOLO models to both `.mlmodel` (CoreML for iOS) and `.tflite` (LiteRT for Android). react-native-fast-tflite handles both formats. CoreML leverages Apple Neural Engine; LiteRT uses vendor NPU delegates (Qualcomm QNN, MediaTek NeuroPilot). An inference router in JS selects the right model file per platform.

### 3. On-Device VLM Inference (Optional Enhanced Mode)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| llama.rn | ^0.11.2 | Run GGUF VLMs on-device for complex dish identification | React Native binding of llama.cpp. Supports multimodal vision (image+text) via mmproj files. GPU offloading on iOS (Metal) and Android (OpenCL on Adreno 700+). Expo config plugin via expo-build-properties. Supports SmolVLM, Qwen3-VL, Gemma 3n. GGUF format is the standard for quantized LLM/VLM distribution. **HIGH confidence** -- verified via npm (0.11.2 published Mar 2026). |

**Why llama.rn over alternatives:**

| Option | Verdict | Reason |
|--------|---------|--------|
| llama.rn (mybigday) | **Use this** | Most mature RN binding of llama.cpp. Active development (weekly releases). Vision/multimodal support merged. Expo plugin. Supports all target VLMs (SmolVLM, Gemma 3n). |
| @pocketpalai/llama.rn | Do not use | Fork of llama.rn with PocketPal-specific changes. Lags behind upstream. No benefit for our use case. |
| react-native-executorch | Not yet | v0.7.2 (pre-1.0). VLM vision support on the roadmap but not yet shipped. Revisit when 1.0 launches. Promising for ExecuTorch-optimized models but immature today. |
| @callstackincubator/ai | Do not use | Wraps llama.rn with Vercel AI SDK compatibility. Adds abstraction we don't need. Use llama.rn directly. |
| LiteRT-LM (Google) | Android only | Official runtime for Gemma 3n on Android. No React Native binding exists. Would require a custom Expo native module in Kotlin. Consider only if Gemma 3n performance via llama.rn is insufficient. |

**VLM model tiers (delivered via asset packs, not bundled):**

| Model | GGUF Size | Runtime RAM | Target Devices | llama.rn Support |
|-------|-----------|-------------|----------------|------------------|
| SmolVLM-256M (Q4) | ~100MB | ~300-500MB | Budget (<=4GB RAM) | Yes (GGUF + mmproj) |
| Moondream 0.5B (INT4) | ~375MB | ~816MB | Mid-range (6GB) | Yes (GGUF) |
| Gemma 3n E2B | ~2GB | ~2GB | Flagship (8GB+) | Yes (GGUF) |

### 4. Bundled Nutrition Database

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Pre-populated SQLite database | N/A | USDA FDC Foundation + SR Legacy nutrition data | Bundled as a `.db` file (~50-80MB compressed). Opened read-only via op-sqlite. Queried with drizzle-orm using the same type-safe API as user data. Shipped as a fast-follow asset pack (auto-downloads after install). The knowledge graph build pipeline (`export_mobile.py`) already produces this format. **HIGH confidence** -- architecture validated by existing export tooling. |

**No new library needed.** op-sqlite opens the bundled database as a second connection. The existing knowledge graph export pipeline (`scripts/knowledge-graph/export_mobile.py`) already produces mobile-friendly SQLite. Regional databases (AFCD, CoFID, CIQUAL) delivered as optional on-demand asset packs.

### 5. Cloud Sync (Optional)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| react-native-cloud-storage | ^2.3.0 | Google Drive and iCloud file-level sync | Unified API for both Google Drive (REST, all platforms) and iCloud (native, iOS only). Google Drive uses `drive.appdata` scope (narrow permission -- hidden folder, no full Drive access). Zero backend dependencies. Supports binary files (v2.3.0, Jun 2025). Works with Expo via config plugin. **HIGH confidence** -- verified via GitHub releases (v2.3.0). |

**Sync strategy (per ADR-005):**

1. **Phase 1 (MVP):** File-level sync via react-native-cloud-storage. Export SQLite database as a file, upload to Google Drive appDataFolder or iCloud. Download and replace on other device. Simple backup/restore.
2. **Phase 2 (if multi-device needed):** Evaluate PowerSync self-hosted Open Edition for row-level sync. But only if users actually use multiple devices simultaneously -- file-level LWW is sufficient for single-user append-heavy food logs.

**Conflict resolution:** Last-write-wins (LWW) with timestamps. Implemented in application logic, not a library. Soft-delete with `is_deleted` + `deleted_at`. No CRDT library needed.

### 6. Asset Delivery (Models + Databases)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| expo-play-asset-delivery | ^1.2.3 | Play Asset Delivery for Android (AI packs, nutrition DBs) | Expo config plugin for configuring asset packs in app.config.js. Supports install-time, fast-follow, and on-demand delivery modes. Required for Play for On-Device AI integration. `npx expo prebuild --clean` after config changes. **MEDIUM confidence** -- last published 2 years ago, but Play Asset Delivery API is stable. May need fork/update for Play for On-Device AI (PODAI) specific features like device targeting. |
| expo-file-system | ^19.0.21 | Download/manage model files and asset packs on iOS | Already installed. Use for On-Demand Resources on iOS (Apple's equivalent of Play Asset Delivery). Download VLM models on-demand, cache in app-local storage. **HIGH confidence** -- already in use. |

**Play for On-Device AI (PODAI) integration note:** PODAI extends Play Asset Delivery with device targeting by RAM, chipset, and device model. The `expo-play-asset-delivery` plugin handles standard asset packs. PODAI-specific device targeting may require a custom Expo config plugin or native module extension. This is a known gap that needs phase-specific research during implementation.

**iOS equivalent:** Apple's On-Demand Resources (ODR) and Background Assets API. No Expo plugin exists. Will require a custom Expo native module wrapping `NSBundleResourceRequest` for ODR. This is straightforward (< 100 lines of Swift) but is custom work.

### 7. Gemini Nano / AICore (Opportunistic)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Custom Expo Native Module (Kotlin) | N/A | ML Kit GenAI Prompt API for Gemini Nano multimodal inference | No React Native binding exists for ML Kit GenAI APIs. Requires a thin Kotlin native module (~200 lines) wrapping `GenerativeModel` from `com.google.mlkit:genai`. Accessed via Expo Modules API. Foreground-only, 1024 token limit, battery-quotaed. **MEDIUM confidence** -- API is alpha (Prompt API) / beta (high-level APIs). React Native integration is custom work. |

**Why build a custom module rather than wait for a library:**
- ML Kit GenAI Prompt API is alpha -- no third-party RN wrapper is likely to appear soon
- The wrapper is thin: `initialize`, `generateContent(image, prompt)`, `close`
- Gemini Nano is a bonus on supported devices, not a critical path
- Zero app size cost (model is system-provided via AICore)

**Not needed for MVP.** Add after core YOLO + VLM pipeline is working.

### 8. Gallery Scanning & Background Processing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| expo-media-library | ^18.2.1 | Gallery access, photo querying, EXIF metadata | Already installed. `getAssetsAsync` for paginated gallery queries sorted by creation time. `getAssetInfoAsync` for EXIF (GPS, timestamp, camera). `addListener` for new-photo events. Requires `ACCESS_MEDIA_LOCATION` Android permission for GPS. **HIGH confidence** -- already in use. |
| expo-background-task | ^1.0.10 | Periodic background gallery scanning | Replaces deprecated expo-background-fetch. Uses WorkManager (Android, min 15-min interval) and BGTaskScheduler (iOS). Single-worker model. Register via expo-task-manager. Does NOT work in Expo Go -- requires Dev Client. **HIGH confidence** -- verified via npm and Expo docs. |
| expo-task-manager | latest | Task registration for background processing | Required companion to expo-background-task. `defineTask` and `isTaskRegisteredAsync`. **HIGH confidence** -- official Expo package. |

### 9. Image Processing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| expo-image-manipulator | ^55.0.9 | Resize/crop images before ML inference | Already part of Expo SDK. Resize gallery photos to model input dimensions (e.g., 640x640 for YOLO) before inference. Crop scale display region for OCR. No additional install needed beyond Expo SDK. **HIGH confidence** -- official Expo package. |

### 10. Scale OCR (Custom 7-Segment Model)

No new library needed beyond react-native-fast-tflite (already listed above). The custom 7-segment OCR model is a TFLite model (17KB-5MB) loaded via the same react-native-fast-tflite runtime used for YOLO. The camera pipeline uses react-native-vision-camera with a frame processor for live viewfinder detection.

**Training pipeline (Python, dev-only):**
- Training data: Roboflow Universe 7-segment datasets (948+ annotated images)
- Model: Small CNN or YOLO-based digit detector
- Export: TFLite INT8 via ultralytics or direct Keras/TF export
- Post-processing: digit+decimal whitelist, range validation (0-5000g)

---

## Supporting Libraries (Already Installed, Carry Forward)

| Library | Version | Purpose | Notes for Local-First |
|---------|---------|---------|----------------------|
| zustand | ^5.0.11 | State management | Use for ML pipeline state (scan queue, processing status, inference results, sync status). No changes needed. |
| expo-file-system | ^19.0.21 | File I/O | Photo storage, model file management, asset pack downloads. No changes needed. |
| expo-media-library | ^18.2.1 | Gallery access | Already installed. Core to gallery scanning feature. |
| expo-image-picker | ^17.0.10 | Manual photo selection | Already installed. Fallback when gallery scan misses a photo. |
| @react-native-async-storage/async-storage | ^2.2.0 | Key-value settings | Already installed. Use for app settings, last sync timestamp, device capability flags. NOT for structured data (use op-sqlite). |

---

## Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Expo Dev Client | Test native modules on device | Required -- op-sqlite, react-native-fast-tflite, llama.rn, vision-camera all require native code. Cannot use Expo Go. |
| npx expo prebuild | Generate native projects (CNG) | Run after adding any new native dependency. Use `--clean` flag when changing config plugins. |
| npx expo run:ios / run:android | Build and run dev client | Local builds for development. EAS Build for CI/distribution. |

---

## Installation

```bash
# === NEW DEPENDENCIES (Local-First Pivot) ===

# Local database (replaces PostgreSQL)
npx expo install @op-engineering/op-sqlite
npm install drizzle-orm
npm install -D drizzle-kit

# On-device ML inference (YOLO pipeline)
npx expo install react-native-fast-tflite
npx expo install react-native-vision-camera
npm install vision-camera-resize-plugin

# On-device VLM inference (optional enhanced mode)
npm install llama.rn

# Cloud sync (optional)
npm install react-native-cloud-storage

# Asset delivery (Android)
npm install expo-play-asset-delivery

# Background processing
npx expo install expo-background-task expo-task-manager

# Image processing (already part of Expo SDK, verify installed)
npx expo install expo-image-manipulator

# === REGENERATE NATIVE PROJECTS ===
npx expo prebuild --clean
```

```bash
# Python (model training/export -- dev machine only, unchanged)
pip install ultralytics>=8.4.8
pip install coremltools>=8.0
```

### app.json Plugin Configuration

```json
{
  "expo": {
    "plugins": [
      [
        "react-native-fast-tflite",
        {
          "enableAndroidGpuLibraries": true
        }
      ],
      [
        "react-native-vision-camera",
        {
          "cameraPermissionText": "FoodTracker needs camera access to read kitchen scale displays",
          "enableFrameProcessors": true
        }
      ],
      [
        "llama.rn",
        {
          "forceCxx20": true,
          "enableOpenCL": true
        }
      ],
      [
        "expo-play-asset-delivery",
        {
          "assetPacks": {
            "nutrition-db": {
              "path": "./assets/nutrition",
              "deliveryMode": "fast-follow"
            },
            "vlm-budget": {
              "path": "./assets/models/smolvlm",
              "deliveryMode": "on-demand"
            }
          }
        }
      ]
    ]
  }
}
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Local DB | op-sqlite + drizzle | expo-sqlite | expo-sqlite is simpler but slower for batch operations. Nutrition DB has 300K+ rows; op-sqlite's 8-9x batch advantage matters. expo-sqlite also lacks WAL mode control needed for concurrent read/write during sync. |
| Local DB | op-sqlite + drizzle | WatermelonDB | WatermelonDB adds lazy-loading and observable queries -- overkill for single-user app. Its sync protocol requires building a backend endpoint, which conflicts with our no-backend architecture. |
| Local DB ORM | drizzle-orm | TypeORM / Prisma | TypeORM has poor React Native support. Prisma doesn't support SQLite on mobile. Drizzle is purpose-built for this: zero runtime, type-safe, official op-sqlite adapter. |
| VLM runtime | llama.rn | react-native-executorch | Executorch RN is v0.7.2 (pre-1.0). VLM vision support not yet shipped. llama.rn has production-ready multimodal support today. Revisit when executorch hits 1.0. |
| VLM runtime | llama.rn | Custom LiteRT-LM native module | LiteRT-LM is Android-only and has no RN binding. Would require maintaining two separate native modules (Kotlin for LiteRT-LM, Swift for CoreML). llama.rn is cross-platform with a single API. |
| Cloud sync | react-native-cloud-storage | PowerSync | PowerSync requires a backend (Postgres). Our architecture has no backend. react-native-cloud-storage works serverless. If row-level sync is later needed, PowerSync self-hosted is the upgrade path. |
| Cloud sync | react-native-cloud-storage | Custom Google Drive REST API calls | react-native-cloud-storage wraps the REST API with auth handling, token refresh, and iCloud support. Writing raw REST calls is error-prone and duplicates work the library handles. |
| Asset delivery | expo-play-asset-delivery | Manual APK expansion files | Expansion files (OBBs) are legacy. Play Asset Delivery is the modern replacement with delta patching and device targeting. |
| YOLO inference | react-native-fast-tflite | @infinitered/react-native-mlkit | ML Kit object detection is generic (not food-specific). Cannot load custom YOLO models. ML Kit is for barcode/text/face, not custom detection. |
| Camera | react-native-vision-camera | expo-camera | expo-camera lacks frame processor support needed for real-time scale OCR. VisionCamera's frame processor pipeline is essential for live inference. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Express.js backend | Removed per ADR-005. No server needed. | All logic on-device via op-sqlite + react-native-fast-tflite + llama.rn. |
| PostgreSQL | Removed per ADR-005. No remote database. | op-sqlite for local structured data. |
| Google ADK agents | Removed per ADR-005. No cloud agent framework. | Direct on-device inference pipeline (YOLO -> VLM -> nutrition lookup). |
| NNAPI delegate | Deprecated in Android 15. | LiteRT with vendor NPU delegates (Qualcomm QNN, MediaTek NeuroPilot) via react-native-fast-tflite. |
| Realm / MongoDB Atlas Device Sync | EOL September 2025. SDKs no longer maintained. | op-sqlite + react-native-cloud-storage. |
| CRDTs (Automerge, Yjs) | Overkill for single-user food logs. Automerge requires WASM (not supported in RN). Yjs adds complexity without benefit for record-oriented data. | LWW conflict resolution in application code. |
| ElectricSQL | Pivoted to read-path only (July 2024). Not bidirectional offline-first. | react-native-cloud-storage for sync. |
| expo-background-fetch | Deprecated in favor of expo-background-task as of SDK 53. | expo-background-task ^1.0.10. |
| Expo Go | Does not support native modules (op-sqlite, TFLite, llama.rn, VisionCamera). | Expo Dev Client with `npx expo prebuild`. |
| @kingstinct/react-native-healthkit | Out of scope for v1.0 per PROJECT.md. Do not add. | Defer to post-v1.0. |
| react-native-health-connect | Out of scope for v1.0 per PROJECT.md. Do not add. | Defer to post-v1.0. |
| LangGraph / agent frameworks | No server-side orchestration needed. Pipeline is deterministic (YOLO -> classify -> lookup). | Direct function calls in TypeScript. |
| Depth Anything V2 / depth estimation | Portion estimation uses reference-based approach (known container weights, food density lookup), not monocular depth. | Existing portion estimator module (Python, already validated). |
| Firebase / Supabase / any BaaS | Adds server dependency and potential costs. Conflicts with zero-subscription guarantee. | react-native-cloud-storage for sync to user's own Google Drive / iCloud. |

---

## Version Compatibility Matrix

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| @op-engineering/op-sqlite@15.2.5 | React Native >= 0.71, Expo SDK >= 50 | Requires `npx expo prebuild --clean`. CNG compatible. |
| drizzle-orm (latest) | @op-engineering/op-sqlite >= 11.x | Use `drizzle-orm/op-sqlite` adapter. Configure metro.config.js for `.sql` migration files. |
| react-native-fast-tflite@2.0.0 | React Native >= 0.76 (New Architecture required) | RN 0.81.5 + `newArchEnabled: true` in project -- compatible. Expo config plugin for CoreML/GPU delegates. |
| react-native-vision-camera@4.7.3 | React Native >= 0.73, Expo SDK >= 51 | Frame processors require react-native-worklets-core. Expo config plugin available. |
| llama.rn@0.11.2 | React Native >= 0.71, Expo SDK >= 50 | Expo config plugin via expo-build-properties. OpenCL support for Adreno 700+ GPUs. |
| react-native-cloud-storage@2.3.0 | React Native >= 0.72, Expo SDK >= 49 | Google Drive requires OAuth2 setup. iCloud requires Apple Developer entitlements. |
| expo-play-asset-delivery@1.2.3 | Expo SDK >= 48 | May need updates for Play for On-Device AI device targeting. Standard asset packs work today. |
| expo-background-task@1.0.10 | Expo SDK >= 52 | Requires expo-task-manager. Does NOT work in Expo Go. |
| YOLO models (.tflite) | LiteRT >= 1.4.0, react-native-fast-tflite >= 2.0.0 | Export: `model.export(format='tflite', int8=True)`. |
| YOLO models (.mlmodel) | CoreML >= 6 (iOS 16+) | Export: `model.export(format='coreml')`. YOLO26 is NMS-free. |
| VLM models (.gguf) | llama.rn >= 0.11.0 | Requires corresponding mmproj file for vision models. Set `ctx_shift: false` for VLMs. |

**Critical note on Expo SDK 54:** SDK 54 is the last version where New Architecture can be disabled. Since our project already has `newArchEnabled: true` and all recommended libraries support New Architecture, this is a non-issue. SDK 55+ will mandate New Architecture.

---

## Managed vs Bare Workflow Impact

**This project uses CNG (Continuous Native Generation)** -- native directories are generated from app.json/app.config.js via `npx expo prebuild`. This is the recommended Expo workflow for apps with native dependencies.

**All recommended libraries support CNG via config plugins:**
- op-sqlite: `npx expo install` handles config
- react-native-fast-tflite: ships Expo config plugin
- react-native-vision-camera: ships Expo config plugin
- llama.rn: config via expo-build-properties plugin
- expo-play-asset-delivery: ships Expo config plugin
- expo-background-task: official Expo package

**Custom native modules needed (not covered by existing libraries):**
1. Gemini Nano / AICore wrapper (Kotlin, ~200 lines) -- use Expo Modules API
2. iOS On-Demand Resources wrapper (Swift, ~100 lines) -- use Expo Modules API

Both can be created as local Expo Modules within the project's `modules/` directory without ejecting or leaving CNG.

---

## Dependency Count Impact

**New runtime dependencies being added:** 8 packages
- @op-engineering/op-sqlite, drizzle-orm, react-native-fast-tflite, react-native-vision-camera, vision-camera-resize-plugin, llama.rn, react-native-cloud-storage, expo-play-asset-delivery

**New dev dependencies:** 1 package
- drizzle-kit

**Existing dependencies repurposed (no new install):** 4 packages
- expo-media-library, expo-file-system, expo-image-manipulator, zustand

**Dependencies being removed (no longer needed):** Express.js backend + all its dependencies, PostgreSQL client, Google ADK SDK. These are in separate workspace directories (`backend/`, `services/ai-agent/`) and can be archived.

**Total new runtime deps: 8.** This is lean for the scope of features being added (local DB, ML inference, VLM, cloud sync, asset delivery).

---

## Sources

### HIGH Confidence (verified via official releases/docs)
- [@op-engineering/op-sqlite npm](https://www.npmjs.com/package/@op-engineering/op-sqlite) -- v15.2.5, published Feb 2026
- [op-sqlite GitHub](https://github.com/OP-Engineering/op-sqlite) -- installation docs, Expo compatibility
- [Drizzle ORM op-sqlite docs](https://orm.drizzle.team/docs/connect-op-sqlite) -- official integration guide
- [react-native-fast-tflite npm](https://www.npmjs.com/package/react-native-fast-tflite) -- v2.0.0, Jan 2026
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite) -- Expo config plugin docs
- [react-native-vision-camera npm](https://www.npmjs.com/package/react-native-vision-camera) -- v4.7.3
- [llama.rn npm](https://www.npmjs.com/package/llama.rn) -- v0.11.2, published Mar 2026
- [llama.rn GitHub](https://github.com/mybigday/llama.rn) -- multimodal vision support, Expo plugin
- [react-native-cloud-storage GitHub releases](https://github.com/Kuatsu/react-native-cloud-storage/releases) -- v2.3.0, Jun 2025
- [expo-play-asset-delivery npm](https://www.npmjs.com/package/expo-play-asset-delivery) -- v1.2.3
- [Expo SDK 54 changelog](https://expo.dev/changelog/sdk-54) -- New Architecture, CNG
- [expo-background-task docs](https://docs.expo.dev/versions/latest/sdk/background-task/) -- official Expo documentation
- [PowerSync React Native benchmarks](https://www.powersync.com/blog/react-native-database-performance-comparison) -- op-sqlite vs expo-sqlite

### MEDIUM Confidence (verified via multiple credible sources)
- [ML Kit GenAI Prompt API](https://developers.google.com/ml-kit/genai/prompt/android) -- alpha status, API surface verified
- [Play for On-Device AI docs](https://developer.android.com/google/play/on-device-ai) -- beta, device targeting features
- [expo-play-asset-delivery GitHub](https://github.com/one-am-it/expo-play-asset-delivery) -- community plugin, may need updates for PODAI
- [Expo Modules API](https://docs.expo.dev/modules/overview/) -- custom native module creation

### LOW Confidence (needs phase-specific validation)
- vision-camera-resize-plugin exact version -- not pinned, use latest
- expo-play-asset-delivery compatibility with Play for On-Device AI device targeting -- untested in combination
- llama.rn OpenCL GPU performance on non-Adreno chipsets -- documented for Adreno 700+ only
- Custom Expo native module effort estimates (Gemini Nano wrapper, iOS ODR wrapper) -- based on API surface analysis, not implementation experience

---

*Stack research for: Local-first AI food tracking pivot (ADR-005)*
*Researched: 2026-03-12*
*Supersedes: Previous STACK.md (2026-02-12) which assumed cloud backend*
