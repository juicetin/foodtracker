# Project Research Summary

**Project:** FoodTracker -- Local-First AI Food Tracking
**Domain:** On-device ML food recognition with bundled nutrition data, zero subscription cost
**Researched:** 2026-03-12
**Confidence:** MEDIUM-HIGH
**Supersedes:** Previous SUMMARY.md (2026-02-12, pre-local-first pivot)

## Executive Summary

FoodTracker is a local-first, subscription-free AI food tracker for mobile (React Native + Expo SDK 54). The product differentiates on three axes no competitor occupies simultaneously: zero recurring cost (no server = no subscription), passive gallery scanning (discovers food photos the user already took), and complete on-device privacy (photos never leave the device). The March 2026 MFP acquisition of Cal AI consolidates the subscription AI tracker market, making the subscription-free niche wider and more defensible. Existing training scripts (YOLO models), a knowledge graph with SQLite export, and a validated portion estimation module provide a strong foundation -- the work ahead is integration into a local-first mobile architecture, not greenfield ML research.

The recommended approach is a layered local-first architecture built on op-sqlite (8-9x faster than expo-sqlite for batch operations), react-native-fast-tflite (JSI-based YOLO inference), and llama.rn (on-device VLM for complex dishes). All libraries support Expo CNG via config plugins, meaning no eject to bare workflow is required. The critical path runs: database foundation -> YOLO pipeline -> nutrition resolution -> diary UI. The data layer must be established first because every other module (inference, nutrition lookup, gallery scanning, sync) reads from or writes to SQLite. Cloud sync via react-native-cloud-storage (Google Drive + iCloud) is fully optional and architecturally independent of the inference pipeline.

The top risks are: (1) OOM crashes from VLM inference on the 30-35% of Android devices with 4GB or less RAM -- mitigated by runtime capability detection and tiered model loading with hard RAM gates; (2) background gallery scanning is severely constrained by iOS's 30-second BGTask limit and Android battery restrictions -- mitigated by designing scanning as a foreground-first feature triggered on app open, not a background daemon; (3) bundled SQLite database management across app updates can silently destroy user data or serve stale nutrition info -- mitigated by separating user data and reference data into independent database files with versioned migrations from Day 1. The development workflow must switch from Expo Go to custom dev builds immediately, as every core library (op-sqlite, react-native-fast-tflite, llama.rn) requires native code that Expo Go cannot run.

## Key Findings

### Recommended Stack

The stack adds 8 new runtime dependencies to the existing React Native + Expo codebase. The core pillars are op-sqlite + drizzle-orm for local storage, react-native-fast-tflite for YOLO inference, llama.rn for VLM inference, and react-native-cloud-storage for optional sync. All dependencies use JSI (zero-copy, no bridge serialization) and support New Architecture, which the project already has enabled. The existing backend (Express + PostgreSQL) and AI agent service (Google ADK) are fully superseded and can be archived. See [STACK.md](STACK.md) for version-pinned recommendations and alternatives analysis.

**Core technologies:**
- **op-sqlite + drizzle-orm:** Local structured data (user entries, recipes, history) -- 8-9x faster than expo-sqlite for batch operations, critical for 300K+ row nutrition DB queries
- **react-native-fast-tflite v2.0.0:** YOLO food detection via CoreML (iOS) and LiteRT (Android) -- JSI-based, GPU/NPU delegate support, 5-8ms NPU inference
- **llama.rn v0.11.2:** On-device VLM for complex dish identification -- multimodal vision support, GGUF format, GPU offloading on iOS Metal and Android OpenCL
- **react-native-cloud-storage v2.3.0:** Google Drive appDataFolder + iCloud sync -- zero backend, file-level backup/restore
- **expo-background-task v1.0.10:** Periodic gallery scanning via WorkManager (Android) / BGTaskScheduler (iOS)
- **Bundled USDA FDC SQLite:** ~50-80MB pre-populated nutrition database, delivered as fast-follow asset pack

**Critical version requirements:**
- React Native >= 0.76 with New Architecture enabled (project has 0.81.5 + newArchEnabled: true)
- react-native-fast-tflite v2.0.0 requires New Architecture (mandatory)
- Expo SDK 54 with CNG (no Expo Go)

### Expected Features

The feature landscape is shaped by the local-first constraint: everything must work fully offline. The primary differentiator is passive gallery scanning -- no competitor offers it. See [FEATURES.md](FEATURES.md) for full competitor analysis and feature dependency graph.

**Must have (table stakes):**
- On-device YOLO food detection (every AI tracker has photo recognition)
- Bundled USDA nutrition database with fast text search
- Food diary UI with per-meal and daily macro totals
- Manual food search and entry (essential fallback when AI is wrong)
- Editable meals (users must be able to correct AI mistakes)
- Gallery scanning with manual trigger (primary differentiator, even before background automation)
- Basic photo deduplication via temporal clustering (structural requirement for gallery scanning)
- Portion estimation integrated with on-device pipeline
- Meal confirmation flow with confidence indicators (green/yellow/red)
- Recipe saving and reuse

**Should have (competitive advantage):**
- Zero subscription cost (business model differentiator, enabled by local-first architecture)
- Passive background gallery scanning (zero-effort logging)
- Scale/weight OCR from photos (custom 7-segment model; no competitor does this)
- Container weight learning (auto-subtract tare weights)
- Configurable UX modes (zero-effort / confirm-only / guided-edit)
- End-of-day summary notification
- Optional VLM for complex/mixed dish identification
- Google Drive + iCloud sync
- Hidden ingredient inference via knowledge graph

**Defer (v2+):**
- Apple Health / Google Fit integration
- Barcode scanning (competes with MFP's 20M food DB -- wrong battlefield)
- 3D LiDAR portion estimation (iPhone Pro only)
- Multi-region nutrition databases (USDA covers most English-speaking users)
- AI coaching / adaptive TDEE (different product category)
- Social features (requires server infrastructure, wrong product)
- Real-time camera food detection (battery drain, thermal issues, no user value over gallery scan)

### Architecture Approach

The architecture is a layered local-first system replacing the cloud backend with on-device services. The presentation layer (React Navigation, existing screens) is kept with extensions. The state layer (Zustand) is refactored from in-memory to SQLite-backed reactive caches. A new service layer replaces the API client with InferenceService (YOLO/VLM/OCR pipelines), NutritionService (bundled DB queries), GalleryService (photo discovery + dedup), and SyncService (pluggable adapter pattern for cloud providers). The data layer moves from PostgreSQL to two separate SQLite databases: a read-write user database and a read-only bundled nutrition database. See [ARCHITECTURE.md](ARCHITECTURE.md) for component inventory, data flow diagrams, and schema migration plan.

**Major components:**
1. **InferenceService** -- orchestrates the YOLO -> classify -> VLM (optional) -> scale OCR (if detected) pipeline with device capability-based routing
2. **NutritionService** -- queries bundled USDA FDC and knowledge graph databases, calculates per-ingredient and total macros
3. **GalleryService** -- discovers food photos via expo-media-library, extracts EXIF metadata, deduplicates via temporal/visual clustering
4. **SyncService** -- pluggable adapter pattern (GoogleDriveSyncAdapter, iCloudSyncAdapter) for file-level backup/restore with LWW conflict resolution
5. **Database layer (op-sqlite)** -- user data DB (read-write, migrations) + nutrition DB (read-only, version-gated replacement on updates)

**Key patterns:**
- Zustand stores as reactive caches over SQLite (write to DB first, then refresh cache)
- Inference router selects pipeline tier by runtime device capability detection (RAM, NPU, installed models)
- Bursty batch processing with thermal cooldown (5 photos, 2.5s pause) for gallery scanning
- Lazy model loading (load on first use, unload after completion) to prevent OOM

### Critical Pitfalls

17 pitfalls identified across 4 severity levels. The top 5 are highlighted below. See [PITFALLS.md](PITFALLS.md) for full analysis with detection signs, prevention strategies, and recovery costs.

1. **Expo Go cannot run native ML modules (CRITICAL)** -- Switch to custom dev builds on Day 1. Run `npx expo prebuild` and verify all config plugins compile together on both platforms before writing any feature code. Never use Expo Go again.
2. **Bundled SQLite DB fails to copy/update across app versions (CRITICAL)** -- Separate user data and nutrition data into independent database files. Add `"db"` to Metro's `resolver.assetExts`. Implement version-gated replacement of the nutrition DB that never touches user data.
3. **VLM inference OOM-kills on 30-35% of Android devices (CRITICAL)** -- Runtime capability detection before loading any model. Hard RAM thresholds: no VLM below 4GB, SmolVLM at 4-6GB, Moondream at 6-8GB, Gemma 3n at 8GB+. Lazy load, immediate unload after inference.
4. **Schema migrations destroy user data on app updates (CRITICAL)** -- Implement version-tracked migration system (PRAGMA user_version) from Day 1, before any user data exists. Every migration in a transaction. Never drop tables. Test upgrade paths across version gaps.
5. **Background gallery scanning killed by iOS 30-second limit (HIGH)** -- Design scanning as a foreground feature first. Background tasks do lightweight metadata indexing only; inference runs when user opens app. Background batch inference is architecturally infeasible on iOS.

**Pitfall interactions to watch:**
- OOM (Pitfall 3) + Image memory explosion (Pitfall 14) = double memory pressure during batch scanning. Always resize images to model input dimensions before decode.
- Bundled DB failures (Pitfall 2) + Schema migration data loss (Pitfall 4) = catastrophic data loss if both hit. Separate databases + version both independently.
- Background limits (Pitfall 5) + Thermal throttling (Pitfall 10) = background inference is near-useless. Foreground scanning is the only reliable path.

## Implications for Roadmap

Based on combined research, the project naturally divides into 6 phases driven by dependency chains. The critical path is: database foundation -> YOLO pipeline -> nutrition resolution -> diary UI. Phases 4-6 add enhancement layers that are valuable but not blocking for a functional MVP.

### Phase 1: Infrastructure + Data Foundation
**Rationale:** Every other module depends on local SQLite storage. The dev build workflow must be established before any native module work. This phase has zero user-facing features but unblocks everything.
**Delivers:** op-sqlite setup with drizzle-orm, SQLite schema + versioned migrations, Zustand stores refactored to SQLite-backed, Expo CNG with all config plugins verified on both platforms, API client layer and backend code removed.
**Addresses:** Local data storage (P1), prerequisite for all other features.
**Avoids:** Pitfall 1 (Expo Go incompatibility), Pitfall 2 (bundled DB copy failures), Pitfall 4 (schema migration data loss), Pitfall 13 (New Architecture compatibility).

### Phase 2: On-Device Food Detection Pipeline
**Rationale:** The core value proposition -- photo to food identification. Depends on Phase 1 for data persistence and dev build infrastructure.
**Delivers:** react-native-fast-tflite integration, YOLO model loading + inference, inference router with device capability detection, binary food/not-food classifier, multi-class food detection with bounding boxes, model export scripts (Python -> CoreML/TFLite).
**Addresses:** On-device YOLO food detection (P1).
**Avoids:** Pitfall 6 (model loading freezes -- use async loading with loading states), Pitfall 7 (cross-platform output divergence -- build validation test suite), Pitfall 15 (NNAPI fallback -- use LiteRT vendor delegates).

### Phase 3: Nutrition Resolution + Diary
**Rationale:** Turns detected food items into actionable macro data. This is where the app becomes usable. Depends on Phase 2 for detection results and Phase 1 for data storage.
**Delivers:** Bundled USDA FDC database (fast-follow asset pack), NutritionService (ingredient -> nutrient lookup), PortionService (TypeScript port of Python module), end-to-end photo -> detection -> portion -> nutrition pipeline, DiaryScreen with real data, manual food search + entry, meal editing, recipe saving.
**Addresses:** Bundled nutrition DB (P1), food diary UI (P1), manual search (P1), editable meals (P1), portion estimation (P1), recipe saving (P1), meal confirmation flow (P1).
**Avoids:** Pitfall 11 (synchronous DB queries -- use async queries + FTS5 + pagination for 300K+ row nutrition DB).

### Phase 4: Gallery Scanning + Deduplication
**Rationale:** The primary UX differentiator. Depends on Phases 2-3 for the inference + nutrition pipeline. This is where the product becomes differentiated, not just functional.
**Delivers:** GalleryService (expo-media-library scanning), EXIF metadata extraction, photo deduplication (temporal clustering, optionally pHash), batch processing with thermal management, scan results confirmation flow.
**Addresses:** Gallery scanning - manual trigger (P1), basic deduplication (P1), EXIF extraction (P1).
**Avoids:** Pitfall 5 (background limits -- foreground-first design), Pitfall 10 (thermal throttling -- bursty processing), Pitfall 14 (image memory explosion -- resize before decode, process one at a time).

### Phase 5: Enhanced Detection + Scale OCR
**Rationale:** Accuracy improvements and novel features that deepen the competitive moat. These are additive -- the app is fully functional without them.
**Delivers:** llama.rn integration, VLM pipeline for complex dish identification, inference router update with VLM tier, scale OCR pipeline (custom 7-segment model), container weight learning, model manager (download/cache/version), device capability indicator UI, configurable UX modes, end-of-day notification, background gallery scanning, hidden ingredient inference.
**Addresses:** Optional VLM (P2), scale OCR (P2), container weight learning (P2), UX modes (P2), end-of-day notification (P2), background gallery scanning (P2), hidden ingredient inference (P2).
**Avoids:** Pitfall 3 (OOM -- tiered model loading with RAM gates), Pitfall 9 (PODAI beta -- CDN fallback for model delivery), Pitfall 12 (real-world OCR failures -- camera guide overlay, range validation, manual fallback).

### Phase 6: Sync + Distribution
**Rationale:** Cloud backup is a user trust feature, not a core feature. Ship it after the on-device experience is solid. Model delivery optimization is a distribution concern.
**Delivers:** react-native-cloud-storage integration, Google Drive sync adapter, iCloud sync adapter, sync UI (settings, status, conflicts), Play for On-Device AI integration (Android model delivery), iOS On-Demand Resources, Gemini Nano opportunistic adapter.
**Addresses:** Google Drive sync (P2), iCloud sync (P2).
**Avoids:** Pitfall 8 (sync corruption -- WAL checkpoint before upload, integrity check after download), Pitfall 16 (iCloud behavioral differences -- platform-specific testing).

### Phase Ordering Rationale

- **Phases 1-3 form the MVP.** A user can take a photo, get food detection + nutrition data, view it in a diary, edit mistakes, and search manually. This validates the on-device pipeline end-to-end.
- **Phase 4 adds the primary differentiator** (gallery scanning) once the pipeline is proven. Shipping gallery scanning on a broken inference pipeline would create a terrible first impression.
- **Phase 5 layers on accuracy improvements** (VLM, scale OCR) and UX refinements (modes, notifications) that are valuable but not table stakes.
- **Phase 6 is infrastructure** (sync, model delivery) that users request after they trust the core product. Shipping sync before the diary works would be premature optimization.
- **Parallelization opportunities:** Within Phase 1, prebuild setup runs parallel to schema work. Within Phase 3, the portion estimator TypeScript port runs parallel to nutrition DB bundling. Phase 5's VLM work (llama.rn) and scale OCR are independent of each other.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Model export pipeline (CoreML + TFLite outputs need cross-platform validation suite). YOLO model fine-tuning for food classes beyond COCO requires dataset curation decisions.
- **Phase 4:** Photo deduplication strategy (temporal clustering vs pHash vs embedding-based) needs experimentation with real gallery data to determine accuracy/performance tradeoffs.
- **Phase 5:** VLM model selection (SmolVLM-256M vs Moondream 0.5B) needs on-device benchmarking with real food photos. Scale OCR custom model training needs dataset collection beyond Roboflow's 948 images. PODAI integration for React Native is undocumented and may need a custom Expo config plugin.
- **Phase 6:** Google Drive OAuth2 setup with `drive.appdata` scope needs implementation research. iCloud entitlement configuration via Expo config plugin needs verification.

Phases with standard patterns (skip research-phase):
- **Phase 1:** op-sqlite + drizzle-orm setup is well-documented with official integration guides. SQLite schema migration is a solved pattern.
- **Phase 3:** Bundling SQLite assets, FTS5 search, and building diary UIs are all well-documented React Native patterns. The portion estimator is a TypeScript port of existing validated Python code.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Core libraries verified via npm/GitHub with pinned versions. Integration of all libraries together (config plugin compatibility) is unverified. expo-play-asset-delivery (2 years old) and PODAI integration are the weakest links. |
| Features | MEDIUM-HIGH | Competitor landscape well-documented (March 2026). Feature prioritization is clear. UX patterns for on-device inference validated by competitor analysis. Passive gallery scanning is genuinely novel -- no competitor reference exists. |
| Architecture | MEDIUM-HIGH | Component boundaries and data flows are well-defined. Schema migration from PostgreSQL to SQLite maps cleanly. Existing codebase inventory (keep/modify/remove/add) is thorough. Build order follows clear dependency chains. |
| Pitfalls | HIGH | 17 pitfalls with specific GitHub issues, version numbers, and platform documentation. Critical pitfalls (OOM, DB corruption, Expo Go) have high confidence from official sources. Recovery strategies documented. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Config plugin compatibility matrix:** All 5+ native libraries with config plugins (op-sqlite, react-native-fast-tflite, llama.rn, react-native-cloud-storage, expo-play-asset-delivery) have not been tested together in a single `npx expo prebuild`. This is the single highest-risk unknown and should be validated in Phase 1 Sprint 1.
- **PODAI React Native integration:** No community library or documented path exists for Play for On-Device AI in React Native. May require a custom Expo config plugin or native module. CDN fallback must be the primary plan.
- **iOS On-Demand Resources via Expo:** No Expo plugin exists. Requires a custom Expo native module (~100 lines Swift). Estimated effort is low but unverified.
- **llama.rn OpenCL GPU performance on non-Adreno chipsets:** Documented for Adreno 700+ only. MediaTek and Samsung Exynos performance is unknown.
- **Gemini Nano custom module effort:** ML Kit GenAI Prompt API is alpha. The ~200-line Kotlin wrapper estimate is based on API surface analysis, not implementation.
- **Real-world scale OCR accuracy:** Training data (948 images from Roboflow) may be insufficient for the diversity of real kitchen scales. Accuracy on user-submitted photos is unvalidated.
- **vision-camera-resize-plugin version pinning:** Version not independently verified; use latest and validate.

## Sources

### Primary (HIGH confidence)
- [@op-engineering/op-sqlite npm v15.2.5](https://www.npmjs.com/package/@op-engineering/op-sqlite)
- [Drizzle ORM op-sqlite integration](https://orm.drizzle.team/docs/connect-op-sqlite)
- [react-native-fast-tflite v2.0.0](https://github.com/mrousavy/react-native-fast-tflite)
- [llama.rn v0.11.2](https://www.npmjs.com/package/llama.rn) -- multimodal vision, Expo plugin
- [react-native-cloud-storage v2.3.0](https://github.com/Kuatsu/react-native-cloud-storage/releases)
- [Expo SDK 54 changelog](https://expo.dev/changelog/sdk-54)
- [Expo Background Task docs](https://docs.expo.dev/versions/latest/sdk/background-task/)
- [SQLite How To Corrupt](https://sqlite.org/howtocorrupt.html)
- [PowerSync React Native benchmarks](https://www.powersync.com/blog/react-native-database-performance-comparison)
- [MFP acquires Cal AI (TechCrunch, March 2026)](https://techcrunch.com/2026/03/02/myfitnesspal-has-acquired-cal-ai-the-viral-calorie-app-built-by-teens/)

### Secondary (MEDIUM confidence)
- [Play for On-Device AI docs (beta)](https://developer.android.com/google/play/on-device-ai)
- [ML Kit GenAI Prompt API (alpha)](https://developers.google.com/ml-kit/genai/prompt/android)
- [expo-play-asset-delivery GitHub](https://github.com/one-am-it/expo-play-asset-delivery)
- [arXiv 2503.21109](https://arxiv.org/abs/2503.21109) -- thermal throttling benchmarks
- [Callstack: Local LLMs on Mobile](https://www.callstack.com/blog/local-llms-on-mobile-are-a-gimmick) -- on-device LLM limitations

### Tertiary (LOW confidence -- needs validation)
- expo-play-asset-delivery compatibility with PODAI device targeting -- untested
- llama.rn OpenCL GPU on non-Adreno chipsets -- undocumented
- Custom Expo native module effort estimates (Gemini Nano, iOS ODR) -- API analysis only
- vision-camera-resize-plugin version compatibility -- not independently pinned

---
*Research completed: 2026-03-12*
*Supersedes: 2026-02-12 pre-pivot version*
*Ready for roadmap: yes*
