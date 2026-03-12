# Domain Pitfalls: Local-First ML Integration in React Native + Expo

**Domain:** Local-first mobile food tracker with on-device ML inference, bundled databases, optional cloud sync
**Researched:** 2026-03-12
**Focus:** Integration pitfalls when ADDING local-first capabilities to an existing React Native + Expo app
**Prior research:** See `docs/research/001-006` for underlying technology research; this document covers implementation pitfalls

---

## Critical Pitfalls

Mistakes that cause rewrites, data loss, or fundamental architecture changes. Address these in design phase or face weeks of rework.

---

### Pitfall 1: Expo Managed Workflow Cannot Run Native ML Modules Without Development Builds

**Severity:** CRITICAL
**Phase to address:** Phase 1 (project setup / infrastructure)

**What goes wrong:** The current app uses Expo managed workflow (Expo Go for development). Libraries like `react-native-fast-tflite`, `llama.rn`, `op-sqlite`, and any custom CoreML/LiteRT native modules cannot run in Expo Go. Developers build features against Expo Go, discover they need native modules, then face a disruptive workflow migration mid-project. Config plugins from different native ML libraries conflict with each other during prebuild, producing broken native projects that compile on one platform but not the other.

**Why it happens:** Expo Go bundles a fixed set of native modules. JSI-based libraries (op-sqlite, react-native-fast-tflite) and custom native modules require Continuous Native Generation (CNG) via `npx expo prebuild` and custom development builds via EAS Build. Teams delay this transition because "we'll add native stuff later" and build significant JS-only features first, only to discover that the dev build workflow is fundamentally different from Expo Go.

**Consequences:**
- Cannot test any ML inference during development without a custom dev build
- EAS Build introduces cloud build times (5-15 min per build) unless local builds are configured
- Config plugin conflicts between react-native-fast-tflite, op-sqlite, llama.rn, and expo-media-library can produce native compilation errors that are opaque from the JS side
- React Native 0.82 permanently disables the old bridge architecture -- all native modules must be JSI/TurboModule compatible

**Prevention:**
1. Switch to custom development builds on Day 1. Run `npx expo prebuild` and configure EAS Build before writing any feature code. The current app.json already has `"newArchEnabled": true` which is correct.
2. Set up local development builds (not just cloud EAS) to avoid 5-15 minute cloud build round-trips during active native module integration.
3. Create a config plugin compatibility matrix early: test that react-native-fast-tflite + op-sqlite + expo-media-library + llama.rn all prebuild together without conflicts on both iOS and Android.
4. Never commit the `android/` and `ios/` directories -- let CNG regenerate them. All native customizations go through config plugins.

**Detection (warning signs):**
- You are still using Expo Go for daily development
- You have not run `npx expo prebuild` and verified both platforms compile
- Native module config plugins are listed in app.json but untested together

**Confidence:** HIGH -- this is a well-documented Expo constraint confirmed by official Expo documentation and community reports.

---

### Pitfall 2: Bundled SQLite Database Fails to Copy or Update Across App Versions

**Severity:** CRITICAL
**Phase to address:** Phase 2 (data layer / nutrition DB integration)

**What goes wrong:** The USDA nutrition database (~50-80MB as SQLite) ships as a bundled asset. On Android, assets inside the APK are not physically on the filesystem -- they are read from the compressed bundle. The app copies the DB to the filesystem on first launch, but subsequent app updates do NOT automatically re-copy the updated database. Users get stale nutrition data forever. Worse, if you force-overwrite the bundled DB on update, you destroy any user customizations (custom foods, saved recipes) if they are in the same database file.

On iOS, Metro bundler does not include `.db` files by default -- you must add `"db"` to Metro's `resolver.assetExts`. Without this config, the database silently fails to bundle and the app opens an empty database. Developers see it work in development (where the Metro dev server serves assets differently) but it breaks in production builds.

**Why it happens:**
- Android's asset system reads directly from the APK/AAB archive; files must be extracted to the app's data directory before SQLite can open them
- `expo-sqlite`'s `assetSource` with `require()` has reported issues where it produces an empty database instead of importing the bundled data
- App updates preserve the app data directory -- if a copied DB already exists at the target path, the asset copy is skipped (by design, to preserve user data)
- Metro does not include `.db` extensions by default, and this is not documented in the "getting started" flow

**Consequences:**
- Nutrition data is stale after updates (silent data quality degradation)
- User data lost if bundled DB is force-overwritten
- Empty database on first launch (app appears broken with no nutrition data)
- App size bloats if old DB versions accumulate on-device

**Prevention:**
1. Separate user data and reference data into different SQLite databases. Bundled nutrition data in a read-only DB (`nutrition.db`). User entries, recipes, corrections in a read-write DB (`user.db`). This lets you overwrite the nutrition DB on app updates without touching user data.
2. Add `"db"` to Metro's `resolver.assetExts` in `metro.config.js` and verify in CI that the production build includes the bundled database.
3. Implement a DB version check: store a `schema_version` in the nutrition DB, compare against the bundled version on app launch, and re-copy only if the bundled version is newer.
4. Test the full lifecycle: fresh install -> use app -> app update with new nutrition DB -> verify user data preserved AND nutrition data updated.
5. For Play Asset Delivery (fast-follow): the nutrition DB arrives seconds after install, but the app must handle the window where it is not yet available. Show a loading state, not an empty food search.

**Detection (warning signs):**
- You have a single SQLite database for both user data and nutrition reference data
- Your metro.config.js does not include `"db"` in assetExts
- You have not tested an app update flow that includes a nutrition DB change
- `expo-sqlite openDatabaseAsync` with `assetSource` works in development but not in production APK

**Confidence:** HIGH -- documented in multiple expo/expo GitHub issues (#10881, #11335, #36429) and react-native-sqlite-storage issue #322.

---

### Pitfall 3: On-Device VLM Inference Crashes Low-RAM Devices via OOM Kill

**Severity:** CRITICAL
**Phase to address:** Phase 3 (ML inference integration)

**What goes wrong:** A VLM model is loaded into memory for inference, but on devices with 4GB RAM (30-35% of active Android devices), the model + React Native runtime + OS overhead exceeds available memory. Android's Low Memory Killer terminates the app without a crash report -- the app simply disappears. On iOS, exceeding ~2-3GB of app memory (from a total of 6GB on iPhone 14 Pro) triggers immediate jetsam termination, also with no standard crash log. The developer sees no error in Crashlytics because OOM kills bypass standard crash reporting.

**Why it happens:**
- React Native runtime + Hermes engine consumes ~150-300MB baseline
- Loading a VLM model allocates memory for model weights in a single allocation (not streaming)
- SmolVLM-256M needs ~300-500MB runtime RAM; Moondream 0.5B needs 816MB; Gemma 3n E2B needs 2GB
- On 4GB devices, Android reserves ~1.5-2GB for OS, leaving ~2-2.5GB for all apps combined
- Image encoding for VLMs is the memory spike: a single 640x640 image generates 300-1000 vision tokens, each consuming memory for attention computation
- The JS side has no visibility into native memory usage -- `Performance.memory` only tracks JS heap

**Consequences:**
- Silent app termination on 30-35% of Android devices when VLM inference is attempted
- No crash reports for OOM kills -- invisible in monitoring dashboards
- Users on budget devices experience repeated "random" crashes, leave 1-star reviews
- Developer believes VLM integration works because it works on their flagship test device

**Prevention:**
1. Implement runtime device capability detection BEFORE loading any VLM model. Check available RAM via native module, not JS. On Android use `ActivityManager.getMemoryInfo()`. On iOS use `os_proc_available_memory()`.
2. Define hard RAM thresholds for model tiers: no VLM below 4GB, SmolVLM-256M at 4-6GB, Moondream 0.5B at 6-8GB, Gemma 3n E2B at 8GB+. Never attempt to load a model that does not fit.
3. Load models lazily (on first inference request, not at app startup) and unload immediately after inference completes. Never keep VLM models resident in memory.
4. Use a separate process or isolate for ML inference where possible. On Android, llama.cpp runs in a native thread -- ensure it releases memory on completion.
5. Monitor memory pressure using Android's `ComponentCallbacks2.onTrimMemory()` and iOS's `UIApplication.didReceiveMemoryWarningNotification`. Abort inference and release models if memory pressure signals arrive.
6. Test on a 4GB device (or Android emulator configured with 4GB RAM) as part of CI. If it crashes, your model tier thresholds are wrong.

**Detection (warning signs):**
- You only test ML inference on your personal flagship phone
- Model loading happens at app startup instead of on-demand
- You have no native memory monitoring -- only JS heap tracking
- OOM crash rate is zero in your crash reporting (because OOM kills are not reported, not because they do not happen)

**Confidence:** HIGH -- confirmed by ADR-005 research: 30-35% of devices have <=4GB RAM. OOM kill behavior documented in Android/iOS developer docs and React Native OOM crash reports (facebook/react-native#34364).

---

### Pitfall 4: SQLite Schema Migrations Destroy User Data on App Updates

**Severity:** CRITICAL
**Phase to address:** Phase 2 (data layer setup)

**What goes wrong:** The app ships with a SQLite schema. A later update adds columns, renames tables, or changes data types. Without a migration system, the app either crashes on launch (schema mismatch) or silently opens the old schema and fails on new queries. The worst case: the developer ships a "fix" that drops and recreates tables, destroying all user food logs and saved recipes. Users who have been tracking for weeks lose everything.

SQLite does not support `ALTER TABLE DROP COLUMN` before SQLite 3.35.0 (React Native ships recent versions but older Android devices may have older system SQLite -- though op-sqlite bundles its own). `ALTER TABLE RENAME COLUMN` was added in SQLite 3.25.0. Teams that assume standard SQL ALTER TABLE support find their migrations fail on specific devices.

**Why it happens:**
- No built-in schema migration framework in op-sqlite or expo-sqlite. You must build or adopt one.
- Developers test fresh installs (empty DB) but not upgrades (populated DB with real user data)
- The app must handle users who skip versions -- a user on v1.0 updating directly to v1.4 must run migrations 1, 2, 3, and 4 sequentially
- Pre-populated databases shipped as assets can overwrite user databases if the copy logic does not check for existing data (see Pitfall 2)

**Consequences:**
- Complete loss of user food logs and saved recipes
- App crashes on launch for users who update from specific versions
- Users who skip versions hit untested migration paths
- 1-star reviews referencing "lost all my data" are irrecoverable -- those users never return

**Prevention:**
1. Implement a version-tracked migration system from Day 1, before any user data exists. Store `schema_version` as a PRAGMA `user_version` in the database. On app launch, read the version and apply all pending migrations sequentially.
2. Every migration runs inside a transaction. If any step fails, `ROLLBACK` the entire migration and report the error -- never leave the DB in a partially migrated state.
3. Never ship a migration that drops data. Use `ALTER TABLE ADD COLUMN` for new columns (with defaults). For destructive changes, create new tables and copy data.
4. Make migrations idempotent: check if the column/table already exists before creating it. Users may re-run the app multiple times during a failed update.
5. Write integration tests that: (a) create a DB at version N, (b) populate it with test data, (c) run migrations to version N+1, (d) verify all data is intact. Run these tests for every version pair (1->2, 1->3, 2->3, etc.).
6. Separate user data DB from bundled nutrition DB (see Pitfall 2). Only the user data DB needs migrations.

**Detection (warning signs):**
- You do not have a `schema_version` or `user_version` PRAGMA in your database
- Your migration tests only test fresh installs
- You have manually run `DROP TABLE` in a migration script
- You have not tested the scenario: install v1.0 -> add data -> update to v1.2 -> verify data intact

**Confidence:** HIGH -- documented extensively in React Native SQLite community (expo/expo#3059, andpor/react-native-sqlite-storage#157, #553).

---

## High-Severity Pitfalls

Mistakes that cause significant rework, poor user experience, or platform-specific failures.

---

### Pitfall 5: Background Gallery Scanning Killed by iOS 30-Second Limit and Android Battery Restrictions

**Severity:** HIGH
**Phase to address:** Phase 3 (gallery scanning)

**What goes wrong:** The app's core value proposition -- "scan your gallery for food photos automatically" -- requires background processing. On iOS, `expo-background-task` (SDK 53+, replacing expo-background-fetch) uses `BGTaskScheduler`, which gives your app up to 30 seconds of execution time. Running YOLO inference on even 5 photos takes longer than 30 seconds when you include model loading, image resizing, inference, and result processing. The OS kills your task mid-inference.

On Android 12+, background execution restrictions are aggressive. Foreground services require a persistent notification (annoying for passive scanning). WorkManager defers tasks based on battery, charging state, and network. On Android 15+, apps flagged as "battery drainers" are permanently restricted.

Additionally, Gemini Nano via AICore is foreground-only -- `BACKGROUND_USE_BLOCKED` error if your app is not the top app. This eliminates AICore as an option for gallery scanning entirely.

**Why it happens:**
- iOS fundamentally opposes long-running background computation for non-navigation apps
- Android battery optimization has tightened every release since Android 8
- Developers test background tasks in debug mode (which has relaxed constraints) and assume production behaves the same
- `BGProcessingTask` timing is entirely system-controlled in production -- it may execute once per day, or defer for days

**Consequences:**
- Gallery scanning processes 0-3 photos per background execution on iOS (not the 20-photo batch envisioned)
- Users see "0 new food photos found" after a day of eating and photographing
- The core differentiator (automatic passive scanning) silently does not work
- App flagged as battery drainer on Android, permanently restricted

**Prevention:**
1. Design gallery scanning as a **foreground feature triggered by user action**, not a background process. When the user opens the app, scan for new photos since last scan. Show a progress indicator. This is reliable and cross-platform.
2. Use background tasks only for lightweight indexing (enumerate new photo metadata, not inference). Store a list of "photos to process" in the background, then run inference when the user opens the app.
3. On iOS, call `BackgroundFetch.finish()` well before the 30-second deadline. Process a maximum of 2-3 photos per background execution, prioritizing the most recent.
4. On Android, use WorkManager constraints (`requiresCharging`, `requiresBatteryNotLow`) to schedule batch inference only when the device is charging. This naturally avoids battery drain complaints.
5. Test background tasks on a real device (not emulator, not debugger-attached) over a 24-hour period. Verify tasks actually execute with the expected frequency.
6. Implement incremental scanning: track the last scanned photo's creation date. On each scan (foreground or background), only process photos newer than that date.

**Detection (warning signs):**
- Your gallery scanning design assumes 20-photo batch processing in the background
- You have only tested background tasks with Xcode debugger attached
- You assume Gemini Nano will be available for background gallery inference
- Background task tests show consistent results (they should be inconsistent -- OS scheduling is non-deterministic)

**Confidence:** HIGH -- iOS 30-second limit confirmed in Expo docs and Apple BGTaskScheduler docs. Gemini Nano foreground-only limitation confirmed in ADR-005 research (doc 003).

---

### Pitfall 6: react-native-fast-tflite App Freezes During Synchronous Model Loading

**Severity:** HIGH
**Phase to address:** Phase 3 (ML inference integration)

**What goes wrong:** `react-native-fast-tflite` uses JSI for zero-copy memory access, which means model operations happen synchronously on the JS thread by default. Loading a TFLite model (especially a YOLO model at ~20MB or a VLM at 100MB+) blocks the JS thread for 500ms-3s depending on model size and device speed. The UI completely freezes during this time -- no animations, no touch response, no loading spinner. Users perceive the app as crashed.

A related issue: if model loading fails (corrupted model file, unsupported operations, memory pressure), the failure may crash the app rather than returning an error, because JSI errors can be unrecoverable.

**Why it happens:**
- JSI provides performance benefits (no serialization) but removes the async bridge's natural thread isolation
- Model loading involves disk I/O + memory allocation + delegate initialization, all synchronous
- Developers load the model as part of component mount (useEffect), which blocks the first render
- The GPU/NPU delegate initialization adds additional startup time beyond just reading the file

**Consequences:**
- App appears to hang/crash on first photo analysis attempt
- Users on slower devices (where model loading takes 2-3s) experience the worst version of this
- If model load is in a useEffect on a screen transition, the navigation animation stutters
- Unrecoverable JSI errors from corrupted models crash the app with no user-facing error message

**Prevention:**
1. Load models lazily and asynchronously using `InteractionManager.runAfterInteractions()` or a dedicated native thread. Do not load models during component mount or navigation transitions.
2. Show an explicit loading state while the model initializes. "Preparing AI..." with a spinner is better than a frozen screen.
3. Wrap model loading in try/catch at the native level. If a model fails to load (unsupported ops, OOM), fall back gracefully to a simpler model or manual entry mode.
4. Pre-warm models at app startup (after the UI has rendered) rather than on first inference request. Cache the loaded model reference in a singleton -- do not reload it per inference call.
5. Test model loading on the lowest-spec target device. If load time exceeds 1 second, implement the loading state UI.

**Detection (warning signs):**
- Model loading call is inside a `useEffect` without any loading state
- First inference attempt causes a visible UI freeze
- You have no error handling around `loadTensorflowModel()` calls
- App crashes on devices where the model's operations are not supported by the available delegates

**Confidence:** HIGH -- confirmed by react-native-fast-tflite GitHub issue #92 (app freezes when loading model) and #130 (app crashes on useTensorflowModel).

---

### Pitfall 7: LiteRT/TFLite and CoreML Produce Different Outputs for the Same Model

**Severity:** HIGH
**Phase to address:** Phase 3 (model export and validation)

**What goes wrong:** A YOLO model trained in PyTorch and exported to both CoreML (iOS) and TFLite/LiteRT (Android) produces different bounding boxes, confidence scores, and class predictions on the same input image. Android users and iOS users get different detection results for the same food photo. Differences stem from: floating point precision (FP16 on iOS Neural Engine vs INT8 on Android NPU), different NMS (non-maximum suppression) implementations, and different preprocessing (image normalization, color space).

Post-training quantization to INT8 for LiteRT can drop mAP by ~6.5 points. CoreML's FP16 loses less accuracy but still diverges from the FP32 Python reference.

**Why it happens:**
- CoreML and TFLite use entirely different inference engines with different numerical precision
- Quantization is not deterministic across frameworks -- the same model quantized to INT8 via TFLite vs CoreML produces different weight approximations
- NMS post-processing (removing overlapping boxes) has subtle implementation differences between platforms
- Image preprocessing pipelines often differ: RGB vs BGR ordering, normalization ranges [0,1] vs [-1,1], resize interpolation method
- Developers validate the model in Python (FP32) and assume both mobile exports match

**Consequences:**
- Inconsistent user experience across platforms -- same food photo gets different results
- Bug reports from one platform cannot be reproduced on the other
- A/B testing and accuracy metrics are platform-dependent
- "It works on iPhone" becomes a persistent development refrain

**Prevention:**
1. Build a cross-platform validation test suite: 100 reference images with known-good outputs from the Python model. Run both CoreML and TFLite exports against these images and compare bounding box coordinates, confidence scores, and class predictions. Accept a tolerance (e.g., IoU > 0.9 for boxes, confidence within 5%).
2. Standardize preprocessing in the native layer: ensure both platforms use identical resize, normalization, and color space conversion. Do not rely on framework defaults.
3. If using INT8 quantization on Android, use quantization-aware training (QAT) rather than post-training quantization (PTQ). QAT preserves ~2-3 more mAP points.
4. Implement NMS in your own code (JS or native) rather than relying on framework-specific NMS. This ensures identical post-processing on both platforms.
5. Pin export tool versions (coremltools, Ultralytics export, TFLite converter) and test exports in CI. A minor version bump can change export behavior.

**Detection (warning signs):**
- You have only tested the model on one platform
- Your test suite runs against the Python model but not the mobile exports
- Users report "detection works great on my iPhone but my friend's Android misses foods"
- You use different preprocessing code on iOS vs Android

**Confidence:** HIGH -- confirmed by prior research (Pitfall 5 in existing docs/research) and ADR-005 noting ~6.5 mAP drop with INT8.

---

### Pitfall 8: Google Drive / iCloud Sync Corrupts SQLite Database

**Severity:** HIGH
**Phase to address:** Phase 4 (sync layer)

**What goes wrong:** File-level sync of a SQLite database via Google Drive `appDataFolder` or iCloud seems simple: upload the .db file, download on the other device. But SQLite uses WAL (Write-Ahead Logging) mode by default in op-sqlite, which creates additional `-wal` and `-shm` files alongside the main database. Uploading only the main `.db` file without the WAL produces a database that is missing recent writes. Uploading all three files non-atomically means the WAL file can be out of sync with the main DB, producing corruption on the receiving device.

Additionally, if the user has the app open on two devices simultaneously (phone and tablet), both writing to a local DB and then syncing to Google Drive, the last-write-wins strategy overwrites the losing device's data entirely -- including food logs entered between the last sync and the overwrite.

**Why it happens:**
- SQLite's WAL mode is transparent to the application layer but critical for file-level sync
- Google Drive and iCloud sync files independently -- there is no atomic multi-file upload guarantee
- react-native-cloud-storage operates at the file level; it has no awareness of SQLite's multi-file structure
- Developers test sync with sequential usage (device A, then device B) but not concurrent usage

**Consequences:**
- Data loss: recent food logs missing after sync
- Database corruption: app crashes on launch after downloading a corrupted DB from cloud
- Silent data overwrite: one device's entries disappear because the other device's DB "won"
- User discovers data loss hours or days later when reviewing their food diary

**Prevention:**
1. Before uploading, checkpoint the WAL into the main database by running `PRAGMA wal_checkpoint(TRUNCATE)`. This merges all pending writes into the `.db` file and removes the WAL. Upload only the single `.db` file.
2. Before downloading and replacing the local DB, close all database connections. After replacing the file, reopen the database and verify integrity with `PRAGMA integrity_check`.
3. Implement a change log / sync version number alongside the database file. Before overwriting, compare versions and warn the user if the cloud version is older than the local version.
4. For the multi-device concurrent case: serialize sync operations with a lock file in the cloud storage, or implement a simple merge strategy (export food logs as JSON, merge by entry ID, import merged result) rather than whole-file replacement.
5. Always keep a local backup of the database before overwriting with the cloud version. If the downloaded DB is corrupt, restore from the local backup.
6. Long-term: if multi-device sync becomes a real need, migrate to PowerSync or row-level sync rather than file-level sync. File-level sync is inherently limited to single-device-at-a-time usage.

**Detection (warning signs):**
- You upload the `.db` file but not the WAL, or upload them non-atomically
- You do not run `PRAGMA wal_checkpoint(TRUNCATE)` before upload
- Sync tests only test one device at a time
- No integrity check after downloading and replacing the local DB
- No backup of the local DB before overwrite

**Confidence:** HIGH -- SQLite WAL corruption from partial file sync is well-documented in SQLite official docs and SQLite forum (data loss after iOS device force restarts).

---

### Pitfall 9: Play for On-Device AI Asset Packs Are in Beta with Undocumented Constraints

**Severity:** HIGH
**Phase to address:** Phase 3 (model delivery)

**What goes wrong:** The entire model delivery strategy for Android depends on Play for On-Device AI (PODAI) for device-targeted model delivery. PODAI is in beta (as of March 2026). Beta APIs can change, device targeting criteria can shift, and there is no SLA for delivery timing. Fast-follow packs "auto-download after install" but the timing is not guaranteed -- users may open the app before the model has downloaded. On-demand packs require explicit download management code.

The Expo/React Native integration with Play Asset Delivery is not well-documented. Most examples are native Android (Kotlin/Java) using the Play Core library. Bridging PODAI into React Native requires either a custom native module or an Expo config plugin that configures the Android Gradle Plugin (8.8+ required) to recognize AI pack modules.

There is no iOS equivalent of PODAI. On iOS, you must use Apple's On-Demand Resources (ODR) or Background Assets API, which have a completely different API surface, different size limits (512MB per tag for ODR), and different delivery semantics.

**Why it happens:**
- PODAI is purpose-built for native Android development, not cross-platform frameworks
- React Native's build system (Metro + Gradle for Android, Xcode for iOS) needs additional configuration to support asset packs
- Developers plan around PODAI features described in docs without testing the actual delivery timing and reliability
- No community libraries wrap PODAI for React Native as of March 2026

**Consequences:**
- Model delivery works in testing (sideloaded APKs with bundled models) but fails in production (Play Store asset pack delivery)
- App launches with no ML model available -- degraded experience on first launch
- Two completely different model delivery implementations needed for iOS and Android
- If PODAI beta is deprecated or fundamentally changed, the entire delivery strategy must be rebuilt

**Prevention:**
1. Build a fallback model delivery system that does not depend on PODAI: host models on a CDN (e.g., Cloudflare R2, free tier) and download them on first launch. This works on both platforms and gives you a working baseline before PODAI integration.
2. For PODAI integration, write an Expo config plugin that adds the AI pack module to the Android Gradle project and configures delivery types. Test this end-to-end by uploading to the Play Store's internal testing track.
3. Design the ML inference layer with a model-not-available state. The app must work (with degraded features) when models have not yet been downloaded. Show "Download AI model for enhanced detection" rather than crashing or showing empty results.
4. For iOS, use Apple's On-Demand Resources tagged by model tier. Test that ODR download + caching works when the user has low storage.
5. Track PODAI beta status closely. If it does not reach GA before your launch target, fall back to the CDN approach.

**Detection (warning signs):**
- Your model delivery only works with sideloaded / locally bundled models
- You have not tested asset pack delivery via the Play Store internal testing track
- Your app crashes or shows empty state when the ML model is not yet available
- You have no fallback for model delivery if PODAI is unavailable

**Confidence:** MEDIUM -- PODAI documentation is current (updated March 2026) but beta status introduces uncertainty. No React Native community libraries for PODAI integration confirmed.

---

## Moderate Pitfalls

Mistakes that cause significant debugging time, user-facing issues, or performance degradation.

---

### Pitfall 10: Thermal Throttling Ruins Batch Photo Processing UX

**Severity:** MODERATE
**Phase to address:** Phase 3 (gallery scanning / batch inference)

**What goes wrong:** Processing 10-20 food photos sequentially triggers thermal throttling after ~2.5 minutes of sustained inference. CPU frequency drops from 3GHz to 1GHz (up to 4.3x performance degradation). The first 5 photos process in 30 seconds; the next 5 take 2 minutes. Users see the progress bar slow to a crawl and assume the app is frozen. On some devices, GPU/NPU offloading causes device instability (screen freezes).

**Prevention:**
1. Process in bursts of 3-5 photos with 2-3 second cooldown pauses between bursts.
2. Query Android's Thermal API (`PowerManager.getThermalStatus()`) before each burst. If thermal status is `THERMAL_STATUS_MODERATE` or higher, increase cooldown time or pause processing.
3. Prefer GPU/NPU delegates over CPU for inference -- GPU runs at 1.3W/60C vs CPU at 10-12W/95C.
4. Show per-photo progress with estimated time remaining. "Processing photo 8 of 15 -- about 1 minute left" is better than a generic spinner.
5. Allow users to cancel batch processing and resume later.

**Detection:** First 5 photos process quickly, then processing speed drops dramatically. Device becomes warm to the touch during batch processing.

**Confidence:** HIGH -- confirmed by arXiv 2503.21109 and ADR-005 research doc 001.

---

### Pitfall 11: op-sqlite Synchronous API Blocks UI Thread on Large Queries

**Severity:** MODERATE
**Phase to address:** Phase 2 (data layer)

**What goes wrong:** op-sqlite's JSI-based API executes queries synchronously, which provides excellent performance for small queries but blocks the JS thread during large operations. Querying the full USDA nutrition database (~300K+ food entries) with a text search, or inserting 100+ food log entries during a sync merge, freezes the UI for 200ms-2s. Users see janky animations and unresponsive touch during database operations.

**Prevention:**
1. Use `executeAsync` methods where available. op-sqlite supports async operations -- prefer them for any query that might touch more than a few hundred rows.
2. For nutrition search, implement a debounced search with `LIMIT 20` results. Never query the full nutrition DB without pagination.
3. Batch large write operations (sync, bulk import) into chunks of 50-100 rows per transaction, yielding to the UI thread between batches.
4. Pre-build FTS5 (Full-Text Search) indexes on the nutrition database at build time, not at app startup. FTS5 queries are significantly faster than `LIKE '%query%'` on 300K+ rows.
5. Profile database operations on the lowest-spec target device. If any query takes >100ms, it needs optimization or async execution.

**Detection:** UI animations stutter when the user types in the food search box. Scrolling food log list causes brief freezes.

**Confidence:** MEDIUM -- confirmed by general op-sqlite documentation and community reports about JSI synchronous execution. Specific USDA DB size benchmarks not verified.

---

### Pitfall 12: Custom 7-Segment OCR Model Fails on Real-World Scale Photos

**Severity:** MODERATE
**Phase to address:** Phase 4 (scale OCR)

**What goes wrong:** The custom TFLite model trained on Roboflow 7-segment datasets (~948 images) achieves high accuracy on the training distribution but fails on real kitchen scale photos due to: reflections on LCD screens, uneven lighting, camera angle distortion, condensation on the display, partial occlusion from the container being weighed, and the wide variety of scale display fonts/sizes across manufacturers.

The decimal point is the hardest character to detect -- it is often a single pixel at low resolution. The `wetr` project found that characters need to be at least ~50 pixels tall for reliable decimal detection. If the user photographs from too far away, all digits may be correct but the decimal point is missed, turning "127.5g" into "1275g" -- a 10x error silently entering the food log.

**Prevention:**
1. Implement a camera guide overlay that frames the expected display region and warns the user to move closer if the display area is too small (below minimum pixel threshold).
2. Train with data augmentation: reflections, varying brightness, angle distortion, partial occlusion. Supplement Roboflow data with your own photos of 10+ different scale models.
3. Post-process with hard range validation: kitchen scales read 0-5000g. Any reading outside this range is rejected. Only one decimal point allowed. Readings that differ from the previous reading by more than 500g within 5 seconds trigger re-confirmation.
4. Always show the OCR result for user confirmation before accepting. Never auto-log a scale reading.
5. Provide manual weight entry as a first-class alternative, not a hidden fallback. Some users will always prefer typing.

**Detection:** OCR accuracy >95% on test images but <70% on user-submitted photos. Decimal point detection rate below 80%.

**Confidence:** MEDIUM -- based on ADR-005 research doc 004 and the `wetr` project findings. Real-world accuracy not yet validated for this specific pipeline.

---

### Pitfall 13: React Native New Architecture Compatibility Breaks Third-Party Native Modules

**Severity:** MODERATE
**Phase to address:** Phase 1 (infrastructure)

**What goes wrong:** The current app has `"newArchEnabled": true` in app.json (React Native 0.81.5). React Native 0.82 permanently disables the old bridge. Third-party libraries that have not migrated to TurboModules/Fabric will crash at runtime. The ML integration stack (react-native-fast-tflite, llama.rn, react-native-cloud-storage) and supporting libraries (react-native-gesture-handler, react-native-reanimated) must all support the new architecture.

**Prevention:**
1. Before adding any new native dependency, verify it supports the New Architecture. Check the library's README for "New Architecture" or "Fabric" support.
2. Major libraries in the current package.json (gesture-handler, reanimated, screens) already support New Architecture. Verify the specific versions installed are compatible.
3. For react-native-fast-tflite and llama.rn: check their GitHub issues for New Architecture compatibility. If not confirmed, plan for writing a thin TurboModule wrapper or finding an alternative.
4. Run `npx expo prebuild` and build for both platforms early to surface any incompatibilities before writing feature code.

**Detection:** Runtime crashes with "TurboModule not found" or "Fabric component not registered" errors after adding a new native library.

**Confidence:** MEDIUM -- major libraries have migrated, but specific ML libraries (llama.rn, react-native-fast-tflite) need individual verification.

---

### Pitfall 14: Image Memory Explosion During Batch Photo Processing

**Severity:** MODERATE
**Phase to address:** Phase 3 (gallery scanning)

**What goes wrong:** Modern phone cameras produce 12-48MP images. A single 12MP photo decompressed into RGBA bitmap consumes ~48MB of memory. Loading 5 photos for batch inference consumes 240MB before any ML model is loaded. Combined with the React Native runtime (~200MB) and a loaded YOLO model (~50-100MB), total memory reaches 500-600MB -- dangerously close to OOM on 4GB devices.

**Prevention:**
1. Never load full-resolution images for inference. Resize to model input size (640x640 for YOLO) before loading. Use platform-native resize (`CGImageSource` on iOS, `BitmapFactory.Options.inSampleSize` on Android) -- do not decode the full image first.
2. Process images one at a time, releasing the previous image before loading the next. Never hold multiple decoded images in memory simultaneously.
3. Use `expo-image-manipulator` or a native module for efficient downsizing before passing to the ML pipeline.
4. Set memory monitoring: if available memory drops below 200MB, pause batch processing and release resources.

**Detection:** App crashes after processing 3-5 photos in sequence. Memory profiler shows sawtooth pattern with peaks near OOM threshold.

**Confidence:** HIGH -- image memory consumption is deterministic (width * height * 4 bytes). Confirmed by React Native OOM discussion (facebook/react-native#33640).

---

## Minor Pitfalls

Annoyances, debugging time sinks, and edge cases that are good to know about.

---

### Pitfall 15: NNAPI Delegate Silently Falls Back to CPU on Android 15+

**Severity:** MINOR
**Phase to address:** Phase 3 (ML inference)

**What goes wrong:** Code that uses the NNAPI delegate for TFLite continues to compile and run on Android 15+ (NNAPI is deprecated but not removed). However, NNAPI may silently fall back to CPU execution for operations not supported by the device's NPU, resulting in 10-50x slower inference than expected. Developers see their model "working" on Android 15 devices but at CPU speed.

**Prevention:** Use LiteRT with vendor-specific NPU delegates (Qualcomm QNN, MediaTek NeuroPilot) instead of NNAPI. react-native-fast-tflite supports GPU delegate configuration -- use it.

**Confidence:** HIGH -- NNAPI deprecation in Android 15 confirmed by official Android NDK docs.

---

### Pitfall 16: iCloud Sync Differs Fundamentally from Google Drive Sync

**Severity:** MINOR
**Phase to address:** Phase 4 (sync layer)

**What goes wrong:** Developers build Google Drive sync first (cross-platform), then assume iCloud is a similar file-level API. iCloud has fundamentally different behavior: iCloud Drive can evict files to save device storage (the file becomes a "ghost" that must be re-downloaded), iCloud has no equivalent of Google Drive's `appDataFolder` (data is visible to the user in Files app), and iCloud conflicts are resolved by the system creating duplicate files rather than overwriting.

**Prevention:**
1. Use react-native-cloud-storage's abstraction layer to hide platform differences, but test platform-specific behaviors separately.
2. For iCloud, use iCloud Key-Value Storage for small data (sync state, preferences) and iCloud Documents for the database file. Handle the "file evicted" case by checking for the file's availability before opening.
3. Accept that iCloud sync will be iOS-only and may have different conflict behavior than Google Drive sync. Document these differences for users.

**Confidence:** MEDIUM -- based on Apple documentation and react-native-cloud-storage library docs.

---

### Pitfall 17: EAS Build Environment Differs from Local Development Environment

**Severity:** MINOR
**Phase to address:** Phase 1 (infrastructure)

**What goes wrong:** A build that works locally fails on EAS Build because: different Node.js version, different CocoaPods version, different Gradle version, or environment variables not configured in EAS secrets. The bundled SQLite database is included in local builds but missing from EAS builds because the asset configuration differs.

**Prevention:**
1. Match local development environment to EAS Build environment. Use `.nvmrc` to pin Node version. Specify `"ios.image"` and `"android.image"` in eas.json.
2. Add a CI step that runs `npx expo prebuild` and compiles for both platforms on every PR.
3. Test asset bundling (SQLite DBs, ML models) in EAS builds early -- not just before launch.
4. Store secrets (Google Drive API keys, etc.) in EAS secrets, not in `.env` files that are not available in cloud builds.

**Confidence:** MEDIUM -- common Expo/EAS issue based on community reports.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Severity | Mitigation |
|-------------|---------------|----------|------------|
| Project Setup | Expo Go incompatibility with native ML modules (Pitfall 1) | CRITICAL | Switch to custom dev builds on Day 1 |
| Project Setup | New Architecture incompatibility (Pitfall 13) | MODERATE | Verify all native deps support New Arch before adding |
| Data Layer | Bundled DB copy failures (Pitfall 2) | CRITICAL | Separate user DB from nutrition DB, configure Metro |
| Data Layer | Schema migration data loss (Pitfall 4) | CRITICAL | Implement versioned migrations from Day 1 |
| Data Layer | Synchronous DB queries blocking UI (Pitfall 11) | MODERATE | Use async queries, FTS5 indexes, pagination |
| ML Integration | OOM crashes on low-RAM devices (Pitfall 3) | CRITICAL | Runtime capability detection, tiered model loading |
| ML Integration | Model loading freezes (Pitfall 6) | HIGH | Async loading, loading states, error handling |
| ML Integration | Cross-platform output divergence (Pitfall 7) | HIGH | Cross-platform validation test suite |
| ML Integration | NNAPI silent CPU fallback (Pitfall 15) | MINOR | Use LiteRT vendor delegates, not NNAPI |
| Gallery Scanning | Background execution limits (Pitfall 5) | HIGH | Foreground-first scanning design |
| Gallery Scanning | Image memory explosion (Pitfall 14) | MODERATE | Resize before decode, process one at a time |
| Gallery Scanning | Thermal throttling (Pitfall 10) | MODERATE | Bursty inference with cooldown |
| Model Delivery | PODAI beta instability (Pitfall 9) | HIGH | CDN fallback, handle model-not-available state |
| Scale OCR | Real-world photo failures (Pitfall 12) | MODERATE | Camera guide overlay, range validation, manual fallback |
| Sync Layer | SQLite WAL corruption (Pitfall 8) | HIGH | WAL checkpoint before upload, integrity check after download |
| Sync Layer | iCloud behavioral differences (Pitfall 16) | MINOR | Platform-specific testing, documented differences |
| Infrastructure | EAS build environment drift (Pitfall 17) | MINOR | Pin versions, CI prebuild checks |

---

## Pitfall Interaction Map

Some pitfalls compound each other when they co-occur:

```
Pitfall 3 (OOM) + Pitfall 14 (Image Memory) = Double memory pressure
  During batch scanning, image decode + model loading hits memory ceiling.
  Combined prevention: resize images first, then load model, never overlap.

Pitfall 2 (DB Copy) + Pitfall 4 (Schema Migration) = Data Loss
  If bundled DB overwrites user DB, AND migrations are not versioned,
  users lose data AND the app cannot detect what schema version they have.
  Combined prevention: separate databases, version both independently.

Pitfall 5 (Background Limits) + Pitfall 10 (Thermal Throttling) = Zero Background Utility
  30-second iOS limit combined with thermal throttling means background
  inference is near-useless. Combined prevention: foreground scanning only.

Pitfall 6 (Model Loading Freeze) + Pitfall 1 (No Dev Build) = Cannot Debug
  If you discover model loading freezes but are still on Expo Go,
  you cannot test native-level fixes. Combined prevention: dev builds first.
```

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|--------------|----------------|
| Expo Go dependency (Pitfall 1) | LOW if caught early, HIGH if features built on Expo Go | Migrate to dev builds, reconfigure all native modules, rebuild |
| Bundled DB failures (Pitfall 2) | MEDIUM | Separate databases, add Metro config, rebuild asset pipeline |
| OOM crashes (Pitfall 3) | MEDIUM | Add capability detection, unload models, add tiering |
| Schema migration data loss (Pitfall 4) | VERY HIGH (data is gone) | Ship emergency update with data recovery attempt from WAL/backup. Apologize to users. Cannot recover data if DB was overwritten. |
| Background scanning limits (Pitfall 5) | HIGH | Redesign scanning as foreground feature, change core UX flow |
| Model loading freeze (Pitfall 6) | LOW | Add async loading wrapper, loading state UI |
| Cross-platform divergence (Pitfall 7) | MEDIUM | Build validation suite, standardize preprocessing, may need retraining |
| Sync corruption (Pitfall 8) | HIGH | Ship WAL checkpoint fix, provide manual backup/restore, may need to rebuild user trust |
| PODAI instability (Pitfall 9) | MEDIUM | Switch to CDN-based delivery, rewrite download management |

---

## Sources

### Expo / React Native
- [Expo Add Custom Native Code docs](https://docs.expo.dev/workflow/customizing/) -- config plugin and CNG workflow
- [Expo Background Task docs](https://docs.expo.dev/versions/latest/sdk/background-task/) -- iOS 30-second limit, WorkManager integration
- [Expo SQLite docs](https://docs.expo.dev/versions/latest/sdk/sqlite/) -- openDatabaseAsync, assetSource
- [expo/expo#10881](https://github.com/expo/expo/issues/10881) -- pre-populated database issues
- [expo/expo#37169](https://github.com/expo/expo/issues/37169) -- SQLite production vs development divergence
- [react-native-fast-tflite#92](https://github.com/mrousavy/react-native-fast-tflite/issues/92) -- app freezes during model loading
- [react-native-fast-tflite#130](https://github.com/mrousavy/react-native-fast-tflite/issues/130) -- app crash on useTensorflowModel
- [facebook/react-native#34364](https://github.com/facebook/react-native/issues/34364) -- Android memory leak and OOM
- [facebook/react-native#33640](https://github.com/facebook/react-native/issues/33640) -- OutOfMemoryError with New Architecture
- [Expo Prebuild (CNG) docs](https://docs.expo.dev/workflow/prebuild/) -- continuous native generation

### On-Device ML
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite) -- JSI-based TFLite
- [llama.rn GitHub](https://github.com/mybigday/llama.rn) -- React Native binding for llama.cpp
- [Play for On-Device AI (beta)](https://developer.android.com/google/play/on-device-ai) -- AI pack delivery, device targeting
- [arXiv 2503.21109](https://arxiv.org/abs/2503.21109) -- thermal throttling at 2.5 min, up to 4.3x degradation
- [arXiv 2511.13453](https://arxiv.org/abs/2511.13453) -- YOLOv8n benchmarks, INT8 accuracy loss

### SQLite / Data
- [SQLite How To Corrupt](https://sqlite.org/howtocorrupt.html) -- official corruption causes and prevention
- [SQLite Forum: Data loss after iOS force restart](https://sqlite.org/forum/info/dbe512e5bc140c6dac2127f7142a62f591903e9ffe227e87a36bab30a2ae0205)
- [andpor/react-native-sqlite-storage#322](https://github.com/andpor/react-native-sqlite-storage/issues/322) -- pre-populated DB update issues
- [React Native SQLite migrations guide (Medium)](https://medium.com/@hamzash863/navigating-sqlite-database-migrations-in-react-native-786d418655e6)
- [React Native SQLite upgrade strategy](https://embusinessproducts.com/react-native-sqlite-database-upgrade-strategy)

### Project Research
- `docs/research/001-android-fragmentation-on-device-ml.md` -- RAM distribution, chipset landscape, thermal throttling
- `docs/research/002-on-device-vlm-feasibility.md` -- VLM memory requirements, inference benchmarks
- `docs/research/003-gemini-nano-aicore.md` -- foreground-only limitation, battery quota
- `docs/research/004-scale-ocr-on-device.md` -- 7-segment display challenges, decimal detection
- `docs/research/005-app-size-install-impact.md` -- APK size thresholds, PODAI delivery
- `docs/research/006-offline-first-sync-patterns.md` -- sync library assessment, WAL considerations
- `docs/adr/005-local-first-no-subscription-architecture.md` -- architecture decisions and risk assessment

---
*Pitfalls research for: Local-first ML integration in React Native + Expo*
*Researched: 2026-03-12*
*Prior research (domain pitfalls): 2026-02-12 -- see existing PITFALLS.md in docs/research or previous .planning version*
