# Architecture Research

**Domain:** On-device ML food tracking with passive gallery scanning
**Researched:** 2026-02-12
**Confidence:** MEDIUM

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          MOBILE DEVICE (iOS / Android)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
│  │  React Native  │  │  Gallery       │  │  Notification  │                  │
│  │  UI Layer      │  │  Scanner       │  │  Manager       │                  │
│  │  (Expo)        │  │  (Native)      │  │  (Native)      │                  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                  │
│          │                   │                   │                            │
│  ┌───────┴───────────────────┴───────────────────┴────────┐                  │
│  │               ML Inference Bridge (Native Module)       │                  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │                  │
│  │  │ YOLO Model   │  │ Dedup Engine  │  │ EXIF         │  │                  │
│  │  │ CoreML/TFLite│  │ pHash        │  │ Extractor    │  │                  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │                  │
│  └─────────────────────────────┬───────────────────────────┘                  │
│                                │                                              │
│  ┌─────────────────────────────┴───────────────────────────┐                  │
│  │               Local State (SQLite / Zustand / MMKV)      │                  │
│  │  scan_queue | processed_photos | pending_entries | hashes │                  │
│  └──────────────────────────────────────────────────────────┘                  │
│                                │                                              │
└────────────────────────────────┼──────────────────────────────────────────────┘
                                 │ HTTPS (structured data only — no raw images
                                 │ unless cloud fallback triggered)
┌────────────────────────────────┼──────────────────────────────────────────────┐
│                          GO BACKEND                                          │
├────────────────────────────────┼──────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────┴───────┐  ┌────────────────┐                  │
│  │  Food Entry    │  │  Cloud ML      │  │  Nutrition DB  │                  │
│  │  API           │  │  Fallback      │  │  Router        │                  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                  │
│          │                   │                   │                            │
│  ┌───────┴───────────────────┴───────────────────┴────────┐                  │
│  │                    PostgreSQL                           │                  │
│  │  users | food_entries | ingredients | photos | recipes  │                  │
│  │  scan_sessions | container_weights | health_sync       │                  │
│  └────────────────────────────────────────────────────────┘                  │
│                                │                                              │
│  ┌─────────────────────────────┴───────────────────────────┐                  │
│  │            External Services                             │                  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │                  │
│  │  │ USDA API │  │ AFCD     │  │ Open     │              │                  │
│  │  │          │  │ /CoFID   │  │ Food     │              │                  │
│  │  │          │  │ /CIQUAL  │  │ Facts    │              │                  │
│  │  └──────────┘  └──────────┘  └──────────┘              │                  │
│  └────────────────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Boundaries

### On-Device Components

| Component | Responsibility | Runs Where | Communicates With |
|-----------|---------------|------------|-------------------|
| **Gallery Scanner** | Monitor photo library for new images; detect food candidates; trigger processing pipeline | Native module (Swift/Kotlin) via Expo Module | EXIF Extractor, Dedup Engine, ML Inference Bridge |
| **ML Inference Bridge** | Load YOLO model, run inference on images, return bounding boxes + labels + confidence scores | Native module wrapping CoreML (iOS) / TFLite (Android) | Gallery Scanner, UI Layer (results display) |
| **EXIF Extractor** | Extract timestamp, GPS, camera metadata from photos | Native module or JS library (react-native-exify) | Gallery Scanner, Go Backend (sends metadata with entries) |
| **Dedup Engine** | Compute perceptual hashes (pHash/dHash), detect near-duplicate photos, group multi-angle shots | On-device (native or JS) | Gallery Scanner, Local State |
| **Scale OCR Module** | Detect digital scale displays, extract weight readings via OCR | On-device (Vision framework iOS / ML Kit Android) | ML Inference Bridge, Go Backend |
| **Local State Store** | Cache scan progress, processed photo hashes, pending entries (offline-first) | SQLite or MMKV on device | All on-device components |
| **Notification Manager** | Schedule end-of-day summary notifications with macro totals | Native module (expo-notifications) | Local State, Go Backend (daily totals) |
| **UI Layer** | Display food diary, entry editing, scan results, nutrition summaries | React Native + Expo | All on-device components, Go Backend |

### Cloud/Backend Components

| Component | Responsibility | Runs Where | Communicates With |
|-----------|---------------|------------|-------------------|
| **Go API Server** | REST API for food entries, recipes, user management, sync | Go service (replaces Express) | PostgreSQL, Nutrition DB Router, Cloud ML Fallback |
| **Cloud ML Fallback** | Re-analyze low-confidence detections using LLM vision (Gemini) | Go service calling Gemini API | Go API Server, mobile client (when triggered) |
| **Nutrition DB Router** | Route ingredient lookups to region-appropriate databases (USDA, AFCD, CoFID, etc.) | Go service | External nutrition APIs, PostgreSQL (cached results) |
| **PostgreSQL** | Source of truth for all user data, entries, ingredients, recipes, sync state | Managed database | Go API Server |
| **Health Platform Sync** | Bidirectional sync with Apple Health / Google Fit | Go backend + native HealthKit/Health Connect modules | Go API Server, native health modules |

## Component Responsibilities (Detailed)

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Gallery Scanner | Detect new photos in device library, filter for food candidates, manage scan state | iOS: PHPhotoLibrary change history + BGProcessingTask; Android: MediaStore observer + WorkManager |
| YOLO Inference | Run object detection model, return bounding boxes with food labels and confidence | iOS: CoreML model loaded via Vision framework; Android: TFLite via ML Kit or direct interpreter |
| Dedup Engine | Generate perceptual hashes, compare against existing hashes, cluster similar photos | pHash/dHash algorithm; Hamming distance < 10 = duplicate; group by EXIF timestamp proximity |
| EXIF Extraction | Read GPS coordinates, timestamps, camera model, focal length from image metadata | react-native-exify or native PHAsset metadata (iOS) / ExifInterface (Android) |
| Scale OCR | Detect scale displays in images, read weight values | iOS: VNRecognizeTextRequest; Android: ML Kit Text Recognition |
| Nutrition Lookup | Convert ingredient names + weights to macro/micronutrient breakdowns | USDA FoodData Central API (primary), region-specific DBs, cached locally |
| Cloud Fallback | Re-analyze images that got low on-device confidence scores | Gemini Vision API via Go backend; only sends food region crops, not full images |
| Notification | Send configurable end-of-day summary with total calories, protein, carbs, fat | expo-notifications with scheduled triggers; data from local state or backend sync |

## Recommended Project Structure

```
foodtracker/
├── apps/
│   └── mobile/                         # React Native + Expo app
│       ├── src/
│       │   ├── screens/                # Screen components
│       │   ├── components/             # Reusable UI components
│       │   ├── navigation/             # React Navigation setup
│       │   ├── store/                  # Zustand stores
│       │   ├── lib/
│       │   │   ├── api/                # Go backend API client
│       │   │   ├── ml/                 # ML inference JS interface
│       │   │   ├── scanner/            # Gallery scan orchestration
│       │   │   ├── dedup/              # Deduplication logic (JS)
│       │   │   └── nutrition/          # Nutrition calculation utils
│       │   └── types/                  # TypeScript types
│       └── modules/                    # Expo native modules
│           ├── yolo-inference/         # CoreML/TFLite bridge
│           │   ├── ios/                # Swift implementation
│           │   ├── android/            # Kotlin implementation
│           │   └── src/                # JS interface
│           ├── gallery-scanner/        # Photo library monitoring
│           │   ├── ios/                # PHPhotoLibrary + BGProcessingTask
│           │   ├── android/            # MediaStore + WorkManager
│           │   └── src/                # JS interface
│           └── scale-ocr/              # Scale weight OCR
│               ├── ios/                # VNRecognizeTextRequest
│               ├── android/            # ML Kit Text Recognition
│               └── src/                # JS interface
├── backend-go/                         # Go backend (replaces backend/)
│   ├── cmd/
│   │   └── server/                     # Main entry point
│   ├── internal/
│   │   ├── api/                        # HTTP handlers
│   │   ├── service/                    # Business logic
│   │   ├── repository/                 # Database access
│   │   ├── nutrition/                  # Nutrition DB router
│   │   ├── ml/                         # Cloud ML fallback client
│   │   └── health/                     # HealthKit/Health Connect sync
│   ├── pkg/
│   │   └── models/                     # Shared types
│   └── migrations/                     # SQL migrations
├── models/                             # Trained ML model artifacts
│   ├── yolo-food.mlpackage             # iOS CoreML model
│   ├── yolo-food.tflite                # Android TFLite model
│   └── training/                       # Training scripts (Python)
│       ├── train_yolo.py               # Fine-tune on food datasets
│       ├── export_models.py            # Export to CoreML/TFLite
│       └── requirements.txt            # Training dependencies
├── backend/                            # Legacy Express backend (deprecated)
├── services/                           # Legacy AI agent service (deprecated)
├── spike/                              # POC experiments
│   └── food-detection-poc/             # YOLO detection spike
└── docs/
    └── adr/                            # Architecture Decision Records
```

### Structure Rationale

- **`apps/mobile/modules/`:** Custom Expo native modules for ML inference, gallery scanning, and OCR. These are platform-specific native code that cannot be implemented in JS. Expo Modules API provides the bridge.
- **`backend-go/`:** New Go backend using standard Go project layout (cmd/internal/pkg). Separate from legacy `backend/` to allow parallel development during migration.
- **`models/`:** Centralized directory for trained ML model artifacts and training scripts. Models are checked into Git LFS or downloaded during build.

## Data Flow

### Primary Flow: Gallery Scan to Food Entry

```
[Phone Gallery]
    │ (new photo added by user)
    ↓
[Gallery Scanner] ─── BGProcessingTask (iOS) / WorkManager (Android)
    │
    ├── 1. Query new photos since last scan token
    │      (PHPhotoLibrary.fetchPersistentChangesSinceToken / MediaStore query)
    │
    ├── 2. Extract EXIF metadata
    │      timestamp, GPS, camera model, focal length
    │
    ├── 3. Compute perceptual hash (pHash/dHash)
    │      Compare against stored hashes → Hamming distance
    │      IF distance < threshold → mark as duplicate, skip or group
    │
    ├── 4. Run YOLO inference on-device
    │      Load CoreML/TFLite model → bounding boxes + labels + confidence
    │      IF no food detected (all confidence < 0.3) → skip photo
    │      IF confidence < configurable threshold (e.g. 0.6) → flag for cloud fallback
    │
    ├── 5. [Optional] Scale OCR
    │      IF scale-like region detected → extract weight reading
    │
    ├── 6. Store detection results locally
    │      SQLite: photo_id, detections[], exif_metadata, phash, scan_session_id
    │
    └── 7. Create pending food entry
           detections + metadata → pending_entries table
           Status: "pending_review" or "auto_confirmed" (based on UX mode)
           ↓
    [Sync to Backend]
    │
    ├── 8. POST structured detection data to Go API
    │      (ingredient names, weights, confidence, EXIF metadata)
    │      Raw images stay on device unless cloud fallback needed
    │
    ├── 9. Go API: Nutrition DB lookup
    │      Route to USDA/AFCD/CoFID based on user region
    │      Return per-ingredient macro/micronutrient breakdown
    │
    ├── 10. [If flagged] Cloud ML fallback
    │       Send cropped food regions (not full image) to Gemini Vision
    │       Replace low-confidence on-device results with cloud results
    │
    └── 11. Persist food entry
           PostgreSQL: food_entries + ingredients + photos (metadata only)
           Return entry to mobile → update local state → show in diary
```

### Cloud Fallback Flow

```
[On-Device Detection]
    │
    ├── Confidence >= threshold (e.g. 0.6)
    │   └── Accept on-device result → proceed to nutrition lookup
    │
    └── Confidence < threshold
        │
        ├── 1. Crop food region from image using bounding box
        │      (Only food region sent, not background — privacy)
        │
        ├── 2. Upload crop to Go backend
        │
        ├── 3. Go backend calls Gemini Vision API
        │      Prompt: "Identify food items, estimate portions in grams"
        │
        ├── 4. Parse structured response → ingredient list
        │
        └── 5. Merge with on-device results
               Keep high-confidence on-device detections
               Replace low-confidence with cloud results
               Mark source as "cloud_fallback" in metadata
```

### Deduplication Flow

```
[New Photo]
    │
    ├── 1. Compute pHash (perceptual hash)
    │      Resize → grayscale → DCT → threshold → 64-bit hash
    │
    ├── 2. Compare against recent hashes (last 24 hours)
    │      Hamming distance = XOR + popcount
    │
    ├── IF distance <= 5 (near-identical)
    │   └── Skip entirely — same photo, different crop or compression
    │
    ├── IF distance 6-15 (similar — multiple angles)
    │   └── Group together as "same meal"
    │       Use photo with highest YOLO confidence as primary
    │       Merge detections across grouped photos
    │
    └── IF distance > 15 (different photo)
        └── Treat as new food event → full pipeline
```

### Notification Flow

```
[Configurable Trigger — e.g. 8:00 PM daily]
    │
    ├── 1. Query today's food entries from local state
    │
    ├── 2. Compute daily totals
    │      Total calories, protein, carbs, fat
    │
    ├── 3. Compare against user goals (if set)
    │
    └── 4. Send local push notification
           "Today: 2,150 kcal | 145g protein | 220g carbs | 78g fat"
           Tap → opens diary screen for today
```

## Architectural Patterns

### Pattern 1: Native Module Bridge for ML Inference

**What:** Expo native module that wraps platform-specific ML frameworks (CoreML on iOS, TFLite on Android) behind a unified TypeScript API. The JS layer calls `runInference(imageUri)` and receives structured detection results.

**When to use:** Any on-device ML inference. The YOLO model must run natively for performance (GPU/NPU access).

**Trade-offs:**
- Pro: Full hardware acceleration (ANE on iOS, NNAPI on Android), ~30ms inference
- Pro: Unified JS API despite different native implementations
- Con: Requires maintaining Swift + Kotlin code
- Con: Expo prebuild (custom dev client) required, not compatible with Expo Go

**Example:**
```typescript
// apps/mobile/modules/yolo-inference/src/index.ts
import { NativeModule, requireNativeModule } from 'expo-modules-core';

interface Detection {
  label: string;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
}

interface YoloInferenceModule extends NativeModule {
  runDetection(imageUri: string): Promise<Detection[]>;
  loadModel(modelName: string): Promise<boolean>;
  isModelLoaded(): boolean;
}

export default requireNativeModule<YoloInferenceModule>('YoloInference');
```

```swift
// apps/mobile/modules/yolo-inference/ios/YoloInferenceModule.swift
import ExpoModulesCore
import Vision
import CoreML

public class YoloInferenceModule: Module {
  private var model: VNCoreMLModel?

  public func definition() -> ModuleDefinition {
    Name("YoloInference")

    AsyncFunction("runDetection") { (imageUri: String) -> [[String: Any]] in
      guard let model = self.model else { throw ModelNotLoadedError() }
      let image = try loadImage(from: imageUri)
      let request = VNCoreMLRequest(model: model)
      let handler = VNImageRequestHandler(cgImage: image)
      try handler.perform([request])
      return self.parseResults(request.results)
    }

    AsyncFunction("loadModel") { (modelName: String) -> Bool in
      let config = MLModelConfiguration()
      config.computeUnits = .all  // Use ANE when available
      let mlModel = try MLModel(contentsOf: Bundle.main.url(
        forResource: modelName, withExtension: "mlmodelc"
      )!, configuration: config)
      self.model = try VNCoreMLModel(for: mlModel)
      return true
    }
  }
}
```

### Pattern 2: Background Gallery Scanning with Change Tokens

**What:** Use platform background task APIs to periodically scan the photo library for new images. Persist a "change token" so each scan only processes photos added since the last scan.

**When to use:** Passive food photo discovery. The scan runs when the device is idle, not during active use.

**Trade-offs:**
- Pro: Zero-effort food logging (user just takes photos normally)
- Pro: Battery-efficient (OS schedules during idle/charging)
- Con: iOS limits BGProcessingTask execution to a few minutes, unpredictable scheduling
- Con: Requires explicit photo library permission (full access, not limited picker)
- Con: Platform differences require separate native implementations

**Implementation detail (iOS):**
```swift
// Gallery Scanner native module — iOS
import Photos

// Persist change token between scans
func scanNewPhotos() async throws -> [PHAsset] {
  let token = loadPersistedChangeToken()

  // Fetch changes since last scan
  let changeResult = try PHPhotoLibrary.shared()
    .fetchPersistentChanges(since: token)

  var newAssets: [PHAsset] = []
  for change in changeResult {
    let details = change.changeDetails(for: PHObjectType.asset)
    let insertedIDs = details?.insertedObjectIdentifiers ?? []

    let fetchResult = PHAsset.fetchAssets(
      withLocalIdentifiers: insertedIDs.map { $0.identifier },
      options: nil
    )
    fetchResult.enumerateObjects { asset, _, _ in
      if asset.mediaType == .image {
        newAssets.append(asset)
      }
    }
  }

  // Persist new token for next scan
  persistChangeToken(changeResult.currentEndToken)
  return newAssets
}
```

**Implementation detail (Android):**
```kotlin
// Gallery Scanner — Android WorkManager
class GalleryScanWorker(context: Context, params: WorkerParameters)
    : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val lastScanTimestamp = getLastScanTimestamp()
        val cursor = contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            arrayOf(MediaStore.Images.Media._ID, MediaStore.Images.Media.DATE_ADDED),
            "${MediaStore.Images.Media.DATE_ADDED} > ?",
            arrayOf(lastScanTimestamp.toString()),
            "${MediaStore.Images.Media.DATE_ADDED} DESC"
        )
        // Process new photos...
        return Result.success()
    }
}
```

### Pattern 3: Confidence-Based Cloud Fallback

**What:** On-device inference runs first for every photo. If the maximum confidence score is below a configurable threshold, crop the food region and send it to a cloud vision model for re-analysis.

**When to use:** When on-device accuracy is insufficient for a specific image (unusual food, poor lighting, complex dish).

**Trade-offs:**
- Pro: Best of both worlds: fast/free for clear photos, accurate for difficult ones
- Pro: Privacy-preserving (only food crops sent, not full images)
- Pro: Cloud usage is proportional to difficulty, not photo volume
- Con: Adds latency for low-confidence detections (2-5s network round trip)
- Con: Requires internet connectivity for fallback
- Con: Cloud API costs for fallback cases (~$0.01-0.03 per image)

### Pattern 4: Offline-First with Background Sync

**What:** All detection results and pending entries are stored locally first (SQLite/MMKV). Sync to the Go backend happens in the background when connectivity is available. The app remains fully functional offline.

**When to use:** All data writes and reads should go through local storage first.

**Trade-offs:**
- Pro: App works without network; entries never lost
- Pro: Instant UI updates (no loading states for local operations)
- Con: Conflict resolution needed if same entry edited on multiple devices
- Con: Nutrition DB lookups require connectivity (or pre-cached DB)

## Anti-Patterns

### Anti-Pattern 1: Sending All Photos to Cloud for Analysis

**What people do:** Upload every food photo to a cloud API (Gemini, GPT-4o) for analysis.
**Why it's wrong:** At 5 photos/day/user, costs $0.05-0.15/day. At 10K users = $500-1500/day. Also adds 2-5s latency per photo and requires constant connectivity.
**Do this instead:** On-device YOLO first (free, 30ms). Cloud fallback only for low-confidence detections. Research from the spike validates YOLO can achieve usable accuracy for common foods.

### Anti-Pattern 2: Blocking the UI Thread with ML Inference

**What people do:** Run ML inference synchronously on the main thread, freezing the UI.
**Why it's wrong:** YOLO inference takes 20-50ms. Background gallery scanning processes many photos. UI becomes unresponsive.
**Do this instead:** All ML inference runs on a background thread/queue in the native module. Results are delivered asynchronously via promises or event emitters. Use `DispatchQueue.global(qos: .userInitiated)` on iOS, `Dispatchers.Default` on Android.

### Anti-Pattern 3: Storing Full-Resolution Photos on the Backend

**What people do:** Upload every original photo to cloud storage (GCS/S3).
**Why it's wrong:** Photos stay on the device library. Uploading duplicates wastes bandwidth and storage costs. Users expect their photos to remain local.
**Do this instead:** Store only photo metadata on the backend (local asset ID, EXIF data, detection results). The mobile app retrieves full-resolution photos from the device library when needed for display. Upload only cropped food regions when cloud fallback is needed.

### Anti-Pattern 4: Single Monolithic ML Model

**What people do:** Try to build one model that does detection + classification + OCR + weight estimation.
**Why it's wrong:** Different tasks need different model architectures and training data. A YOLO model is optimized for bounding boxes; a text recognizer is optimized for OCR. Combining them degrades all tasks.
**Do this instead:** Pipeline of specialized models: YOLO for detection, separate classifier for dish identification, platform OCR API for scale reading. Each model can be updated independently.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| USDA FoodData Central | REST API from Go backend; cache results in PostgreSQL | Free API key, 1000 req/hr; cache aggressively (food nutrient data rarely changes) |
| AFCD / CoFID / CIQUAL | Download dataset files, import into PostgreSQL | No real-time API; periodic bulk import (quarterly when DBs update) |
| Open Food Facts | REST API fallback from Go backend | Rate-limited; good for branded products and barcodes |
| Apple HealthKit | Native module via react-native-health | Requires custom dev client; write nutrition data, read exercise/weight |
| Google Health Connect | Native module via Health Connect SDK | Android equivalent to HealthKit; write nutrition, read exercise/weight |
| Gemini Vision API | REST API from Go backend (cloud fallback) | Only triggered for low-confidence detections; send cropped food regions |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| JS Layer <-> Native ML Module | Expo Modules API (async functions + event emitters) | Detection results returned as structured JSON; images passed by URI |
| Mobile App <-> Go Backend | REST API over HTTPS with JWT auth | Structured data only; no image uploads except cloud fallback crops |
| Go Backend <-> PostgreSQL | Database driver (pgx) with connection pool | Standard repository pattern; migrations managed separately |
| Go Backend <-> Nutrition APIs | HTTP client with circuit breaker + cache | Cache nutrition data locally to reduce external API calls |
| Gallery Scanner <-> Photo Library | Platform APIs (PhotoKit / MediaStore) | Requires broad photo access permission; change token for incremental scan |
| Background Task <-> OS Scheduler | BGProcessingTask (iOS) / WorkManager (Android) | OS controls scheduling; app requests but cannot guarantee timing |

## Model Serving Architecture

### On-Device (Primary — Production Target)

| Platform | Format | Framework | Hardware |
|----------|--------|-----------|----------|
| iOS | CoreML (.mlpackage) | Vision framework (VNCoreMLRequest) | Apple Neural Engine (ANE) on A14+, GPU fallback on older |
| Android | TFLite (.tflite) | TensorFlow Lite Interpreter or ML Kit | NNAPI delegate (Qualcomm Hexagon DSP, Samsung NPU), GPU delegate fallback |

**Model export pipeline (from training):**
```
YOLOv8/v11 (.pt)
    │
    ├── ultralytics export format=coreml nms=True → .mlpackage (iOS)
    │
    └── ultralytics export format=tflite → .tflite (Android)
```

**Model distribution:** Bundle models with the app binary (EAS Build) or download on first launch from a CDN. Model size for YOLOv8n: ~6MB (CoreML), ~6MB (TFLite). Acceptable for app bundle.

**react-native-fast-tflite** is a viable cross-platform option for Android. However, for iOS, a custom Expo module wrapping CoreML via the Vision framework provides better performance because CoreML can target the Apple Neural Engine directly. The recommended approach is a custom Expo module with platform-specific implementations (Swift for CoreML, Kotlin for TFLite).

**Confidence:** MEDIUM. The Expo Modules API for wrapping CoreML/TFLite is well-documented and multiple community examples exist ([Julius Hietala's React Native CoreML YOLOv8 guide](https://hietalajulius.medium.com/building-a-react-native-coreml-image-classification-app-with-expo-and-yolov8-a083c7866e85), [mrousavy/react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite)). However, the specific combination of background processing + ML inference + Expo is less commonly documented.

### Cloud Fallback (Secondary)

| Provider | Model | Cost | Latency |
|----------|-------|------|---------|
| Google Gemini | gemini-2.0-flash | ~$0.01/image | 1-3s |
| Google Gemini | gemini-2.0-pro | ~$0.03/image | 2-5s |

Called via Go backend only. Never called from client directly (API key protection).

### ONNX Runtime in Go (Alternative for Server-Side)

For any server-side inference needs (e.g., batch reprocessing), `onnxruntime-go` allows loading ONNX-exported YOLO models directly in Go. This avoids needing a separate Python microservice (per ADR-002 option 2). Use this only if server-side inference is needed; the primary path is on-device.

## Gallery Scanning Architecture

### iOS

- **PHPhotoLibrary.fetchPersistentChanges(since: token)** — incremental scanning since last known state
- **BGProcessingTask** — register a background processing task that iOS schedules during idle/charging
- **Expo Background Task (expo-background-task)** — Expo SDK 53+ wrapper around BGProcessingTask
- **Constraints:** iOS limits background processing to a few minutes per execution. The app must complete its scan within this window. For large backlogs, process in batches across multiple background sessions.
- **Permissions:** Requires `NSPhotoLibraryUsageDescription` (full photo access) and `UIBackgroundModes: processing` in Info.plist

### Android

- **MediaStore.Images with DATE_ADDED filter** — query for photos added since last scan timestamp
- **WorkManager with PeriodicWorkRequest** — schedule recurring background scans (minimum 15-minute interval)
- **Constraints:** WorkManager respects Doze Mode and App Standby Buckets. Battery-optimized devices may delay execution significantly.
- **Permissions:** `READ_MEDIA_IMAGES` (Android 13+) or `READ_EXTERNAL_STORAGE` (older)

### Foreground Trigger

In addition to background scanning, the app should offer a manual "Scan Now" button that triggers an immediate scan in the foreground. This provides deterministic behavior when the user wants to process photos right away.

**Confidence:** MEDIUM. expo-background-task is relatively new (SDK 53). PHPhotoLibrary change tokens are well-established iOS API. WorkManager is the standard Android solution. Platform differences in background execution reliability are a known challenge — iOS is more restrictive than Android.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-1k users | Single Go server, single PostgreSQL instance. On-device ML handles most inference. Cloud fallback via Gemini API directly from Go server. Nutrition DB results cached in PostgreSQL. |
| 1k-100k users | Add read replicas for PostgreSQL. Pre-cache popular nutrition lookups. Rate-limit cloud fallback per user. Consider CDN for model distribution if models are downloaded at runtime. |
| 100k+ users | Nutrition DB becomes the bottleneck (external API rate limits). Mirror USDA/AFCD data locally in PostgreSQL. Cloud fallback needs queue + rate limiter to manage Gemini API costs. Model updates via OTA (over-the-air) to avoid app store re-submission. |

### Scaling Priorities

1. **First bottleneck: Nutrition API rate limits.** USDA allows 1000 req/hr with free key. At 100 users doing 5 meals/day with 3 ingredients each = 1500 lookups/day. Solution: Cache all nutrition results aggressively (they rarely change). Pre-populate cache with common foods.

2. **Second bottleneck: Cloud fallback costs.** If 20% of photos need cloud fallback at $0.01/image, 10K users = $100/day. Solution: Improve on-device model accuracy (fine-tuning on more food datasets) to reduce fallback rate. Set per-user daily fallback limits.

## Suggested Build Order

Build order is dictated by dependencies between components. Each phase enables the next.

```
Phase 1: ML Foundation
├── Fine-tune YOLO on food datasets (Food-101, ISIA-500)
├── Export to CoreML + TFLite
├── Build Expo native module for on-device inference
└── Enables: All subsequent food detection features

Phase 2: Gallery Scanning + EXIF
├── Build gallery scanner native module (iOS + Android)
├── Implement EXIF metadata extraction
├── Implement perceptual hash deduplication
├── Local state management (scan queue, processed photos)
└── Depends on: Phase 1 (needs inference to classify photos as food/not-food)

Phase 3: Go Backend + Nutrition
├── Implement Go API server (food entries, ingredients, photos)
├── Integrate USDA FoodData Central (primary nutrition DB)
├── Build nutrition DB router (region-based routing)
├── Implement cloud ML fallback endpoint
└── Depends on: Phase 1 (needs model inference results to populate entries)

Phase 4: Entry Management UI
├── Food entry creation from scan results
├── Diary view with daily totals
├── Retrospective ingredient editing
├── Photo-linked entries with review capability
└── Depends on: Phase 2 (gallery scanning provides entries), Phase 3 (backend stores entries)

Phase 5: Scale Detection + Weight Refinement
├── Build scale OCR native module
├── Implement hypothesis branching (tared vs gross weight)
├── Container weight learning over time
└── Depends on: Phase 1 (YOLO detection), Phase 3 (backend for weight storage)

Phase 6: Health Platform + Notifications
├── Apple Health / Google Fit integration
├── Configurable end-of-day summary notifications
├── Correlation graphs (nutrition vs exercise vs weight)
└── Depends on: Phase 4 (needs food entry data to write to health platforms)
```

**Phase ordering rationale:**
- Phase 1 first because every other feature depends on having a working on-device food detection model
- Phase 2 before Phase 4 because the gallery scanner generates the data that the UI displays
- Phase 3 can partially parallel Phase 2 (backend doesn't need gallery scanning to exist)
- Phase 5 after core pipeline works because scale detection is an accuracy refinement, not core flow
- Phase 6 last because it requires a populated food diary to be useful

**Critical dependency:** The on-device ML model quality determines the entire product quality. If the fine-tuned YOLO model cannot achieve >70% accuracy on common foods, the cloud fallback will be triggered too frequently, increasing costs and latency. The model training (Phase 1) should be validated thoroughly before building the rest of the pipeline.

## Sources

- [mrousavy/react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite) — High-performance TFLite for React Native
- [Julius Hietala — Building a React Native CoreML App with YOLOv8](https://hietalajulius.medium.com/building-a-react-native-coreml-image-classification-app-with-expo-and-yolov8-a083c7866e85) — Expo module pattern for CoreML
- [Expo Background Task documentation](https://docs.expo.dev/versions/latest/sdk/background-task/) — Background processing in Expo SDK 53+
- [Expo Background Fetch documentation](https://docs.expo.dev/versions/latest/sdk/background-fetch/) — Periodic background execution
- [Apple PHPhotoLibrary documentation](https://developer.apple.com/documentation/Photos/PHPhotoLibrary) — Photo library change tracking
- [Apple BGProcessingTask documentation](https://developer.apple.com/documentation/backgroundtasks/bgprocessingtask) — iOS background processing
- [Android WorkManager documentation](https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started) — Android background task scheduling
- [Ultralytics YOLO Export](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) — CoreML/TFLite export
- [idealo/imagededup](https://idealo.github.io/imagededup/) — Perceptual hashing for image deduplication
- [react-native-exify](https://github.com/lodev09/react-native-exify) — EXIF metadata extraction for React Native
- [react-native-health](https://github.com/agencyenterprise/react-native-health) — Apple HealthKit integration for React Native
- [Expo Media Library documentation](https://docs.expo.dev/versions/latest/sdk/media-library/) — Device media library access
- [YOLO26 — arXiv 2509.25164](https://arxiv.org/html/2509.25164v3) — Latest YOLO architecture with improved edge deployment

---
*Architecture research for: on-device ML food tracking with passive gallery scanning*
*Researched: 2026-02-12*
