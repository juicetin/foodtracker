# Stack Research: On-Device ML Food Detection & Passive Gallery Scanning

**Domain:** AI-powered food tracking with on-device ML, passive gallery scanning, health platform integration
**Researched:** 2026-02-12
**Confidence:** MEDIUM (verified core libraries via npm/GitHub releases; some version-pinning is best-effort)

---

## Recommended Stack

### On-Device ML Inference

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| react-native-fast-tflite | ^2.0.0 | Run TFLite/LiteRT models in React Native | Built by mrousavy (same author as VisionCamera). Uses JSI with zero-copy ArrayBuffers -- 40-60% faster than bridge-based alternatives. v2.0.0 (Jan 2026) upgrades to LiteRT 1.4.0, adds Android 16KB page size support. GPU delegates via CoreML/Metal (iOS) and OpenGL/NNAPI (Android). Directly loads `.tflite` files at runtime without rebuild. **HIGH confidence** -- verified via GitHub releases. |
| react-native-vision-camera | ^4.7.3 | Camera capture + frame processors | Frame Processor plugin system allows synchronous native ML inference per-frame. Pairs with react-native-fast-tflite for realtime camera detection. Same author ecosystem ensures tight integration. **HIGH confidence** -- verified via npm. |
| vision-camera-resize-plugin | latest | Frame preprocessing (resize, crop, YUV-to-RGB) | SIMD-accelerated frame resizing required before feeding frames to TFLite models. Essential companion to VisionCamera + TFLite pipeline. **MEDIUM confidence** -- version not independently verified. |

### Food Detection Model

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| YOLO26 (Ultralytics) | ultralytics >=8.4.8 | Object detection model for food recognition | YOLO26 (released Jan 2026) is purpose-built for edge deployment: NMS-free end-to-end inference, DFL removed for clean CoreML/TFLite export, 43% faster CPU inference than YOLO11-N, and STAL (Small-Target-Aware Label Assignment) specifically improves accuracy on small/occluded objects like food items on a plate. Export to both `.mlmodel` (iOS CoreML) and `.tflite` (Android LiteRT) natively supported. **HIGH confidence** -- verified via Ultralytics GitHub releases and YOLO26 paper. |
| Roboflow Universe | N/A (platform) | Food-specific training datasets + annotation | Hosts multiple food detection datasets (4,900+ images). Supports YOLO26 training directly on platform. Use for fine-tuning on food-specific classes beyond COCO's ~10 food categories. **MEDIUM confidence** -- verified via Roboflow blog. |

### Gallery Scanning & Background Processing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| expo-media-library | ^18.2.1 (SDK 54) | Access photo gallery, query assets, subscribe to changes | Already in project. Provides `getAssetsAsync` for paginated gallery access, `addListener` for new-photo events, creation time sorting, and EXIF/GPS metadata via `getAssetInfoAsync`. Sufficient for SDK 54; upgrade path to SDK 55's object-oriented `/next` API when stable. **HIGH confidence** -- already in use, verified via Expo docs. |
| expo-background-task | ^1.0.10 | Periodic background gallery scanning | Replaces deprecated expo-background-fetch. Uses WorkManager (Android, min 15-min interval) and BGTaskScheduler (iOS). Single-worker model -- all background tasks funnel through one scheduler. Suitable for periodic "scan new photos since last check" batch processing. **HIGH confidence** -- verified via npm + Expo blog. |
| expo-task-manager | latest | Register and manage background task definitions | Required companion to expo-background-task. Provides `defineTask` and `isTaskRegisteredAsync`. **HIGH confidence** -- official Expo package. |

### EXIF Metadata Extraction

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| expo-media-library (getAssetInfoAsync) | ^18.2.1 | Primary EXIF extraction (GPS, timestamp, dimensions) | Built-in to existing dependency. Returns `exif` object with GPS coordinates, DateTimeOriginal, camera info. Requires `ACCESS_MEDIA_LOCATION` permission on Android for GPS data. Use this first before reaching for a separate library. **HIGH confidence** -- verified via Expo docs. |
| react-native-exify | latest | Fallback EXIF read/write if expo-media-library gaps found | Dedicated native EXIF reader/writer. Only add if expo-media-library's EXIF output is insufficient (e.g., missing fields, write-back needed). Actively maintained on GitHub. **LOW confidence** -- version not independently verified, may not be needed. |

### Health Platform Integration

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| @kingstinct/react-native-healthkit | ^13.1.1 | Apple HealthKit (iOS) nutrition data sync | Most actively maintained HealthKit library (updated 4 days ago as of research date). Full TypeScript support, Expo plugin config, Promise-based API close to native HealthKit naming. Supports writing `HKQuantityType` for dietary energy, macros, micronutrients. **HIGH confidence** -- verified via npm (13.1.1 published Feb 2026). |
| react-native-health-connect | ^3.5.0 | Google Health Connect (Android) nutrition data sync | Standard wrapper for Android Health Connect API. Supports NutritionRecord for logging meals with macro breakdowns. Published 2 months ago. **HIGH confidence** -- verified via npm. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| zustand | ^5.0.11 | State management | Already in project. Use for ML pipeline state (scan queue, processing status, results cache). |
| expo-file-system | ^19.0.21 | File I/O for model files and image processing | Already in project. Use for downloading updated models OTA, caching processed images. |
| expo-notifications | latest | Notify user of gallery scan results | When background scan completes and finds food photos. Local notifications only. |

### Model Training & Export (Python, development-only)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ultralytics | >=8.4.8 | Train and export YOLO26 models | `pip install ultralytics` gives access to YOLO26. Export via `model.export(format='coreml')` and `model.export(format='tflite')`. Supports INT8 quantization for smaller mobile models. |
| coremltools | >=8.0 | Post-export CoreML model optimization | Apple's official tool for CoreML model conversion and optimization. Needed for iOS-specific quantization and metadata. |
| Roboflow | N/A | Dataset management, annotation, augmentation | Web platform for managing food training datasets. Exports in YOLO format directly. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Expo Dev Client | Test native modules on device | Required -- expo-background-task, ML inference, and HealthKit do not work in Expo Go. |
| expo-dev-client | Build custom dev client | `npx expo prebuild` to generate native projects, then build with `npx expo run:ios` / `npx expo run:android`. |
| Flipper / React Native Debugger | Debug ML pipeline | Inspect TFLite model loading, inference timing, memory usage. |

---

## On-Device vs Cloud Decision Matrix

| Component | On-Device | Cloud | Recommendation | Rationale |
|-----------|-----------|-------|----------------|-----------|
| Food detection (YOLO) | ~30ms, free, private, offline | 100-500ms, $0.01-0.03/image, no offline | **On-device** | Cost at scale ($0.02 x 5 photos/day x 10K users = $1K/day), latency, privacy. YOLO26-N is small enough for mobile NPU. |
| Food classification (refine detection) | Possible with larger model | More accurate with LLM | **On-device primary, cloud fallback** | Use on-device for >80% of cases. Cloud fallback (Gemini/GPT-4o) for low-confidence detections only. |
| Weight/portion estimation | Feasible with depth model | More accurate with reference objects | **On-device primary** | Monocular depth estimation (Depth Anything V2) runs on-device. Fiducial marker (credit card) improves accuracy without cloud. |
| Nutrition lookup | N/A (needs database) | USDA FDC API | **Cloud** | Nutrition databases are large and change. Query Go backend which proxies USDA FDC. Cache common foods on-device. |
| Gallery scanning | Must be on-device | N/A | **On-device** | Photos never leave device. Privacy is non-negotiable for gallery access. |
| Health data sync | Native APIs are on-device | N/A | **On-device** | HealthKit/Health Connect are local-first APIs. Backend receives synced summaries, not raw health data. |

---

## React Native vs Native Trade-offs for ML Workloads

### Verdict: Stay with React Native + Native Modules

**Why not go fully native (Swift/Kotlin)?**

1. **react-native-fast-tflite v2.0.0 closes the gap.** JSI-based, zero-copy, GPU-delegated inference is within 5-10% of native performance. The bottleneck is the model, not the bridge.

2. **Frame processors are synchronous native code.** VisionCamera frame processors run C++/Swift/Kotlin natively -- JS only orchestrates. The actual ML inference happens at native speed.

3. **Rewriting the app is a 3-6 month detour.** The existing React Native app has navigation, state management, UI, and backend integration. Rewriting for marginal ML perf gains is poor ROI.

4. **Expo Modules API enables escape hatches.** If a specific ML pipeline needs fully native performance (e.g., custom CoreML pipeline), write an Expo Module in Swift/Kotlin and call it from JS. No need to rewrite the whole app.

**When to consider native modules (not a full rewrite):**

- Custom CoreML pipeline with model chaining (detection -> classification -> portion estimation) that needs sub-10ms latency
- Background gallery processing that needs to run as an iOS App Extension
- Advanced HealthKit integration (background delivery, workout sessions) not supported by the JS wrapper

**When to consider Nitro Modules (margelo) instead of Expo Modules:**

- If building a reusable, high-throughput native module (e.g., custom image preprocessing pipeline)
- Nitro is 13-59x faster than Expo Modules in micro-benchmarks, uses Swift <> C++ interop with zero Objective-C overhead
- However, Expo Modules have better ecosystem integration and documentation; prefer Expo Modules unless profiling proves it's a bottleneck

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| On-device inference | react-native-fast-tflite | @infinitered/react-native-mlkit | MLKit's object detection is generic (not food-specific) and you cannot load custom YOLO models. MLKit is good for barcode/text/face but not custom detection. |
| On-device inference | react-native-fast-tflite | TensorIO | Stale -- last meaningful update years ago. Fast-tflite is actively maintained by the VisionCamera author. |
| Detection model | YOLO26 | RF-DETR (Roboflow) | RF-DETR achieves higher mAP on COCO but requires GPU (transformer backbone). YOLO26 is optimized for CPU/NPU on edge devices, has cleaner TFLite/CoreML export, and 43% faster CPU inference. For mobile food detection, YOLO26 is the better trade-off. |
| Detection model | YOLO26 | YOLOv8/YOLO11 | YOLO26 has cleaner export (no DFL/NMS ops that cause CoreML issues), faster CPU inference, and better small-object detection (STAL). If YOLO26 export proves unstable, fall back to YOLO11 which is battle-tested. |
| Background tasks | expo-background-task | react-native-background-actions | expo-background-task is the official Expo replacement for background-fetch, uses modern OS APIs (BGTaskScheduler/WorkManager), and integrates with expo-task-manager. Third-party alternatives risk breaking with Expo SDK updates. |
| HealthKit (iOS) | @kingstinct/react-native-healthkit | react-native-health (agencyenterprise) | react-native-health is rewriting from Obj-C to Swift (not yet stable). @kingstinct is already Swift-based, actively maintained (updated days ago), has full TypeScript types, and Expo plugin support. |
| Health Connect (Android) | react-native-health-connect | @stridekick/react-native-health-connect | The matinzd version is the community standard with better documentation and more frequent updates. |
| Agent orchestration | LangGraph (Python, server-side) | Google ADK | ADK is being dropped per project context. LangGraph is better for stateful multi-step workflows (detect -> classify -> estimate -> lookup -> log). Use server-side only -- no agent framework needed on device. |
| EXIF metadata | expo-media-library built-in | react-native-exif / exifreader | Start with what expo-media-library provides. Only add a dedicated library if gaps are found. Fewer dependencies = less maintenance. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Google ADK (Agent Development Kit) | Being dropped from project. Tightly coupled to Gemini, limited orchestration control. | LangGraph for server-side orchestration, or direct API calls for simple pipelines. |
| expo-background-fetch | Deprecated in favor of expo-background-task as of SDK 53. | expo-background-task ^1.0.10 |
| react-native-tensorflow (shaqian) | Last updated 5+ years ago. Uses the old RN bridge, no JSI support. | react-native-fast-tflite ^2.0.0 |
| TensorFlow Lite (direct) | Rebranded to LiteRT. TF Lite Python APIs being removed from TensorFlow package. The npm packages still use "tflite" naming but the underlying runtime is LiteRT. | react-native-fast-tflite (already uses LiteRT 1.4.0 internally in v2.0.0) |
| Running ML inference on Go backend for production | Defeats purpose of on-device inference. Adds latency, cost, privacy concerns. | On-device YOLO26 via react-native-fast-tflite. Keep Go backend for nutrition DB queries, user data, and cloud fallback only. |
| Expo Go for development | Does not support native modules (TFLite, HealthKit, Background Tasks, VisionCamera). | Expo Dev Client with `npx expo prebuild`. |
| Multimodal LLMs as primary detector | $0.01-0.03 per image, 2-5s latency, no offline, no bounding boxes for portion estimation. | YOLO26 on-device primary, LLM cloud fallback for low-confidence only. |

---

## Stack Patterns by Variant

**If accuracy is the top priority (cloud fallback aggressive):**
- Run YOLO26 on-device for initial detection
- If confidence < 0.7, send image to Gemini 2.0 Flash via Go backend for classification refinement
- Use cloud-based Depth Anything V2 for portion estimation
- Higher cost, higher accuracy, requires connectivity

**If privacy/offline is the top priority:**
- YOLO26 on-device only, no cloud fallback
- On-device portion estimation with local depth model
- Cache USDA nutrition data locally (top 500 foods)
- Full offline capability, slightly lower accuracy on rare foods

**If development speed is the priority (MVP):**
- Skip on-device inference initially
- Send photos to Go backend running YOLO26 via Python microservice (ADR-002 option 3)
- Migrate to on-device after model is validated
- Fastest to ship, not production architecture

---

## Version Compatibility Matrix

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| react-native-fast-tflite@2.0.0 | React Native >= 0.76 (New Architecture) | v2.0.0 requires New Architecture (Fabric + TurboModules). RN 0.81.5 in project is compatible. |
| react-native-vision-camera@4.7.3 | React Native >= 0.73, Expo SDK >= 51 | Frame processors require Worklets (react-native-worklets-core). |
| expo-background-task@1.0.10 | Expo SDK >= 52, requires expo-task-manager | Does NOT work in Expo Go. Requires dev client. |
| @kingstinct/react-native-healthkit@13.1.1 | Expo SDK >= 49, iOS >= 13 | Requires Expo dev client (not Expo Go). Expo plugin in app.json. |
| react-native-health-connect@3.5.0 | Android SDK >= 28 (Android 9+), requires Health Connect app | Health Connect preinstalled on Android 14+. Older devices need to install from Play Store. |
| YOLO26 models (.tflite) | LiteRT >= 1.4.0, react-native-fast-tflite >= 2.0.0 | Export with `model.export(format='tflite', int8=True)` for smallest size. |
| YOLO26 models (.mlmodel) | CoreML >= 6 (iOS 16+), Xcode >= 15 | Export with `model.export(format='coreml', nms=False)` -- YOLO26 is NMS-free by design. |

---

## Installation

```bash
# On-device ML inference
npx expo install react-native-fast-tflite
npx expo install react-native-vision-camera
npm install vision-camera-resize-plugin

# Background processing
npx expo install expo-background-task expo-task-manager

# Health platforms
npm install @kingstinct/react-native-healthkit
npm install react-native-health-connect

# EXIF (only if expo-media-library is insufficient)
# npm install react-native-exify

# Already installed (verify versions)
# expo-media-library, expo-file-system, zustand
```

```bash
# Python (model training -- dev machine only)
pip install ultralytics>=8.4.8
pip install coremltools>=8.0
pip install roboflow
```

```bash
# Build dev client (required for native modules)
npx expo prebuild
npx expo run:ios
npx expo run:android
```

---

## Sources

### HIGH Confidence (verified via official releases / docs)
- [react-native-fast-tflite v2.0.0 release](https://github.com/mrousavy/react-native-fast-tflite/releases) -- GitHub releases page, verified Jan 2026
- [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera) -- GitHub, v4.7.3
- [YOLO26 paper](https://arxiv.org/html/2509.25164v4) -- arXiv, architectural details and benchmarks
- [YOLO26 overview](https://blog.roboflow.com/yolo26/) -- Roboflow, deployment guide
- [expo-background-task docs](https://docs.expo.dev/versions/latest/sdk/background-task/) -- Official Expo documentation
- [expo-background-task announcement](https://expo.dev/blog/goodbye-background-fetch-hello-expo-background-task) -- Expo blog
- [expo-media-library docs](https://docs.expo.dev/versions/latest/sdk/media-library/) -- Official Expo documentation
- [@kingstinct/react-native-healthkit](https://github.com/kingstinct/react-native-healthkit) -- GitHub, v13.1.1 verified
- [react-native-health-connect](https://github.com/matinzd/react-native-health-connect) -- GitHub, v3.5.0 verified
- [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases) -- GitHub, v8.4.8+ verified
- [LiteRT (TFLite successor)](https://github.com/google-ai-edge/LiteRT) -- Google AI Edge, official repo
- [Expo SDK 55 beta announcement](https://expo.dev/changelog/sdk-55-beta) -- Expo changelog

### MEDIUM Confidence (verified via multiple credible sources)
- [RF-DETR vs YOLO26 comparison](https://medium.com/@abrhamadamu05/yolo-26-vs-rf-detr-a-comparison-of-two-leading-object-detection-models-d9a306742201) -- Medium, cross-referenced with Roboflow blog
- [Nitro Modules benchmarks](https://github.com/mrousavy/NitroBenchmarks) -- GitHub, micro-benchmarks (real-world perf varies)
- [Food weight estimation research](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787865/) -- PMC, peer-reviewed 2026
- [Monocular food portion estimation](https://arxiv.org/html/2411.10492v1) -- arXiv, MFP3D framework
- [Expo Modules API overview](https://docs.expo.dev/modules/overview/) -- Official Expo docs
- [Roboflow food datasets](https://universe.roboflow.com/) -- Roboflow Universe, available datasets verified

### LOW Confidence (single source or unverified)
- react-native-exify version and maintenance status -- GitHub only, not independently verified
- vision-camera-resize-plugin exact version -- not independently verified
- Depth Anything V2 on-device feasibility for mobile -- research papers only, no production React Native integration found
- LangGraph suitability for food tracking agent orchestration -- general framework docs, no food-domain-specific validation

---

*Stack research for: AI-powered food tracking with on-device ML*
*Researched: 2026-02-12*
