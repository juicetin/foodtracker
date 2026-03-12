# ADR-005: Local-First, No-Subscription Architecture

**Status:** Proposed
**Date:** 2026-03-12
**Supersedes:** ADR-002 (Go backend), ADR-004 (USDA API as runtime dependency)
**Builds on:** ADR-003 (YOLO on-device inference)

## Context

The current architecture assumes a cloud backend: Express.js API server, PostgreSQL database, Google ADK agents calling Gemini 2.0 Flash, and USDA FoodData Central as a runtime API dependency. This creates ongoing costs (server hosting, API billing) that would need to be passed to users as a subscription.

Competing apps (MacroFactor, MyFitnessPal, Cronometer) all charge $7-20/month. Their Android sizes range from 75-145MB. A no-subscription model would be a significant differentiator -- but only if the app can deliver comparable accuracy without cloud inference.

Several factors make this viable now:

1. **On-device ML maturity:** YOLOv8n achieves 5-8ms inference with NPU acceleration, 50-80ms on CPU 4-core (arXiv 2511.13453). Sub-1B VLMs like SmolVLM-256M (~100MB Q4) and Moondream 0.5B (375MB INT4, 816MB runtime RAM) are production-ready on mobile. Gemma 3n E2B delivers 60-70 tok/s at 2GB RAM.
2. **ADR-003 already chose on-device YOLO** as the primary detection path. The cloud Gemini path was a fallback, not the core.
3. **Nutrition data is static:** USDA FDC, AFCD, CoFID, etc. are public datasets that update infrequently. They can be bundled rather than queried at runtime.
4. **The knowledge graph already exports to mobile-friendly SQLite** via `export_mobile.py`.
5. **Offline-first sync libraries are mature:** PowerSync, WatermelonDB, and react-native-cloud-storage are production-tested in React Native. Realm/MongoDB Device Sync hit EOL Sept 2025 -- the ecosystem has moved to SQLite-based sync.
6. **Google Play for On-Device AI** (beta) enables delivering ML models as asset packs with per-device targeting by RAM, chipset, and device model.

## Decision

Pivot to a **local-first architecture** where all inference, data storage, and core functionality runs entirely on the user's device with no backend server required.

### Inference: On-Device Only

- **Primary pipeline:** YOLO nano models (binary -> detection -> classification) via CoreML (iOS) / LiteRT (Android, successor to TFLite). Already planned per ADR-003. Performance validated: 5-8ms with NPU, 50-80ms CPU 4-core -- works on all devices from the last ~4 years.
- **Enhanced mode (optional download):** A small on-device VLM for complex dish identification, delivered via Play for On-Device AI (Android) or On-Demand Resources (iOS). Candidate models ranked by feasibility:

  | Model | Download Size | Runtime RAM | Speed | Suitability |
  |-------|--------------|-------------|-------|-------------|
  | SmolVLM-256M (Q4) | ~100MB | ~300-500MB | Fast | Best for budget devices; basic food ID |
  | Moondream 0.5B (INT4) | 375MB | 816MB | Fast | Good accuracy/size tradeoff |
  | Gemma 3n E2B | ~2GB | 2GB | 60-70 tok/s | Best quality; flagship only (8GB+ RAM) |

  Use Play for On-Device AI device targeting to deliver the right model tier automatically based on device RAM and chipset.

- **Scale OCR:** Custom 7-segment digit recognition model (TFLite, 17KB-5MB), NOT ML Kit or Apple Vision -- both explicitly do not support 7-segment/LCD displays. A two-stage pipeline: (1) detect display region, (2) classify digits with a small CNN trained on Roboflow 7-segment datasets (~948 annotated images available). Post-process with digit+decimal whitelist and range validation (0-5000g).
- **Portion estimation:** Already pure local logic -- no changes needed.
- **Gemini Nano via AICore** as an opportunistic bonus on supported devices (Pixel 8+, Galaxy S24+, expanding). Caveats: foreground-only, per-app battery quota, 1024 token limit per prompt. The Atomic Robot "Can I Eat It?" case study validates food classification is feasible via the Prompt API, though accuracy required significant prompt engineering.
- **No cloud fallback tier.** If on-device confidence is low, the app prompts the user to confirm/correct rather than silently calling a cloud API. This keeps the zero-cost guarantee honest.

### Storage: On-Device SQLite

- Replace PostgreSQL with **op-sqlite** for all structured data (8-9x faster than alternatives per benchmarks by @craftzdog).
- Migrate the existing schema (`users`, `food_entries`, `photos`, `ingredients`, `custom_recipes`, etc.) to SQLite.
- Photos stored in app-local filesystem via Expo FileSystem (already in use).
- The Express.js backend and its routes are **removed entirely** for this architecture.

### Nutrition Data: Bundled Databases

- Bundle USDA FDC (Foundation + SR Legacy subsets) as a pre-populated SQLite database. Estimated ~50-80MB compressed.
- Regional databases (AFCD, CoFID, CIQUAL) available as optional downloads via on-demand asset packs.
- Open Food Facts barcode data as an optional download for packaged food scanning.
- Database updates shipped via app updates or Play for On-Device AI delta patching.

### Sync: Pluggable, Optional

Provide optional cloud sync via a pluggable adapter pattern:

| Provider | Mechanism | Priority | Notes |
|----------|-----------|----------|-------|
| Google Drive | App data folder via REST API + react-native-cloud-storage | P0 | Cross-platform; `drive.appdata` scope (narrow permission); 15GB free quota |
| iCloud | react-native-cloud-storage (native APIs) | P1 | iOS only; expected by iOS users |
| WebDAV | Standard protocol (Nextcloud, etc.) | P2 | Self-hosters |

Sync is **not required** for the app to function. All sync is user-initiated or opt-in background sync.

**Conflict resolution:** Last-write-wins (LWW) with server timestamps. This is sufficient for food logging data -- records are single-user, append-heavy, with occasional edits. CRDTs are overkill (validated by Cinapse case study: structured record-oriented data doesn't benefit from per-field CRDT tracking). Soft-delete with `is_deleted` + `deleted_at` for deletion propagation.

If a managed sync layer is later needed, **PowerSync** (free self-hosted Open Edition, or cloud with free tier) is the top recommendation -- purpose-built for SQLite sync with React Native SDK, server-authoritative model, handles up to 1M rows per client.

### Resulting Architecture

```
Mobile App (React Native / Expo)
├── ML Inference Layer
│   ├── YOLO Pipeline (CoreML / LiteRT)        <- primary, ~20MB
│   ├── Optional VLM (tiered by device)         <- Play AI pack / On-Demand Resources
│   │   ├── SmolVLM-256M (budget devices)       <- ~100MB
│   │   ├── Moondream 0.5B (mid-range)          <- ~375MB
│   │   └── Gemma 3n E2B (flagship)             <- ~2GB
│   ├── Gemini Nano via AICore (when available)  <- zero-size, system-provided
│   └── Custom 7-segment OCR (TFLite)           <- scale reading, <5MB
├── Data Layer
│   ├── op-sqlite                               <- user data
│   ├── Bundled Nutrition DB (SQLite)           <- USDA FDC, ~50-80MB
│   ├── Optional Regional DBs                   <- on-demand asset packs
│   └── Expo FileSystem                         <- photo storage
└── Sync Layer (optional)
    ├── Google Drive adapter
    ├── iCloud adapter
    └── WebDAV adapter
```

## Consequences

### Positive

- **Zero recurring cost** for users and for us (no servers, no API billing).
- **Privacy by default** -- photos and food data never leave the device unless the user opts into sync.
- **Offline-first** -- full functionality without internet.
- **Simpler operations** -- no backend to deploy, monitor, or scale.
- **Competitive differentiator** -- only subscription-free AI food tracker in the market.
- **Eliminates the Express backend, PostgreSQL, and Google ADK service** -- significant reduction in codebase complexity.

### Negative

Each concern below has been validated with research. See Appendix for sources.

#### 1. Android Device Fragmentation (HIGH RISK)

**The problem:** ~30-35% of active Android devices have 4GB RAM or less. The 2026 DRAM shortage is reversing the upward trend -- TrendForce reports budget phones reverting to 4GB, mid-range capping at 8GB. Android Go powers 250M+ monthly active devices.

**Chipset diversity:** MediaTek holds 43% of Android SoC share, Qualcomm 34%, UNISOC 12%. UNISOC chips (3.2 TOPS NPU) have minimal ML acceleration. ~30-40% of active Android devices lack meaningful NPU capability (<5 TOPS), making CPU/GPU delegates essential fallbacks.

**Impact on our pipeline:**
- YOLO nano: **Low risk.** 5-8ms with NPU, 50-80ms on CPU 4-core. Works on all devices from the last ~4 years.
- Optional VLM: **High risk.** Creates a two-tier experience. SmolVLM-256M mitigates this (runs on 4GB devices), but Gemma 3n E2B is flagship-only.

**Mitigations:**
1. **Tiered model delivery via Play for On-Device AI.** Device targeting by RAM and chipset delivers the right model automatically: SmolVLM-256M for <=4GB, Moondream 0.5B for 6GB, Gemma 3n for 8GB+. Individual AI pack limit is 1.5GB, total app limit 4GB.
2. **Gemini Nano via AICore** on supported devices (Pixel 8+, Galaxy S24+, expanding to OnePlus, Xiaomi, HONOR). Zero app size cost, hardware-optimized. But: foreground-only, battery quota limited, 1024 token/prompt cap.
3. **Set a minimum spec and be transparent.** Display a "your device supports: Basic / Enhanced" indicator based on runtime capability detection rather than silently degrading.
4. **Use LiteRT (not NNAPI)** for inference delegation. NNAPI was deprecated in Android 15. LiteRT provides vendor-specific NPU delegates (Qualcomm QNN, MediaTek NeuroPilot) with automatic SDK download. 64/72 tested models fully delegate to NPU on Snapdragon 8 Elite.

#### 2. Thermal Throttling During Batch Processing (MEDIUM RISK)

**The problem:** Thermal throttling triggers after ~2.5 minutes of continuous inference (arXiv 2503.21109). CPU frequency drops from 3GHz to as low as 1GHz. Results in up to 4.3x performance degradation. Our batch processing feature (up to 20 photos) could trigger this on sustained runs.

**Mitigations:**
1. **Bursty inference pattern.** Process in short bursts with cooldown pauses rather than continuous inference. For 20 photos: process 5, pause 2-3s, process 5 more.
2. **Adaptive batch sizing.** Detect device thermal state at runtime (Android Thermal API) and reduce batch concurrency when temperature rises.
3. **GPU/NPU offloading.** GPU-accelerated inference runs at 1.3W/60C vs CPU at 10-12W/90-95C (OnePlus 13R case study). Prefer NPU/GPU delegates to avoid thermal hotspots on CPU.
4. **Progress UI.** Show per-photo progress during batch processing so users understand why it's sequential rather than instant.

#### 3. Scale OCR Without Cloud Vision (MEDIUM RISK)

**The problem:** Both Google ML Kit Text Recognition v2 and Apple Vision (`VNRecognizeTextRequest`) explicitly do not support 7-segment/LCD display reading. Apple's own documentation recommends training a custom detector. ML Kit frequently confuses digits (5/6, 8/3, 7/1) on segmented displays and unreliably handles decimal points.

**Cloud Vision is better but not reliable either.** Google Cloud Vision API achieves 98-99% on printed text but has no specific support for digital displays. Gemini exhibits hallucination in OCR output -- it may silently edit digits.

**Mitigations:**
1. **Custom TFLite 7-segment model.** Purpose-built models outperform all general-purpose OCR for this task. Roboflow has 948+ annotated 7-segment images. Edge Impulse has produced models as small as 17KB (INT8 TFLite). Two-stage pipeline: detect display region -> classify digits.
2. **Post-processing.** Whitelist digits + single decimal only. Validate range (kitchen scales: 0-5000g). Reject implausible readings.
3. **Manual entry fallback.** Always allow manual weight input. This is necessary regardless of OCR quality.
4. **The `wetr` project** (github.com/jacnel/wetr) demonstrates this approach for kitchen scales specifically: Tesseract with digit+decimal whitelisting. Character height of ~50px minimum for reliable decimal detection.

#### 4. Reduced VLM Accuracy vs Cloud (MEDIUM RISK)

**The problem:** On-device VLMs (0.5-3B params) are significantly less capable than cloud models (Gemini Flash/Pro) on complex dishes. A 2025 FoodNExTDB study found open-source VLMs significantly lag behind closed-source models on food recognition. All models struggle with fine-grained cooking style differences and visually similar foods. Fast food is the most challenging category.

**Mitigations:**
1. **Domain-specific fine-tuning/distillation.** The JDNet precedent achieved 91.2% Top-1 on Food-101 and 84.0% on UECFood-256 in a model 4x smaller than its teacher via multi-stage knowledge distillation. VL2Lite (CVPR 2025) demonstrates task-specific VLM distillation outperforming traditional knowledge distillation.
2. **YOLO as the workhorse.** A well-tuned YOLO classifier on food-specific datasets handles common foods without needing a VLM at all. The VLM is for edge cases.
3. **User correction loop.** When confidence is low, prompt the user to confirm/correct. Build a feedback dataset for continuous model improvement.

#### 5. App Size Impact on Installs (MEDIUM RISK)

**The problem:** Google's research shows every 6MB increase in APK size drops install conversion by ~1%. In emerging markets, removing 10MB correlates with ~2.5% more installs. The 200MB threshold triggers Wi-Fi warnings on both platforms. Competitor food trackers on Android range from 75-145MB.

**Estimated footprint:**
| Component | Size | Delivery |
|-----------|------|----------|
| Base app (React Native + YOLO models) | ~60-80MB | Install-time |
| Bundled USDA nutrition DB | ~50-80MB | Fast-follow AI pack |
| SmolVLM-256M (budget tier) | ~100MB | On-demand AI pack |
| Moondream 0.5B (mid-range) | ~375MB | On-demand AI pack |
| Gemma 3n E2B (flagship) | ~2GB | On-demand AI pack |
| Regional nutrition DBs | ~20-50MB each | On-demand AI pack |

**Mitigations:**
1. **Keep base APK under 100MB.** Ship YOLO models in-app; deliver nutrition DB as fast-follow (auto-downloads after install, ready in seconds). Base download stays competitive with Cronometer (75-100MB Android).
2. **Play for On-Device AI asset packs.** Supports install-time, fast-follow, and on-demand delivery with automatic delta patching for updates. Device targeting delivers appropriately-sized models per device.
3. **Apple On-Demand Resources / Background Assets API** for iOS equivalent.
4. **Emerging market sensitivity.** 46-71% of apps are uninstalled within 30 days in developing markets (AppsFlyer). Android users uninstall at 2x the rate of iOS users due to storage constraints. Keeping installed size lean reduces churn.

#### 6. Sync Complexity (LOW RISK)

**The problem:** Offline-first sync is inherently harder than a central server of truth.

**Research validated that this is manageable for food logging:**
- LWW is sufficient for ~95% of apps where users work with their own data (practitioner consensus).
- Food logs are single-user, append-heavy, with occasional edits -- the simplest sync scenario.
- CRDTs are overkill and add complexity without benefit for record-oriented data (Cinapse case study).

**Library options:**
- **Google Drive `appDataFolder`:** Hidden per-app folder, `drive.appdata` scope (narrow permission), 15GB free. File-level sync (coarse-grained but simple). Use react-native-cloud-storage (v2.3.0, maintained, supports both Google Drive and iCloud).
- **PowerSync** (if row-level sync is later needed): Free self-hosted Open Edition. Streams from Postgres into client SQLite. React Native SDK with reactive queries. Handles up to 1M rows per client.
- **WatermelonDB:** Battle-tested across 15+ apps, 500K+ users, 99.8% sync success rate. Requires building your own sync endpoint.

#### 7. iOS vs Android Performance Parity (LOW RISK)

**The problem:** iOS has a ~30-50% performance advantage for on-device LLM/VLM inference due to superior memory bandwidth and tighter hardware-software integration. iPhone 17 Pro achieves 136 tok/s vs Galaxy S25 Ultra at 91 tok/s (Cactus v1 benchmark).

**Why this is low risk:** The YOLO pipeline (primary path) runs in single-digit milliseconds on both platforms. The performance gap only matters for the optional VLM, where both platforms deliver acceptable latency for a non-real-time food identification task. Vision encoding is the bottleneck (~6s/image for larger VLMs), not text generation.

### Open Question: Conditional Cloud Fallback Tier

This ADR commits to local-first as the default. However, if real-world testing reveals that a significant segment of devices cannot achieve sufficient on-device accuracy to be useful, we should revisit adding an optional cloud inference tier -- but only if:

1. **The affected device segment represents actual users** who would provide sufficient revenue to justify the added complexity (server costs, API billing, backend maintenance).
2. **The accuracy gap is material** -- i.e., on-device inference fails on common foods, not just edge cases.
3. **The cloud tier is opt-in** and clearly communicated (e.g., "Enhanced Cloud Mode - requires internet"), preserving the zero-cost local-first guarantee for users who don't need it.

This is not a pre-commitment to build a cloud fallback. It is a commitment to measure on-device accuracy across the device spectrum and make a data-driven decision about whether one is needed. The measurement criteria and thresholds should be defined during model evaluation (Phase 3).

### ADRs Affected

- **ADR-002 (Go backend + Python ML inference):** Superseded. No backend server needed. Python training scripts remain for model development, but inference is on-device via CoreML/LiteRT.
- **ADR-003 (YOLO-based detection):** Reinforced. YOLO becomes the *only* inference path, not just the primary one.
- **ADR-004 (USDA FDC as API):** Superseded for runtime. USDA data is now bundled as a static SQLite database rather than queried via API. The API may still be used in the build pipeline to refresh bundled data.

## Appendix: Research Sources (March 2026)

### Android Fragmentation
- Google "Shrinking APKs, Growing Installs" study (6MB = ~1% conversion drop)
- Counterpoint Research Q1 2025: MediaTek 36%, Qualcomm 28%, UNISOC 10% global SoC share
- TrendForce: 2026 DRAM shortage reverting budget phones to 4GB RAM
- Android Go Edition: 250M+ MAUs across 16,000+ device models
- arXiv 2511.13453 (Gherasim, Nov 2025): YOLOv8n benchmarks on Samsung Galaxy Tab S9

### Thermal Throttling
- arXiv 2503.21109 (March 2025): throttling at ~150s, up to 4.3x degradation, CPU 3GHz->1GHz
- arXiv 2410.03613: LLM-specific thermal data on mobile
- OnePlus 13R case study: GPU 1.3W/60C vs CPU 10-12W/90-95C

### On-Device VLMs
- SmolVLM paper (arXiv 2504.05299): 256M-2.2B parameter family
- Moondream 0.5B: 375MB INT4, 816MB runtime RAM, QAT-optimized
- Gemma 3n E2B: 5B total / 2B effective, 60-70 tok/s, 2GB RAM
- JDNet (IEEE 2020): 91.2% Food-101 via multi-stage knowledge distillation
- VL2Lite (CVPR 2025): task-specific VLM distillation
- FoodNExTDB (arXiv 2504.06925, 2025): VLM food recognition benchmarks
- MobileAIBench (arXiv 2406.10290): multimodal TTFT >60s on typical mobile VLMs

### Gemini Nano / AICore
- ML Kit GenAI Prompt API (alpha, Oct 2025): multimodal image+text input
- Atomic Robot "Can I Eat It?" case study: food classification via Gemini Nano
- Supported devices: Pixel 8+, Galaxy S24+, OnePlus 13, Xiaomi 15, expanding
- Limitations: foreground-only, per-app battery quota, 1024 tokens/prompt, 4096 session context

### Scale OCR
- Apple Developer Forums: VNRecognizeTextRequest explicitly unsupported for LCD/LED digits
- ML Kit Text Recognition v2: trained on printed/handwritten text, not 7-segment
- Roboflow Universe: 948+ annotated 7-segment display images
- Edge Impulse: 7-segment models at 17KB INT8 TFLite
- jacnel/wetr: kitchen scale OCR app using Tesseract with digit whitelisting

### Sync Patterns
- MongoDB Atlas Device Sync: EOL September 30, 2025
- PowerSync: free self-hosted Open Edition, React Native SDK, 1M rows/client
- WatermelonDB: 15+ apps, 500K+ users, 99.8% sync success rate
- react-native-cloud-storage v2.3.0: unified Google Drive + iCloud API
- Cinapse case study: moved from CRDTs to PowerSync for structured data
- Google Drive appDataFolder: hidden per-app folder, 15GB free, 20K API calls/100s

### App Size
- Google Play: 6MB increase = ~1% install conversion drop; 2.5x impact in emerging markets
- Competitor sizes (Android): Cronometer 75-100MB, MacroFactor 87-102MB, MyFitnessPal 97-145MB, Yazio 52-74MB
- Play for On-Device AI: 1.5GB per AI pack, device targeting by RAM/chipset/model
- AppsFlyer 2024: 46-71% app uninstall rate within 30 days in emerging markets

### Inference Frameworks
- llama.cpp: VLM support merged May 2025, most mature cross-platform option
- ExecuTorch 1.0 GA (Oct 2025): 50KB base runtime, 12+ hardware backends
- LiteRT (successor to TFLite): vendor NPU delegates for Qualcomm QNN, MediaTek NeuroPilot
- NNAPI deprecated in Android 15, LiteRT is the migration path
