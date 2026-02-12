# Project Research Summary

**Project:** FoodTracker — AI-powered macro/food tracking mobile app
**Domain:** On-device ML food detection, passive gallery scanning, nutrition tracking
**Researched:** 2026-02-12
**Confidence:** MEDIUM-HIGH

## Executive Summary

FoodTracker is an AI-powered food tracking app differentiated by passive gallery scanning — automatically discovering and processing food photos from the user's photo library, eliminating the need to remember to open the app. The research validates that this zero-effort approach is technically feasible using on-device ML (YOLO26 via react-native-fast-tflite) with ~30ms inference times, making real-time food detection and gallery scanning practical without cloud costs or privacy concerns. The stack centers on React Native + Expo with custom native modules for ML inference, a Go backend for nutrition data, and HealthKit/Health Connect integration for health platform sync.

The primary technical risk is iOS's "limited photo access" permission model, which fundamentally conflicts with passive scanning. Most privacy-conscious users will grant access to only selected photos, preventing background discovery. This must be addressed architecturally in the gallery scanning phase by designing for manual selection as the primary UX with passive scanning as an opt-in enhancement. The secondary risk is model accuracy: training data bias causes 15-50% calorie errors for non-Western cuisines, and weight estimation from 2D images introduces 10-40% error even with perfect food identification. These require curated training datasets with per-cuisine validation and a mandatory manual correction flow.

The recommended approach is to start with on-device ML (Phase 1), validate detection accuracy before building UX on top of it, implement gallery scanning with explicit handling of limited photo access (Phase 2), and only then build the nutrition lookup and entry management UX (Phase 3-4). Scale detection and health platform integration are accuracy enhancers that should come after the core pipeline proves usable. This ordering prevents building elaborate UX on top of a foundation that may not work due to iOS permissions or model accuracy constraints.

## Key Findings

### Recommended Stack

The core technology stack is built around on-device ML inference to preserve privacy and minimize costs. React Native + Expo remains viable with custom native modules bridging to CoreML (iOS) and TFLite (Android). YOLO26 is the recommended detection model — released January 2026, it's purpose-built for edge deployment with 43% faster CPU inference than YOLO11 and NMS-free export that simplifies CoreML/TFLite conversion. The Go backend handles nutrition database routing (USDA FDC primary, multi-region expansion planned per ADR-004) and provides cloud fallback for low-confidence detections via Gemini Vision API.

**Core technologies:**
- **react-native-fast-tflite v2.0.0:** JSI-based zero-copy ML inference wrapper, 40-60% faster than bridge-based alternatives, uses LiteRT 1.4.0 internally with GPU delegates
- **YOLO26 (Ultralytics 8.4.8+):** On-device food detection model, NMS-free end-to-end inference, native CoreML/TFLite export, STAL (Small-Target-Aware Label Assignment) improves detection of small food items
- **expo-background-task v1.0.10:** Periodic gallery scanning using BGProcessingTask (iOS) / WorkManager (Android), replaces deprecated expo-background-fetch
- **expo-media-library v18.2.1:** Photo library access with change token tracking for incremental scanning, EXIF metadata extraction via getAssetInfoAsync
- **@kingstinct/react-native-healthkit v13.1.1:** Apple HealthKit integration for writing nutrition data (70+ types) and reading weight/exercise
- **Go backend (replaces Express):** Strongly-typed backend for nutrition DB routing, cloud ML fallback orchestration, food entry persistence

**Alternatives explicitly avoided:**
- Google ADK (being dropped per project context)
- expo-background-fetch (deprecated)
- Multimodal LLMs as primary detector (too expensive at scale: $0.02/image x 5 photos/day x 10K users = $1K/day)
- Expo Go for development (does not support native modules)

### Expected Features

Research identified a clear split between table stakes (features users assume exist) and differentiators (features that set the product apart). Missing table stakes makes the product feel incomplete; differentiators create competitive advantage.

**Must have (table stakes):**
- **Food photo recognition** — every AI tracker does this; users expect snap-and-log
- **Macro tracking dashboard** — calories, protein, carbs, fat per meal and daily totals (core function of every calorie counter)
- **Nutrition database lookup** — per-ingredient breakdowns from trusted source (USDA FDC with 400k+ foods)
- **Manual food search and entry** — even with AI, users need to correct/override; MacroFactor's 10-action fast logger is the UX benchmark
- **Recipe/meal saving** — users eat the same ~20 meals repeatedly; saving and reusing eliminates re-logging friction
- **Edit/correct logged meals** — AI will be wrong; users must be able to fix portions, swap ingredients, adjust quantities
- **Apple Health / Google Fit integration** — standard integration point; users expect weight and exercise data to flow in
- **Weight trend tracking** — users weigh themselves and expect trends (MacroFactor's trended weight algorithm is the gold standard)

**Should have (competitive advantage):**
- **Passive gallery scanning** — PRIMARY DIFFERENTIATOR. Zero-effort food logging. No competitor offers this. Users photograph food naturally; app discovers and processes photos in background. Eliminates the #1 reason people stop tracking (84% report tracking is tedious).
- **Intelligent photo deduplication** — users take multiple photos of same meal (angles, group shots); without dedup, same meal logged 3x. Requires temporal clustering (photos within N minutes at same location) and visual similarity (embedding-based).
- **Scale/weight detection via OCR** — when kitchen scale visible in photo, read weight directly. Orders of magnitude more accurate than visual estimation. iOS Vision framework provides text recognition on-device.
- **Container weight learning** — user-managed tare weights. If app knows a specific bowl weighs 350g, it can subtract tare from scale readings automatically.
- **Configurable UX modes** — three modes: zero-effort (passive scan + auto-log), confirm-only (review before logging), guided-edit (step-by-step correction). Addresses different user trust levels.

**Defer (v2+):**
- **3D LiDAR portion estimation** — SnapCalorie proves +/-80 cal accuracy vs +/-130 without LiDAR, but only works on iPhone Pro models. High engineering investment for limited device coverage.
- **Correlation graphs** — nutrition vs exercise vs weight over time. Requires stable multi-stream data from Health integration.
- **Barcode scanning** — photo-first is the differentiator; barcode scanning pulls UX toward packaged food logging (MFP's strength)
- **Micronutrient deep-dive views** — Cronometer users love this, but overwhelming for target user (fitness enthusiast tracking macros, not clinical nutritionist)

**Anti-features (deliberately excluded):**
- AI coaching / auto-adjust programs — shifts product from tracking tool to diet coach with different regulatory implications
- Social features / sharing — massive engineering investment, not the target user profile
- Meal planning / recommendations — different product entirely
- Gamification (streaks, badges) — creates anxiety-driven engagement; health tracking should reduce stress, not add it

### Architecture Approach

The architecture follows an on-device-first pattern with cloud fallback for low-confidence detections. All ML inference runs on the device using CoreML (iOS) or TFLite (Android) wrapped in custom Expo native modules. Photos never leave the device unless cloud fallback is explicitly triggered (confidence < configurable threshold, e.g., 0.6). The Go backend receives structured detection data (ingredient names, weights, EXIF metadata) but not raw images. An offline-first local state store (SQLite/MMKV) ensures the app works without connectivity.

**Major components:**
1. **Gallery Scanner (Native Module)** — monitors photo library for new images using PHPhotoLibrary change tokens (iOS) / MediaStore observers (Android), filters for food candidates, triggers processing pipeline. Handles iOS "limited access" permission model explicitly.
2. **ML Inference Bridge (Native Module)** — loads YOLO model, runs detection on images, returns bounding boxes + labels + confidence. Wraps CoreML via Vision framework (iOS) or TFLite Interpreter (Android). JSI-based zero-copy for performance.
3. **Dedup Engine** — computes perceptual hashes (pHash/dHash), detects near-duplicates (Hamming distance < threshold), groups multi-angle shots as same meal. Prevents passive scanning from creating duplicate entries.
4. **Go API Server** — REST API for food entries, recipes, user management. Routes nutrition lookups to region-appropriate databases (USDA, AFCD, CoFID). Orchestrates cloud ML fallback (Gemini Vision) for low-confidence detections.
5. **Local State Store** — SQLite or MMKV for offline-first storage. Caches scan progress, processed photo hashes, pending entries. Background sync to Go backend when connectivity available.

**Key architectural patterns:**
- **Native Module Bridge:** Expo Modules API wraps platform-specific ML frameworks (CoreML/TFLite) behind unified TypeScript API
- **Background Scanning with Change Tokens:** Platform background task APIs (BGProcessingTask/WorkManager) scan library incrementally using persisted change tokens
- **Confidence-Based Cloud Fallback:** On-device inference runs first; if confidence < threshold, crop food region and send to cloud vision model for re-analysis
- **Offline-First with Background Sync:** All writes go to local storage first; sync to backend happens in background when connectivity available

**Anti-patterns to avoid:**
- Sending all photos to cloud for analysis (cost explosion at scale)
- Blocking UI thread with ML inference (use background threads/queues)
- Storing full-resolution photos on backend (photos stay on device library)
- Single monolithic ML model (use pipeline of specialized models: YOLO for detection, separate OCR for scale reading)

### Critical Pitfalls

1. **Training data bias destroys non-Western food accuracy** — YOLO models trained on Food-101 fail dramatically on Asian, Indian, Middle Eastern cuisines. Energy overestimated for Western diets by ~249 cal but underestimated for Asian diets by ~363 cal. **Prevention:** Audit training data by cuisine region before training, supplement with region-specific datasets (IndianFood, VIREO-172, UEC-FOOD-256), compute per-cuisine mAP separately, implement confidence threshold + cloud fallback.

2. **Portion/weight estimation errors compound into unusable calorie numbers** — Even with perfect food identification, 2D image weight estimation introduces 10-40% error. Volume estimation errors span 0.10% to 38.3%. On a 2,000-calorie diet, this means 200-600 cal/day error. **Prevention:** Do NOT rely solely on bounding-box-area-to-weight regression. Implement reference object system (detect plate/bowl edges, use known diameters). Prioritize scale integration (OCR on digital scales). Present calorie estimates as ranges, not false-precision single numbers. Allow easy manual weight override.

3. **iOS photo library "limited access" breaks passive gallery scanning** — Since iOS 14, users can grant "limited" access (selected photos only). Passive scanner sees zero new photos unless user explicitly adds them via system picker. The entire passive scanning value proposition silently fails. **Prevention:** Design for limited access as PRIMARY path, not full access. Use PHPhotoLibrary.requestAuthorization(for: .readWrite) with access-level parameter. Implement manual "scan these photos" flow as primary UX, with passive scanning as enhancement for users who grant full access. Test limited access case explicitly before building ML pipeline on top.

4. **Hidden ingredients make image-only calorie tracking fundamentally inaccurate** — Cooking oils, butter, sauces, marinades are invisible in photos but account for 200-500+ cal/meal. A grilled chicken breast and a pan-fried-in-butter chicken breast look identical. Users systematically under-count calories by 15-30%. **Prevention:** Always prompt users to add cooking method and condiments after AI detection. Build "common additions" quick-add UI. For recurring meals, save full recipe including hidden ingredients. Be transparent about limitations. Cannot be solved by better ML models — requires UX that makes adding invisible ingredients fast.

5. **CoreML/TFLite model deployment fails silently on real devices** — Models that perform well in Python produce incorrect bounding boxes, crash, or run at 2 FPS after conversion. Half-precision floating point causes accuracy degradation. iPhone has only ~2-3GB available to apps — exceeding this causes termination with no crash report. **Prevention:** Convert to CoreML/TFLite EARLY before full training. Validate on-device inference outputs numerically against Python outputs. Test on oldest supported device (e.g., iPhone 12). Use YOLOv8n/YOLO26n (nano), not medium/large variants. Pin Ultralytics and CoreML tools versions.

6. **Food-to-nutrition-DB mapping is harder than food detection** — YOLO detects "rice" but USDA FDC has 200+ rice entries. Fuzzy text matching returns incorrect entries. Different DB entries for same visual food can vary by 50-100% in calories. **Prevention:** Build curated mapping table from YOLO class labels to specific nutrition DB entries (FDC IDs), not text search. Use USDA Foundation/SR Legacy datasets (research-grade) for default mappings. Implement preparation-method modifier system (base food + cooking method = specific nutrition profile). Cache and version mapping table.

## Implications for Roadmap

Based on research, the roadmap should follow a risk-retirement strategy: validate the highest-risk technical unknowns first (iOS permissions, model accuracy) before building elaborate UX on top. The suggested phase structure front-loads model training and gallery scanning to validate feasibility, then layers on nutrition lookup and entry management once the detection pipeline proves usable.

### Suggested Phase Structure

#### Phase 1: On-Device ML Foundation
**Rationale:** Nothing else works without a functional food detection model. Model quality determines entire product quality — if fine-tuned YOLO cannot achieve >70% accuracy on common foods, cloud fallback will be triggered too frequently, increasing costs and latency. This is the critical dependency that gates all other features.

**Delivers:**
- Fine-tuned YOLO26 model on food datasets (Food-101, ISIA-500, region-specific datasets)
- Export to CoreML (.mlpackage) and TFLite (.tflite)
- Custom Expo native module wrapping CoreML (iOS) and TFLite (Android)
- Validation: per-cuisine mAP >70% on stratified test set

**Addresses features:**
- Food photo recognition (table stakes)

**Avoids pitfalls:**
- Training data bias (audit cuisines before training)
- CoreML/TFLite deployment failures (convert early, validate on-device outputs numerically)

**Research flag:** Standard patterns (YOLO training well-documented). No additional research needed.

---

#### Phase 2: Gallery Scanning & Deduplication
**Rationale:** This is the PRIMARY DIFFERENTIATOR. Must validate iOS "limited photo access" handling before building the rest of the app around passive scanning. If limited access prevents passive scanning architecturally, the entire product UX must pivot to manual photo selection. This is a go/no-go decision point.

**Delivers:**
- Gallery scanner native module (iOS: PHPhotoLibrary + BGProcessingTask, Android: MediaStore + WorkManager)
- Explicit handling of iOS limited photo access (design for limited as primary, full as enhancement)
- EXIF metadata extraction (timestamp, GPS for meal context)
- Perceptual hash deduplication (pHash/dHash with Hamming distance threshold)
- Local state management (scan queue, processed photo hashes)
- Manual "scan these photos" UX as primary flow

**Addresses features:**
- Passive gallery scanning (differentiator — but with limited access handling)
- Intelligent photo deduplication (differentiator)
- EXIF metadata extraction (enables dedup and meal context)

**Avoids pitfalls:**
- iOS limited photo access breaking passive scanning (design for it from day one)
- Photo deduplication failures (temporal + visual clustering)
- Battery drain from background processing (throttle to 10-20 photos per BGProcessingTask execution)

**Dependencies:** Requires Phase 1 (ML inference to classify photos as food/not-food)

**Research flag:** NEEDS DEEPER RESEARCH. iOS photo library permissions and background task scheduling have platform-specific constraints. Recommend `/gsd:research-phase` focused on: PHPhotoLibrary limited access API patterns, BGProcessingTask scheduling constraints, WorkManager reliability on battery-optimized Android devices.

---

#### Phase 3: Go Backend & Nutrition Database
**Rationale:** Can partially parallel Phase 2. Backend doesn't need gallery scanning to exist, but detection results are needed to populate food entries. USDA FDC integration is straightforward (already prototyped in POC), but the food-to-nutrition mapping table is a data curation problem requiring significant time.

**Delivers:**
- Go API server (food entries, ingredients, photos, recipes)
- USDA FoodData Central integration (primary nutrition DB)
- Curated YOLO-class-to-FDC-ID mapping table (not fuzzy text search)
- Nutrition DB router (foundation for multi-region expansion)
- Cloud ML fallback endpoint (Gemini Vision API for low-confidence detections)
- PostgreSQL schema (users, food_entries, ingredients, photos, recipes)

**Addresses features:**
- Nutrition database lookup (table stakes)
- Manual food search and entry (table stakes — fallback when AI is wrong)
- Recipe/meal saving (table stakes)

**Avoids pitfalls:**
- Food-to-nutrition-DB mapping errors (curated mapping table, not text search)
- Cloud fallback cost explosion (per-user daily limits, queue + rate limiter)
- USDA API rate limits (cache aggressively, nutrition data rarely changes)

**Dependencies:** Requires Phase 1 (model inference results to populate entries)

**Research flag:** Standard patterns (REST API, PostgreSQL, nutrition API integration well-documented). USDA FDC API is straightforward. No additional research needed.

---

#### Phase 4: Entry Management UI
**Rationale:** Depends on both gallery scanning (Phase 2 provides detected entries) and backend (Phase 3 stores entries). This is where the user-facing value materializes — converting detected food into a usable diary.

**Delivers:**
- Food entry creation from scan results (structured data from Phase 2 + nutrition from Phase 3)
- Diary view with daily macro totals (calories, protein, carbs, fat)
- Retrospective ingredient editing (users can fix AI errors)
- Photo-linked entries with review capability
- Configurable UX mode: confirm-only (review detected meals before logging)
- Fast manual entry fallback (target: MacroFactor's 10-action benchmark)

**Addresses features:**
- Macro tracking dashboard (table stakes)
- Edit/correct logged meals (table stakes)
- Configurable UX modes (differentiator — start with confirm-only, add zero-effort after validation)

**Avoids pitfalls:**
- Hidden ingredient blindness (always prompt for cooking method + condiments after detection)
- Single calorie number with false precision (show ranges: "350-420 kcal")
- Making correction flow harder than initial entry (one-tap to edit any detected food)
- Auto-logging without user review (start with confirm-only; add zero-effort mode only after accuracy validated)

**Dependencies:** Requires Phase 2 (gallery scanning provides entries) and Phase 3 (backend stores entries and provides nutrition data)

**Research flag:** Standard patterns (React Native UI, diary views, CRUD operations). No additional research needed.

---

#### Phase 5: Scale Detection & Weight Refinement
**Rationale:** This is an accuracy multiplier, not a core flow requirement. Only add after core pipeline (detect → review → log) proves usable. Scale detection via OCR addresses the weight estimation error problem (10-40% error from 2D images) for users who weigh food.

**Delivers:**
- Scale OCR native module (iOS: VNRecognizeTextRequest, Android: ML Kit Text Recognition)
- Hypothesis branching (tared vs gross weight detection)
- Container weight learning (user-managed tare weights stored locally)
- Weight override UI (manual grams/oz entry for any food)

**Addresses features:**
- Scale/weight detection via OCR (differentiator)
- Container weight learning (differentiator)

**Avoids pitfalls:**
- Portion/weight estimation errors (scale OCR provides ground truth when available, manual override for all cases)

**Dependencies:** Requires Phase 1 (YOLO detection), Phase 3 (backend for weight storage), Phase 4 (review UI to present scale-detected weights)

**Research flag:** Standard patterns (iOS Vision framework and Android ML Kit are well-documented). No additional research needed.

---

#### Phase 6: Health Platform Integration & Notifications
**Rationale:** Depends on populated food diary (Phase 4) to be useful. HealthKit/Health Connect sync writes nutrition data and reads weight/exercise data for correlation. Notifications drive daily habit formation. Both are engagement enhancers, not core pipeline components.

**Delivers:**
- Apple Health integration (@kingstinct/react-native-healthkit)
- Google Health Connect integration (react-native-health-connect)
- Write nutrition data (70+ types to HealthKit)
- Read weight and exercise data
- Configurable end-of-day summary notifications (calories, protein, carbs, fat)
- Weight trend tracking (MacroFactor-style trended weight algorithm)

**Addresses features:**
- Apple Health / Google Fit integration (table stakes)
- Weight trend tracking (table stakes)
- End-of-day notification (engagement driver)

**Avoids pitfalls:**
- HealthKit App Store rejection (HealthKit UI visible in app, privacy policy covers health data usage, never store health data in iCloud)
- HealthKit data integrity (mark AI-estimated entries with metadata indicating estimation method and confidence)

**Dependencies:** Requires Phase 4 (needs food entry data to write to health platforms)

**Research flag:** NEEDS VALIDATION. HealthKit App Store review requirements are strict. Recommend `/gsd:research-phase` if team has no prior HealthKit submission experience. Focus: required UI elements, privacy policy wording, metadata tagging for AI-estimated data.

---

### Phase Ordering Rationale

- **Phase 1 first:** Every other feature depends on having a working on-device food detection model. Model quality is the foundation of product quality.
- **Phase 2 before Phase 4:** Gallery scanner generates the data that the UI displays. Validating iOS limited photo access handling early prevents building a UX that assumes full library access.
- **Phase 3 can partially parallel Phase 2:** Backend doesn't need gallery scanning to exist, but detection results are needed to populate entries. Can start backend API design while Phase 2 gallery scanning is in progress.
- **Phase 5 after core pipeline works:** Scale detection is an accuracy refinement, not a core flow requirement. Don't optimize weight estimation until basic detection → review → log flow proves usable.
- **Phase 6 last:** Requires populated food diary to be useful. Health platform sync and notifications are engagement enhancers, not core value delivery.

**Critical dependency chain:** Phase 1 (ML model) → Phase 2 (gallery scanning) → Phase 4 (entry UI). These three phases form the minimum viable passive food tracking pipeline. Phase 3 (backend) supports Phase 4 but can be built in parallel with Phase 2. Phase 5 (scale) and Phase 6 (health) are accuracy/engagement enhancers that layer on top.

### Research Flags

**Phases needing deeper research:**
- **Phase 2 (Gallery Scanning):** iOS PHPhotoLibrary limited access API patterns, BGProcessingTask scheduling constraints, WorkManager reliability on battery-optimized Android devices. RECOMMEND `/gsd:research-phase`.
- **Phase 6 (Health Integration):** HealthKit App Store review requirements if team has no prior HealthKit submission experience. RECOMMEND `/gsd:research-phase` if needed.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (ML Foundation):** YOLO training well-documented, Ultralytics export to CoreML/TFLite is standard.
- **Phase 3 (Backend & Nutrition):** REST API, PostgreSQL, USDA FDC integration are straightforward. Food-to-nutrition mapping is a data curation problem, not a research problem.
- **Phase 4 (Entry Management UI):** React Native UI patterns, diary views, CRUD operations are standard.
- **Phase 5 (Scale Detection):** iOS Vision framework and Android ML Kit text recognition are well-documented.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core libraries verified via npm/GitHub releases (react-native-fast-tflite v2.0.0, YOLO26, expo-background-task v1.0.10). React Native + Expo for on-device ML is validated pattern. Some version-pinning is best-effort. |
| Features | HIGH | Strong competitive evidence (analyzed 9 competitor apps). Table stakes vs differentiators clearly delineated. Passive gallery scanning validated as novel (no competitor does this). MVP feature set is well-defined. |
| Architecture | MEDIUM | On-device ML patterns are well-established (CoreML/TFLite via native modules). Gallery scanning architecture is documented but iOS limited access handling and background task reliability are known pain points. Offline-first + background sync is standard mobile pattern. |
| Pitfalls | MEDIUM-HIGH | Pitfalls sourced from academic papers, competitor failures, developer forums, Apple docs. Training data bias and weight estimation errors are well-documented problems with validated solutions. iOS photo library limited access is a known constraint with architectural implications. |

**Overall confidence:** MEDIUM-HIGH

The stack and features are high-confidence — technologies are verified, competitors analyzed, differentiation validated. The architecture and pitfalls are medium-high confidence — patterns exist, but iOS-specific constraints (photo library permissions, background task scheduling) require explicit validation. The primary unknowns are behavioral, not technical: Will users grant full photo access for passive scanning? Will they tolerate the inherent inaccuracy of image-based portion estimation?

### Gaps to Address

**Gap 1: iOS limited photo access behavioral unknown**
- **What we know:** iOS 14+ supports "limited" photo access (user selects specific photos). Technically, passive scanning only works with full library access.
- **What's uncertain:** What percentage of target users will grant full access? If most users grant limited access, passive scanning becomes a niche feature, not the primary value prop.
- **How to handle:** Architect for limited access from day one (manual photo selection as primary UX). During beta testing, instrument permission grant rates and passive vs manual scan usage. Be prepared to pivot UX if full library access adoption is low.

**Gap 2: Food detection model accuracy in production**
- **What we know:** Research shows training data bias causes 15-50% calorie errors for non-Western cuisines. Fine-tuning on region-specific datasets improves accuracy.
- **What's uncertain:** What accuracy threshold is "good enough" for users to trust the app? At what error rate do users abandon the app?
- **How to handle:** Phase 1 includes stratified evaluation with per-cuisine mAP targets (>70%). Beta testing with target user segment (Australian fitness enthusiasts) provides real-world accuracy validation. Implement cloud fallback for low-confidence detections to catch model failures.

**Gap 3: Background task scheduling reliability on iOS**
- **What we know:** BGProcessingTask is system-controlled; timing cannot be guaranteed. iOS restricts execution to a few minutes per session.
- **What's uncertain:** How frequently will background scans actually run in production? Will users perceive the app as "not working" if scans are delayed by hours/days?
- **How to handle:** Implement manual "scan now" button as primary UX. Market passive scanning as a convenience feature, not the only way to log food. During beta, instrument BGProcessingTask execution frequency and correlate with user engagement metrics.

**Gap 4: Weight estimation accuracy threshold for user trust**
- **What we know:** 2D image weight estimation introduces 10-40% error even with perfect food detection. Scale OCR and manual override reduce error but add friction.
- **What's uncertain:** Will users tolerate 20-30% calorie estimation error if correction is easy? Or will they abandon the app for manual-entry apps with higher accuracy?
- **How to handle:** Present estimates as ranges, not false-precision numbers. Make correction UX fast (one-tap edit). During beta, track user correction rates and correlate with retention. If correction rate > 50%, weight estimation model needs improvement or scale integration needs to be promoted more aggressively.

## Sources

### Primary (HIGH confidence)
- react-native-fast-tflite v2.0.0 GitHub releases — verified Jan 2026
- YOLO26 arXiv paper (2509.25164) — architectural details and benchmarks
- Expo Background Task documentation — official Expo SDK 53+ docs
- Apple PHPhotoLibrary documentation — iOS photo library change tracking
- USDA FoodData Central API documentation — official USDA docs
- @kingstinct/react-native-healthkit GitHub — v13.1.1 verified, updated Feb 2026
- React Native Vision Camera GitHub — v4.7.3 verified

### Secondary (MEDIUM confidence)
- Competitor feature analysis (MyFitnessPal, MacroFactor, Foodvisor, Lose It, SnapCalorie, Cal AI, Cronometer) — product features verified via app testing and marketing sites
- WhatTheFood accuracy research — AI calorie counter accuracy ranges, dataset bias statistics
- University of Sydney study — cultural food bias, Asian diet underestimation
- PMC academic papers — portion estimation challenges, weight estimation error ranges
- Hacker News discussions — real user complaints about AI calorie tracking apps

### Tertiary (LOW confidence)
- react-native-exify version — not independently verified, may not be needed
- Depth Anything V2 on-device feasibility — research papers only, no production React Native integration found
- LangGraph suitability for food tracking — general framework docs, no food-domain-specific validation

---
*Research completed: 2026-02-12*
*Ready for roadmap: yes*
