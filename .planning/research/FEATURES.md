# Feature Research

**Domain:** Local-first, subscription-free AI food tracking (mobile)
**Researched:** 2026-03-12
**Confidence:** MEDIUM-HIGH (competitive landscape well-documented; on-device UX patterns validated by competitors; local-first sync patterns verified with official docs)
**Supersedes:** Previous FEATURES.md (2026-02-12, pre-local-first pivot)

## Context: What Changed

ADR-005 (2026-03-12) pivoted from cloud backend to fully local-first architecture. This reshapes the feature landscape:

1. **All inference runs on-device** -- no cloud fallback, no API billing. UX must handle variable inference speed and accuracy across the device spectrum.
2. **Nutrition data is bundled** -- USDA FDC as pre-populated SQLite (~50-80MB), not runtime API. First-use experience changes (data must be available before first log).
3. **No subscription** -- zero recurring cost. Competitive differentiator against MFP ($80/yr), MacroFactor ($72/yr), Cronometer ($50/yr).
4. **Sync is optional** -- Google Drive / iCloud backup, not a central database. App must feel complete without any network connection.
5. **MFP acquired Cal AI (March 2026)** -- the dominant player has absorbed the fastest-growing AI tracker. Subscription-free is now even more differentiated.

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete or unusable. All must work fully offline.

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| **On-device food photo recognition** | Every AI tracker does this (MFP Meal Scan, MacroFactor AI, Yazio AI, Cal AI). Users expect snap-and-identify. | MEDIUM | YOLO models exported to CoreML/LiteRT | YOLO pipeline already validated (ADR-003). 5-8ms NPU, 50-80ms CPU. Must work on all devices from last ~4 years. Fine-tuned model needed beyond COCO's limited food classes. |
| **Macro tracking dashboard** | Core function of every calorie counter. Daily calories, protein, carbs, fat per meal and total. | LOW | op-sqlite for local data | Pure UI work. Every competitor has this. Zustand store already exists (`useFoodLogStore`). |
| **Bundled nutrition database lookup** | Users expect per-ingredient breakdowns from a trusted source. Cronometer (92 nutrients) sets the accuracy bar. | MEDIUM | USDA FDC SQLite DB bundled via fast-follow asset pack | Knowledge graph already built with SQLite export (`export_mobile.py`). Bundle as fast-follow pack to keep base APK < 100MB. Must be available before first food log. |
| **Manual food search and entry** | Even with AI, users must correct/override. MacroFactor's fast logger (10 actions per food) sets the UX bar. | LOW | Bundled nutrition DB, op-sqlite | Fallback when AI detection fails. Must be fast (local SQLite query, no network). Search must handle fuzzy matching, common misspellings, and alternate names. |
| **Daily food diary UI** | Users want to see what they ate organized by meal (breakfast/lunch/dinner/snacks) with per-meal and daily totals. | LOW | op-sqlite, macro dashboard | `DiaryScreen.tsx` already exists as scaffold. Timeline/card layout with photo thumbnails, ingredient list, macros per meal. Swipe to edit/delete. |
| **Editable meals** | AI will be wrong. Users must fix portions, swap ingredients, adjust quantities after logging. No app ships without this. | LOW | Food diary UI, nutrition DB | Post-log editing is non-negotiable. Include: change ingredient, adjust weight/portion, add/remove items, reassign to different meal. |
| **Recipe/meal saving and reuse** | Users eat the same ~20 meals. Saving and reusing eliminates re-logging friction. MFP, MacroFactor, Cronometer all do this. | LOW | op-sqlite, nutrition DB | One-tap reuse of saved meals. Serving size adjustment. Knowledge graph already has recipe schema. |
| **Weight trend tracking** | Users weigh themselves and expect the app to show trends. MacroFactor's trended weight (smoothing daily fluctuations) is best in class. | LOW | op-sqlite | Simple line chart with smoothing (exponentially weighted moving average). Manual weigh-in entry. Apple Health / Google Fit import is v1.x. |

### Differentiators (Competitive Advantage)

Features that set this product apart. Not expected by users, but create significant value.

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| **Zero subscription cost** | Only AI food tracker with no recurring fee. MFP ($80/yr), MacroFactor ($72/yr), Cronometer ($50/yr), Cal AI ($30/yr). SnapCalorie is free but cloud-dependent. FatSecret is free but has no AI photo recognition. | LOW (business model) | Local-first architecture (no server costs) | This IS the product positioning. Every technical decision supports this. No server = no cost to pass to users. |
| **Passive gallery scanning** | Zero-effort food logging. No competitor does this. Users photograph food naturally; app discovers and processes photos. Eliminates the #1 friction point (remembering to open the app). 84% of users find tracking tedious; 70% abandon within 2 weeks. | HIGH | On-device food classifier (binary food/not-food), EXIF metadata extraction, photo deduplication | **Primary UX differentiator.** iOS: PHAsset API with limited library access permissions. Android: MediaStore query. Must handle "Limited Photos" mode on iOS gracefully. Background processing via WorkManager (Android) / BGTaskScheduler (iOS). |
| **Intelligent photo deduplication** | Users take multiple photos of the same meal (angles, group shots, before/during). Without dedup, same meal gets logged 3x. No competitor addresses this because no competitor scans the gallery. | HIGH | EXIF metadata (timestamp, GPS), photo embeddings (visual similarity) | Structural requirement for gallery scanning -- not optional. Phase 1: temporal clustering (photos within 5 min at same GPS = same meal). Phase 2: visual similarity via lightweight embedding model. |
| **Scale/weight OCR detection** | When a kitchen scale is visible in the photo, read the weight. Orders of magnitude more accurate than visual estimation. No competitor does this. | MEDIUM | Custom 7-segment TFLite model (17KB-5MB), display region detector | ADR-005 validates custom model approach. ML Kit and Apple Vision explicitly do not support 7-segment displays. Two-stage pipeline: detect display region, classify digits. Post-process with range validation (0-5000g). |
| **Container weight learning** | User-managed tare weights. App learns that "blue bowl = 350g" and auto-subtracts from scale readings. Zero ongoing effort after first teach. | LOW | Scale OCR, op-sqlite | Simple lookup table. User teaches once per container. Reduces friction of scale-based tracking to near zero for routine meals. |
| **Configurable UX modes** | Three modes: zero-effort (auto-scan, auto-log, daily review), confirm-only (review before logging), guided-edit (step-by-step correction). No competitor offers this spectrum. | MEDIUM | Gallery scanning, food diary UI, notification system | Addresses different user trust levels. New users want confirm-only until they trust AI accuracy. Power users want zero-effort. Default: confirm-only. |
| **End-of-day summary notification** | Configurable push notification with daily macro totals, missing meals flagged, one-tap review. Simple retention driver (3x retention rate for apps using push notifications vs. not). | LOW | Local notification scheduling (Notifee), food diary data | Schedule via Notifee with daily trigger. Personalized: show actual data ("1,847 cal, 142g protein"). User-configurable time (default: 8 PM). No server needed -- purely local scheduling. |
| **Hidden ingredient inference** | When user photographs "chicken alfredo," infer butter, cream, parmesan, garlic from recipe knowledge graph. No competitor attempts this systematically. | MEDIUM | Dish identification, knowledge graph (already built, 1003 dishes) | Knowledge graph already has recipe-to-ingredient mappings. On-device lookup after dish classification. Users can override inferred ingredients. |
| **On-device privacy** | Photos and food data never leave the device unless user opts into sync. No cloud processing of food images. No analytics on eating habits. | LOW | Local-first architecture | Not a feature users search for, but a differentiator they appreciate when they discover it. Particularly relevant in the EU (GDPR) and for health-conscious users who distrust data harvesting. |

### On-Device ML UX Patterns

These are not features but critical UX patterns that must be designed for the on-device inference pipeline. Competitors using cloud inference can hide latency behind network spinners -- we cannot.

| Pattern | What It Is | Why It Matters | Implementation |
|---------|------------|----------------|----------------|
| **Confidence-based result display** | Color-coded confidence indicators: green (>=85% -- auto-accept in zero-effort mode), yellow (60-84% -- present for confirmation), red (<60% -- prompt manual entry). | Users must understand when to trust the AI and when to intervene. Without this, low-confidence results erode trust in the entire app. | YOLO confidence scores map directly. VLM outputs include confidence. Threshold-based routing to confirm/edit flows. |
| **Shimmer/skeleton loading states** | During inference (50-80ms YOLO, potentially 1-6s VLM), show skeleton placeholder of expected results layout rather than a spinner. | On-device inference is fast for YOLO but slow for VLM. Skeleton loaders reduce perceived wait time vs. spinners. Content layout shift is prevented. | Use `react-native-fast-shimmer` or Reanimated-based shimmer. Keep skeleton count low for budget devices. |
| **Progressive result reveal** | Show YOLO bounding boxes instantly (~50ms), then refine with dish names (VLM, ~1-3s), then portion estimates, then macro calculations. Each stage updates the UI incrementally. | Instant feedback (bounding boxes in <100ms) makes the app feel responsive even if full analysis takes seconds. Users see "something is happening" immediately. | Pipeline stages: (1) YOLO detection -> draw boxes, (2) classification -> label boxes, (3) portion estimate -> show weight, (4) nutrition lookup -> show macros. Each stage updates UI as it completes. |
| **Device capability indicator** | On first launch, detect device capabilities and show "Your device supports: Basic / Enhanced" with explanation of what each means. | Prevents user frustration when VLM features aren't available on budget devices. Manages expectations upfront rather than failing silently. | Runtime detection: available RAM, NPU presence (LiteRT delegate probe), chipset. Map to capability tier. Store in preferences. |
| **Batch processing progress** | During gallery scan (5-20 photos), show per-photo progress with thumbnails, not a single progress bar. "3 of 12 photos analyzed." | Gallery scanning is the primary flow. Users need to see progress and feel in control. Thermal throttling (after ~150s of continuous inference) means batches may slow down mid-process. | Process in bursts of 5 with 2-3s cooldown to avoid thermal throttling. Show completed results immediately as each photo finishes. Allow cancel mid-batch. |
| **Manual fallback always accessible** | Every AI result screen has a "Not right? Search manually" escape hatch. Manual search must be reachable in one tap from any AI result. | AI will be wrong 15-30% of the time on complex dishes. Users must never feel trapped by a wrong result. | Bottom sheet or inline "Edit" button on every detected item. Tapping opens manual food search pre-populated with AI's best guess as search query. |
| **Offline-first data freshness** | No "offline mode" banner. The app is always offline. Show subtle sync status only when sync is enabled: last synced timestamp, syncing indicator during active sync, conflict resolution prompts. | Local-first means offline IS the normal state. Showing "offline" banners creates anxiety. Only show sync status when user has explicitly opted into sync. | Sync status in settings/profile, not in the main diary UI. During active sync: small spinner in header. After sync: "Last synced: 2 min ago." Conflict: "This entry was edited on another device. Keep this version or the other?" |

### Anti-Features (Commonly Requested, Often Problematic)

Features to explicitly NOT build. Rationale documented to prevent scope creep.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Cloud-based AI fallback** | "What if on-device accuracy isn't good enough?" | Breaks the zero-cost guarantee. Requires server infrastructure, API billing, user accounts. Creates two-tier experience (free local vs. paid cloud). ADR-005 explicitly decided: low confidence = prompt user, not call cloud. | Prompt user to confirm/correct when confidence < 60%. Use correction data to improve models. Higher-capability VLM tier for flagship devices addresses accuracy gap. |
| **AI coaching / adaptive TDEE** | MacroFactor's adaptive algorithm is popular. Users want to be told what to eat. | Different product category (coach vs. tracker). Regulatory implications for health advice. MacroFactor does this better than we ever will -- compete on their weakness (friction), not their strength (coaching). | Provide data (trends, averages) and let users draw conclusions. Export to coaching apps via Apple Health / Google Fit. |
| **Barcode scanning** | MFP, Lose It, Cronometer all have it. Seems like table stakes. | Photo-first is the differentiator. Barcode scanning pulls UX toward packaged food, which is MFP's domain (20M foods). Building/maintaining barcode-to-nutrition mappings is a major ongoing cost. Open Food Facts has 3M+ products but data quality varies. | Defer to v2+. If users photograph a barcode, treat it as a signal to search branded food database. Don't build a dedicated scanner UI. |
| **Real-time camera food detection** | "Point camera at food, see nutrition overlay in AR" sounds impressive. | Adds complexity without value for the passive-scanning use case. Users review photos taken earlier, not point cameras at food in real time. Continuous camera + ML inference = battery drain + thermal throttling in minutes. | On-demand photo analysis (pick from gallery or snap). Not live camera feed. |
| **Social features / sharing** | Social accountability drives engagement in some apps. MFP has social feed. Cal AI added Groups in 2025. | Massive engineering investment (profiles, feeds, privacy, moderation). Personal health data sharing creates privacy risk. Target user is a solo fitness enthusiast, not a community seeker. Requires server infrastructure. | None. This is not the product. |
| **Meal planning / recommendations** | Natural extension of tracking. "You tracked, now here's what to eat next." | Different product entirely. Recipe recommendation engines are complex. Shifts from retrospective tracking to prospective planning. Requires server-side personalization. | Track what you eat, show patterns. Let users plan independently. |
| **Gamification (streaks, badges, points)** | Boosts engagement by ~40% per retention studies. Many apps use this. MFP added Streaks in 2026 Winter release. | Creates anxiety-driven engagement. "Red numbers when you miss targets" is what MacroFactor rejected. Health tracking should reduce stress, not add it. Target user is intrinsically motivated. | Show consistency stats neutrally (e.g., "logged 28 of 30 days") without gamifying them. No shame mechanics. |
| **Micronutrient deep-dive UI (84+ nutrients)** | Cronometer users love this. "More data is better." | Overwhelming for target user (fitness enthusiast tracking macros). UI complexity increases. Micronutrient accuracy drops significantly for AI-estimated meals vs. weighed/measured foods. Cronometer serves this niche already. | Store micronutrient data from USDA in the bundled DB. Don't build specialized UI for v1. Surface macros prominently, micronutrients as optional deep-dive in v2+. |
| **Multi-device real-time sync** | Users want seamless switching between phone and tablet. | Requires either a backend server (breaks local-first) or complex CRDT/OT sync (overkill per ADR-005). Food logging is overwhelmingly single-device. LWW sync via Google Drive handles the backup/restore case. | Google Drive / iCloud backup-and-restore. Not real-time multi-device sync. Single primary device model. |

## Feature Dependencies

```
[On-Device Inference Pipeline]
    |-- requires --> [YOLO Model Export] (CoreML for iOS, LiteRT for Android)
    |       |-- already exists --> [YOLO Training Scripts] (01-03)
    |       |-- already exists --> [VLM Benchmark] (01-04)
    |
    |-- requires --> [Inference Router] (device capability detection -> model selection)
    |       |-- enhanced by --> [Optional VLM] (SmolVLM-256M / Moondream 0.5B / Gemma 3n)
    |       |-- enhanced by --> [Gemini Nano via AICore] (opportunistic, supported devices only)
    |
    |-- feeds into --> [Dish Classification] -> [Nutrition DB Lookup] -> [Macro Dashboard]

[Gallery Scanning]
    |-- requires --> [On-Device Inference Pipeline]
    |-- requires --> [Binary Food/Not-Food Classifier] (gate before full YOLO)
    |-- requires --> [EXIF Metadata Extraction] (timestamp, GPS for meal grouping)
    |-- requires --> [Photo Deduplication] (temporal clustering, later visual similarity)
    |-- feeds into --> [Meal Review UX] (user confirms/edits detected meals)

[Local Data Storage]
    |-- requires --> [op-sqlite] (user data: entries, recipes, history)
    |-- requires --> [Bundled Nutrition DB] (USDA FDC pre-populated SQLite)
    |       |-- delivered via --> [Fast-Follow Asset Pack] (Android) / [On-Demand Resources] (iOS)
    |       |-- already exists --> [Knowledge Graph SQLite Export] (01-02)
    |-- feeds into --> [Food Diary UI] (CRUD against local SQLite)
    |-- feeds into --> [Manual Food Search] (local FTS query)

[Scale OCR]
    |-- requires --> [Custom 7-Segment TFLite Model] (not ML Kit, not Apple Vision)
    |-- requires --> [Display Region Detector] (locate scale display in photo)
    |-- enhanced by --> [Container Weight Learning] (tare weight lookup)
    |-- feeds into --> [Portion Estimation] (replaces visual estimation when scale visible)
    |       |-- already exists --> [Portion Estimator Module] (01-04)

[Optional Cloud Sync]
    |-- requires --> [Local Data Storage] (must work without sync)
    |-- uses --> [react-native-cloud-storage] (Google Drive + iCloud)
    |-- uses --> [LWW Conflict Resolution] (last-write-wins with timestamps)
    |-- independent of --> [On-Device Inference Pipeline] (sync is data only, not models)

[UX Modes]
    |-- requires --> [Gallery Scanning] (zero-effort mode depends on passive scan)
    |-- requires --> [Meal Review UX] (confirm-only and guided-edit modes)
    |-- requires --> [End-of-Day Notification] (zero-effort mode's primary review touchpoint)

[End-of-Day Notification]
    |-- requires --> [Local Notification Scheduling] (Notifee)
    |-- requires --> [Food Diary Data] (to populate summary content)
    |-- independent of --> [Cloud Sync] (purely local scheduling)
```

### Dependency Notes

- **Gallery Scanning requires Photo Deduplication:** Without dedup, passive scanning produces massive duplicate entries. This is structural, not optional. Temporal clustering alone handles ~80% of cases.
- **Scale OCR requires Custom Model:** ML Kit and Apple Vision explicitly do not support 7-segment displays. This is validated in ADR-005 research. Cannot use off-the-shelf OCR.
- **UX Modes require Gallery Scanning:** The "zero-effort" mode only functions if passive scanning and auto-logging work. Ship confirm-only first, upgrade to zero-effort when accuracy is validated.
- **Bundled Nutrition DB must be available before first food log:** If delivered as fast-follow asset pack, the app must handle the "DB still downloading" state gracefully (show shimmer, block logging until ready, show download progress).
- **Cloud Sync is fully independent of inference:** Users can sync their food diary without ever using the VLM. Sync is data transport, not feature-gating.

## MVP Definition

### Launch With (v1.0)

Minimum viable product -- validate the "photo-to-nutrition on-device" pipeline with zero recurring cost.

- [ ] **On-device YOLO food detection** -- binary food/not-food classifier + multi-class food detection via CoreML/LiteRT. Must run on devices from last ~4 years. Existing training scripts (01-03) produce the models; this phase exports and integrates them.
- [ ] **Bundled USDA nutrition database** -- pre-populated SQLite delivered as fast-follow asset pack. Knowledge graph export (01-02) provides the data pipeline. Must support fuzzy text search for manual lookup.
- [ ] **Local data storage via op-sqlite** -- food entries, recipes, user preferences. Schema migration from backend PostgreSQL design to SQLite. All CRUD operations local.
- [ ] **Food diary UI with CRUD** -- daily view organized by meal (breakfast/lunch/dinner/snacks). Per-meal and daily macro totals. Photo thumbnails. Edit/delete entries. `DiaryScreen.tsx` scaffold exists.
- [ ] **Manual food search and entry** -- fast fallback when AI is wrong. Search bundled USDA DB. Target: 5-7 taps to log a food manually (better than MacroFactor's 10).
- [ ] **Gallery scanning (manual trigger)** -- user taps "scan recent photos" to process last N hours of gallery. Not yet passive/background. Shows batch progress. Results in confirm flow.
- [ ] **Basic photo deduplication** -- temporal clustering (photos within 5 min window = same meal). GPS clustering when available. Prevents duplicate logging from multi-angle photos.
- [ ] **EXIF metadata extraction** -- timestamp and location from photo metadata for meal context and dedup.
- [ ] **Portion estimation** -- existing portion estimator module (01-04) integrated with on-device pipeline. Density tables + reference geometry + fallback chain.
- [ ] **Meal confirmation flow** -- after gallery scan or single photo analysis, present results for user review before logging. Confidence indicators (green/yellow/red). Edit/accept/reject per item.
- [ ] **Recipe saving** -- save corrected meals for one-tap reuse. Serving size adjustment.

### Add After Validation (v1.x)

Features to add once the on-device pipeline proves accurate and the core diary UX is solid.

- [ ] **Background/periodic gallery scanning** -- move from manual trigger to automatic via WorkManager (Android) / BGTaskScheduler (iOS). Trigger: users report manual scan works well and request automation.
- [ ] **Visual similarity deduplication** -- embedding-based dedup using lightweight model to catch non-temporal duplicates. Trigger: users report duplicate entries from manual scanning.
- [ ] **Scale/weight OCR detection** -- custom 7-segment TFLite model + display region detector. Trigger: user demand for better portion accuracy.
- [ ] **Container weight learning** -- tare weight lookup table. Trigger: scale OCR is shipped and users request tare management.
- [ ] **End-of-day summary notification** -- configurable push via Notifee. Daily macro summary with one-tap review. Trigger: users have enough data flowing for summaries to be useful.
- [ ] **Configurable UX modes** -- zero-effort, confirm-only, guided-edit. Trigger: AI accuracy reaches the point where auto-logging is trustworthy (>85% top-1 on common foods).
- [ ] **Optional VLM for complex dishes** -- SmolVLM-256M / Moondream 0.5B / Gemma 3n delivered via Play for On-Device AI. Trigger: YOLO accuracy insufficient for complex/mixed dishes.
- [ ] **Google Drive sync** -- backup/restore via appDataFolder. LWW conflict resolution. Trigger: users request cross-device data safety.
- [ ] **iCloud sync** -- iOS-only backup/restore via react-native-cloud-storage. Trigger: iOS users request it.
- [ ] **Hidden ingredient inference** -- knowledge graph recipe lookup after dish classification. Trigger: dish identification accuracy high enough for recipe matching.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Apple Health / Google Fit integration** -- sync weight, exercise, nutrition data. Deferred per PROJECT.md out-of-scope.
- [ ] **3D depth/LiDAR portion estimation** -- iPhone Pro only. SnapCalorie validates approach but significant engineering investment for single device family.
- [ ] **Correlation graphs** -- requires stable multi-stream data (nutrition + exercise + weight). Needs Health integration first.
- [ ] **Multi-region nutrition databases** -- AFCD, CoFID, CIQUAL as downloadable packs. USDA covers most English-speaking users initially.
- [ ] **Barcode scanning** -- only if photo-first approach proves insufficient for packaged foods.
- [ ] **WebDAV sync** -- P2 priority per ADR-005. Self-hosters want this but it's a small audience.

## Feature Prioritization Matrix

| Feature | User Value | Impl. Cost | Priority | Existing Work | Rationale |
|---------|------------|------------|----------|---------------|-----------|
| On-device YOLO detection | HIGH | MEDIUM | P1 | Training scripts (01-03), VLM benchmark (01-04) | Core value prop. Everything depends on this. |
| Bundled nutrition DB | HIGH | LOW | P1 | Knowledge graph SQLite export (01-02) | Straightforward: bundle existing export as asset pack. |
| Local data storage (op-sqlite) | HIGH | MEDIUM | P1 | Schema design from backend | Foundation for all user data. Must handle migrations. |
| Food diary UI | HIGH | LOW | P1 | DiaryScreen.tsx scaffold, Zustand store | Table stakes UI. Scaffold exists. |
| Manual search & entry | HIGH | LOW | P1 | Nutrition DB | Essential fallback. FTS5 on SQLite. |
| Gallery scanning (manual) | HIGH | MEDIUM | P1 | Photo picker component exists | Primary differentiator, even as manual trigger. |
| Basic deduplication | HIGH | LOW | P1 | -- | Structural requirement for gallery scanning. |
| EXIF extraction | MEDIUM | LOW | P1 | -- | Low effort, needed for dedup and meal context. |
| Portion estimation integration | HIGH | LOW | P1 | Portion estimator module (01-04) | Module exists; wire to on-device pipeline. |
| Meal confirmation flow | HIGH | MEDIUM | P1 | -- | Bridge between AI output and user trust. |
| Recipe saving | MEDIUM | LOW | P1 | Knowledge graph recipe schema | High reuse value. Schema exists. |
| Background gallery scanning | HIGH | MEDIUM | P2 | -- | Passive = zero-effort. But manual validates first. |
| Scale/weight OCR | MEDIUM | MEDIUM | P2 | Research done (ADR-005, doc 004) | Accuracy multiplier. Custom model needed. |
| End-of-day notification | MEDIUM | LOW | P2 | -- | Simple retention driver. Notifee + local data. |
| Optional VLM | MEDIUM | HIGH | P2 | VLM benchmark (01-04) | Complex dish accuracy. Device-tiered delivery. |
| Google Drive sync | MEDIUM | MEDIUM | P2 | Research done (doc 006) | Data safety. react-native-cloud-storage. |
| iCloud sync | MEDIUM | LOW | P2 | Research done (doc 006) | iOS expectation. Same library. |
| UX modes | MEDIUM | MEDIUM | P2 | -- | Needs stable AI accuracy first. |
| Container weight learning | LOW | LOW | P2 | -- | Simple table. Depends on scale OCR. |
| Hidden ingredient inference | MEDIUM | MEDIUM | P2 | Knowledge graph (01-02) | Accuracy boost for mixed dishes. |
| Visual similarity dedup | MEDIUM | MEDIUM | P2 | -- | Improves dedup but temporal handles 80%. |
| Health platform integration | MEDIUM | MEDIUM | P3 | -- | Deferred per PROJECT.md. |
| 3D LiDAR estimation | MEDIUM | HIGH | P3 | -- | iPhone Pro only. SnapCalorie validates. |
| Correlation graphs | MEDIUM | HIGH | P3 | -- | Needs mature multi-stream data. |
| Multi-region DBs | LOW | MEDIUM | P3 | DB configs exist (01-02) | USDA covers most users initially. |

## Competitor Feature Analysis (March 2026)

| Feature | MyFitnessPal | MacroFactor | Cronometer | Yazio | Cal AI (now MFP) | SnapCalorie | **Our Approach** |
|---------|-------------|-------------|------------|-------|-------------------|-------------|-----------------|
| **AI photo recognition** | Premium (Meal Scan + Cal AI acquisition) | AI logging (photo, label, voice, text) | Gold only (AI Photo Logging) | PRO only | Core feature (cloud AI) | Free (cloud AI + depth) | On-device YOLO + optional VLM. Free. Works offline. |
| **Passive gallery scanning** | No | No | No | No | No | No | **Primary differentiator.** Background discovery of food photos. |
| **Scale OCR** | No | No | No | No | No | No | **Novel.** Custom 7-segment model + container learning. |
| **Subscription cost** | $80/yr (Premium+) | $72/yr | $50/yr (Gold) | $45/yr (PRO) | $30/yr | Free (cloud-dependent) | **Free. Forever. No server costs.** |
| **Offline capability** | Partial (cached data) | Partial | Partial | Partial | No (cloud AI) | No (cloud AI) | **Full offline.** All features work without internet. |
| **Database size** | 20M+ (crowd-sourced, accuracy varies) | 26.5K NCC verified | 1.1M verified, 92 nutrients | Large (unspecified) | MFP database (20M) | 500K+ USDA-verified | USDA FDC bundled (~400K verified). Quality over quantity. |
| **Logging speed** | 15 actions/food | 10 actions/food (fastest manual) | 10+ actions (manual) | Manual + photo | Photo snap, instant | Photo snap, instant | Gallery auto-scan + 2-3 taps to confirm. Target: fastest overall. |
| **Privacy** | Cloud processing, analytics | Cloud processing | Cloud processing | Cloud processing | Cloud processing | Cloud processing | **On-device only.** Photos never uploaded. |
| **Deduplication** | N/A (manual logging) | N/A | N/A | N/A | N/A | N/A | **Novel.** Temporal + visual clustering. |
| **UX modes** | One mode | One mode | One mode | One mode | One mode | One mode | **Three modes.** Zero-effort, confirm-only, guided-edit. |
| **Smart scales integration** | Via Bluetooth smart scales | Via Bluetooth | Via Bluetooth | Via Bluetooth | No | No | **OCR from photo.** No Bluetooth pairing. Any scale works. |

### Competitive Positioning Summary

**Where competitors are strong (do not compete directly):**
- MyFitnessPal: 20M+ food database, 220M+ users, brand recognition, now owns Cal AI
- MacroFactor: adaptive TDEE algorithm, fastest manual food logger, coaching
- Cronometer: 92 micronutrients, lab-verified data, clinical-grade accuracy
- SnapCalorie: depth-sensor portion estimation, academic research backing

**Where we differentiate:**
1. **Zero subscription cost** -- only AI food tracker that is truly free (no cloud costs to pass on)
2. **Passive gallery scanning** -- nobody does this; inverts the "open app to log" paradigm
3. **Full offline operation** -- competitors degrade without internet; we don't
4. **Scale OCR from photos** -- nobody does this; any kitchen scale works without Bluetooth
5. **Complete privacy** -- photos and data never leave device unless user opts in

**Market timing insight:** MFP's acquisition of Cal AI (March 2026) consolidates the subscription AI tracker market. The subscription-free niche is now wider open, not smaller. MFP/Cal AI's cloud AI dependency means they cannot offer true offline or true privacy -- our structural advantages grow as the incumbents consolidate.

## Sources

### Competitor Products (verified March 2026)
- [MyFitnessPal Winter 2026 Release](https://blog.myfitnesspal.com/winter-release-2026-nutrition-tracking-updates/) -- Photo Upload, Streaks, redesigned Today tab
- [MFP acquires Cal AI](https://techcrunch.com/2026/03/02/myfitnesspal-has-acquired-cal-ai-the-viral-calorie-app-built-by-teens/) -- 15M+ downloads, $30M+ ARR, deal closed Dec 2025
- [MacroFactor AI Food Logging](https://macrofactor.com/ai-food-logging/) -- photo, label scan, voice, text description
- [Cronometer Features](https://cronometer.com/features/index.html) -- 92 nutrients, Gold tier with AI Photo Logging
- [Yazio AI Calorie Tracking](https://help.yazio.com/hc/en-us/articles/39137901903889) -- PRO-only AI photo recognition
- [SnapCalorie](https://www.snapcalorie.com/) -- free cloud AI, depth sensor, USDA-verified, ex-Google AI founders
- [Cal AI](https://www.calai.app/) -- photo + barcode + voice, now under MFP umbrella

### UX Patterns and User Research
- [Photo Food Logging UX](https://askvora.com/blog/photo-food-logging-track-macros) -- 85% accuracy with consistent logging beats 99% with inconsistent logging
- [AI Nutrition Tracker Comparison](https://www.snapeatai.com/blogs/ai-nutrition-tracker-showdown) -- friction analysis across major apps
- [Why People Quit Calorie Tracking](https://i-rakshitpujari.medium.com/why-most-people-quit-calorie-tracking-and-how-i-fixed-it-with-ai-9b450bcb650f) -- manual entry tedium, 30-50% underestimation
- [Confidence-Based UI Design](https://medium.com/design-bootcamp/designing-a-confidence-based-feedback-ui-f5eba0420c8c) -- green/yellow/red thresholds, progressive disclosure
- [Offline-First Architecture Patterns](https://developer.android.com/topic/architecture/data-layer/offline-first) -- Android official guidance
- [Push Notification Retention Impact](https://userpilot.com/blog/push-notification-best-practices/) -- 3x retention with push, personalized = 4x open rate
- [AI Food Recognition PMC Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11314244/) -- MFP 97% accuracy, AI image recognition accuracy for mixed/cultural dishes poor

### Technical Sources
- [FoodTracker: Real-time Food Detection (arXiv 1909.05994)](https://arxiv.org/abs/1909.05994) -- mobile YOLO food detection with bounding box overlay UX
- [React Native Shimmer Placeholders](https://github.com/tomzaku/react-native-shimmer-placeholder) -- skeleton loading for inference states
- [Notifee Local Notifications](https://medium.com/@ftardasti96/daily-reminder-with-push-notifications-in-react-native-expo-e69d0077a4b8) -- daily scheduling, Android channel requirements
- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) -- mobile inference landscape
- [Callstack: Local LLMs on Mobile](https://www.callstack.com/blog/local-llms-on-mobile-are-a-gimmick) -- honest assessment of on-device LLM limitations

---
*Feature research for: Local-first, subscription-free AI food tracking*
*Researched: 2026-03-12 (post-ADR-005 local-first pivot)*
*Supersedes: 2026-02-12 pre-pivot version*
