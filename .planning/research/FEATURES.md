# Feature Research

**Domain:** AI-powered food/nutrition tracking mobile apps
**Researched:** 2026-02-12
**Confidence:** MEDIUM-HIGH (strong competitive evidence, some on-device feasibility claims need validation)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete or unusable.

| Feature | Why Expected | Complexity | On-Device Feasible | Notes |
|---------|--------------|------------|-------------------|-------|
| **Food photo recognition** | Every AI tracker (Cal AI, Foodvisor, Lose It, SnapCalorie) does this. Users expect snap-and-log. | MEDIUM | YES (YOLO on-device ~30ms) | ADR-003 validates YOLO approach. Need fine-tuned model beyond COCO's ~10 food classes. Food-101/ISIA-500 training required. |
| **Macro tracking dashboard** | Core function of every calorie counter. Calories, protein, carbs, fat display per meal and daily total. | LOW | N/A (UI only) | Every competitor has this. Not doing it = not a food tracker. |
| **Nutrition database lookup** | Users expect to see per-ingredient breakdowns from a trusted source. Cronometer sets the accuracy bar (84 micronutrients, lab-verified). | MEDIUM | Partial (cache locally, API for lookup) | ADR-004: USDA FDC primary, multi-region (AFCD, CoFID, CIQUAL, OFF) for production. 400k+ foods via USDA alone. |
| **Manual food search & entry** | Even with AI, users need to correct/override. MacroFactor's fast logger (10 actions vs MFP's 15) sets the UX bar. | LOW | YES | Fallback when photo recognition fails. Must be fast. MacroFactor proves minimal-tap UX is critical. |
| **Daily/weekly nutrition summary** | Users want to see totals and trends. Every app from MFP to Cronometer provides this. | LOW | N/A (UI only) | End-of-day summary notification is in PROJECT.md active scope. |
| **Recipe/meal saving** | Repeat meals are common (users eat the same ~20 meals). Saving and reusing eliminates re-logging friction. | LOW | YES | Already validated in existing backend (recipe CRUD exists). |
| **Edit/correct logged meals** | AI will be wrong. Users must be able to fix portions, swap ingredients, adjust quantities. No app ships without this. | LOW | N/A (UI only) | Already validated (retrospective ingredient editing exists). |
| **Apple Health / Google Fit integration** | Standard integration point. MacroFactor, Lose It, Cronometer, Foodvisor all do this. Users expect weight and exercise data to flow in. | MEDIUM | N/A (platform API) | HealthKit writes nutrition data (70+ types). Apple Health getting major 2026 redesign with AI agent and meal tracking. Being a good HealthKit citizen matters more than ever. |
| **Weight trend tracking** | Users weigh themselves and expect the app to show trends. MacroFactor's trended weight (smoothing daily fluctuations) is the gold standard. | LOW | YES | MacroFactor's algorithm is the benchmark: imputation for missed weigh-ins, noise smoothing, weekly trends. |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not expected by users, but create significant value when present.

| Feature | Value Proposition | Complexity | On-Device Feasible | Notes |
|---------|-------------------|------------|-------------------|-------|
| **Passive gallery scanning** | Zero-effort food logging. No app currently does this. Users photograph food naturally; app discovers and processes photos in the background. Eliminates the single biggest friction point (remembering to open the app). | HIGH | YES (photo library access + on-device classification) | **Primary differentiator.** No competitor offers this. iOS PHPicker/PHAsset APIs allow limited library access but require user permission for full library scan. Privacy design is critical. Must handle "Limited Photos Library" mode gracefully. |
| **Intelligent photo deduplication** | Users take multiple photos of the same meal (angles, group shots, before/during/after). Without dedup, same meal gets logged 3x. Only GrubCircle addresses dedup (for multi-user), nobody does single-user temporal/visual dedup. | HIGH | Partial (embedding similarity on-device, clustering logic on-device) | Requires photo embeddings, temporal clustering (photos within N minutes at same location = same meal), and visual similarity. Novel problem in this space. |
| **Scale/weight detection via OCR** | When a kitchen scale is visible in the photo, read the weight directly. Orders of magnitude more accurate than visual estimation. No competitor does this. | MEDIUM | YES (OCR on-device via Vision framework on iOS) | iOS Vision framework provides text recognition. Need to detect scale display region, extract numeric value, handle unit conversion (g/oz/lb). Container weight learning over time is a unique twist. |
| **Container weight learning** | User-managed tare weights. If the app knows a specific bowl weighs 350g, it can subtract tare from scale readings automatically. Builds accuracy over time with zero ongoing effort. | LOW | YES (local storage) | Depends on scale detection. Simple lookup table per recognized container. User teaches the app once per container. |
| **Correlation graphs (nutrition vs exercise vs weight)** | Most apps silo nutrition, exercise, and weight data. Showing correlations (e.g., "you lose weight faster in weeks where protein > 150g") creates unique insight. MacroFactor's adaptive TDEE is adjacent but doesn't visualize cross-domain correlations. | MEDIUM | N/A (UI + analytics) | Requires HealthKit exercise data + weight data + nutrition data. Time-series visualization with configurable windows. |
| **Configurable UX modes** | Three modes: zero-effort (passive scan, auto-log, daily summary), confirm-only (review detected meals before logging), guided-edit (step-by-step correction). No competitor offers this spectrum. | MEDIUM | N/A (UI only) | Addresses different user trust levels. New users want confirm-only until they trust the AI. Power users want zero-effort. |
| **3D depth/LiDAR portion estimation** | SnapCalorie proves this works: +/-80 cal on 500 cal dish with LiDAR vs +/-130 without. 2x more accurate than visual estimation. No other competitor uses depth sensors. | HIGH | YES (ARKit/LiDAR on iPhone Pro) | Only works on iPhone Pro models with LiDAR. Graceful fallback to visual estimation on other devices. SnapCalorie validates the approach but is a photo-first app, not gallery-scanning. |
| **Hidden ingredient inference** | When user photographs "chicken alfredo," infer butter, cream, parmesan, garlic even though they're not visible. Use recipe knowledge graph. No competitor attempts this systematically. | MEDIUM | Partial (recipe DB local, inference logic could be on-device or cloud) | Requires dish identification -> recipe lookup -> ingredient list. Accuracy depends on recipe database coverage. Users can override. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems. Deliberately excluded.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **AI coaching / auto-adjust programs** | MacroFactor's adaptive algorithm is popular. Users want to be told what to eat. | Shifts product from tracking tool to diet coach. Different regulatory implications (health advice). Creates opinionated product that alienates users with specific dietary philosophies. MacroFactor does this well already -- don't compete on their strength. | Provide the data (correlations, trends, TDEE estimates) and let users draw conclusions. Export to apps that do coaching. |
| **Social features / sharing** | Social accountability drives engagement in some apps. MFP has social feed. | Massive engineering investment (profiles, feeds, privacy, moderation). Personal health data sharing creates privacy risk. Target user is a solo fitness enthusiast, not a community seeker. | None. This is not the product. |
| **Meal planning / recommendations** | Natural extension of tracking. "You tracked, now here's what to eat next." | Shifts from retrospective tracking to prospective planning. Different product entirely. Recipe recommendation engines are complex and highly competitive (Mealime, MFP Premium+). | Track what you eat, show patterns. Let users plan independently. |
| **Barcode scanning** | MFP, Lose It, Cronometer all have it. Seems like table stakes. | Photo-first is the differentiator. Barcode scanning pulls UX toward packaged food logging, which is MFP's strength. Building barcode infra (maintaining barcode-to-nutrition mappings) is a maintenance burden. | Defer to v2+. If users photograph a barcode, treat it as a signal to search the branded food database. Don't build a dedicated scanner. |
| **Gamification (streaks, badges, points)** | Boosts engagement by 40% per retention studies. Many apps use this. | Creates anxiety-driven engagement. "Red numbers when you miss targets" is exactly what MacroFactor rejected. Health tracking should reduce stress, not add it. Fitness enthusiast target user is intrinsically motivated. | Show consistency stats (e.g., "logged 28 of 30 days") without gamifying them. Neutral tone. |
| **Real-time camera food detection** | "Point camera at food, see nutrition overlay" sounds impressive. AR nutrition overlay is a trend. | Adds complexity without clear value for the passive-scanning use case. Users are not pointing cameras at food in real time -- they're reviewing photos taken earlier. Battery drain from continuous camera + ML inference. | On-demand photo analysis (snap or pick from gallery). Not live camera feed. |
| **Micronutrient deep-dive (84 nutrients)** | Cronometer users love this. Seems like "more is better." | Overwhelming for target user (fitness enthusiast tracking macros, not clinical nutritionist). UI complexity increases. Data accuracy drops for micronutrients in AI-estimated meals. | Track macros (calories, protein, carbs, fat) as primary. Store micronutrient data from USDA but don't surface it prominently. Add micronutrient views as optional deep-dive. |

## Feature Dependencies

```
[Gallery Scanning]
    |-- requires --> [Food Photo Classification] (is this a food photo?)
    |       |-- requires --> [YOLO Food Detection Model]
    |       |-- enhances --> [EXIF Metadata Extraction] (timestamp, location for meal context)
    |
    |-- requires --> [Photo Deduplication] (don't log same meal 3x)
    |       |-- requires --> [Photo Embedding Model] (visual similarity)
    |       |-- requires --> [EXIF Metadata Extraction] (temporal/spatial clustering)
    |
    |-- feeds into --> [Meal Review UX] (user confirms/edits detected meals)

[YOLO Food Detection]
    |-- feeds into --> [Dish Identification] (what food is this?)
    |       |-- feeds into --> [Hidden Ingredient Inference] (recipe knowledge graph)
    |       |-- feeds into --> [Nutrition Database Lookup] (per-ingredient macros)
    |
    |-- feeds into --> [Portion Estimation] (how much food?)
            |-- enhanced by --> [Scale/Weight Detection via OCR]
            |       |-- enhanced by --> [Container Weight Learning]
            |-- enhanced by --> [3D Depth/LiDAR Estimation]
            |-- fallback --> [Visual Size Estimation] (reference objects, plate size)

[Nutrition Database Lookup]
    |-- requires --> [USDA FDC Integration] (primary)
    |-- enhanced by --> [Multi-Region DB Routing] (AFCD, CoFID, CIQUAL, OFF)
    |-- feeds into --> [Macro Tracking Dashboard]
    |-- feeds into --> [Daily/Weekly Summary]

[Health Platform Integration]
    |-- requires --> [HealthKit/Google Fit API]
    |-- feeds into --> [Weight Trend Tracking]
    |-- feeds into --> [Correlation Graphs] (nutrition vs exercise vs weight)

[Configurable UX Modes]
    |-- requires --> [Meal Review UX]
    |-- requires --> [Gallery Scanning]
    |-- requires --> [Notification System]
```

### Dependency Notes

- **Gallery Scanning requires Food Photo Classification:** The first step is knowing which gallery photos contain food. A lightweight binary classifier (food/not-food) runs before full YOLO detection to save compute.
- **Gallery Scanning requires Photo Deduplication:** Without dedup, passive scanning will produce massive duplicate entries. This is not optional for the passive approach -- it is structural.
- **Scale Detection enhances Portion Estimation:** Scale detection is an accuracy multiplier but not required. Portion estimation must work without a scale visible (visual estimation fallback).
- **Correlation Graphs require Health Platform Integration:** Cannot correlate nutrition with exercise/weight unless those data streams are connected.
- **Configurable UX Modes require Gallery Scanning + Review UX:** The "zero-effort" mode only works if passive scanning and auto-logging are functional.

## MVP Definition

### Launch With (v1)

Minimum viable product -- validate the core "passive photo-to-nutrition" pipeline.

- [ ] **On-device food photo classification** -- binary "is this food?" classifier on gallery photos. Gate for everything else.
- [ ] **YOLO food detection with fine-tuned model** -- identify specific food items in detected food photos. Must exceed 80% accuracy on test set (ADR-003 target).
- [ ] **Basic portion estimation** -- visual estimation using plate/bowl size as reference. Not perfect, but functional.
- [ ] **USDA FDC nutrition lookup** -- convert detected food + estimated weight to macros. Already validated in POC.
- [ ] **Macro tracking dashboard** -- daily view of calories, protein, carbs, fat per meal and total.
- [ ] **Manual search and edit** -- fallback when AI is wrong. Fast logger UX (target: MacroFactor's 10-action benchmark).
- [ ] **Recipe saving** -- save corrected meals for future reuse. Already exists in backend.
- [ ] **Gallery scanning (manual trigger)** -- user taps "scan gallery" to process recent photos. Not yet passive/background.
- [ ] **Basic deduplication** -- temporal clustering (photos within 5 min at same location = same meal). Visual similarity deferred.

### Add After Validation (v1.x)

Features to add once the core pipeline proves accurate enough to trust.

- [ ] **Background/periodic gallery scanning** -- move from manual trigger to automatic. Trigger: users report manual scan works well.
- [ ] **Photo deduplication with visual similarity** -- embedding-based dedup to catch non-temporal duplicates. Trigger: users report duplicate entries from manual scanning.
- [ ] **End-of-day summary notification** -- configurable push notification with daily macro totals. Trigger: users have enough data flowing to make summaries useful.
- [ ] **Apple Health / Google Fit integration** -- sync weight and exercise data in. Trigger: core nutrition tracking is stable.
- [ ] **Scale/weight detection via OCR** -- detect kitchen scale in photos, read displayed weight. Trigger: users request better portion accuracy.
- [ ] **Confirm-only and zero-effort UX modes** -- enable configurable trust levels. Trigger: AI accuracy reaches point where auto-logging is trustworthy.
- [ ] **Hidden ingredient inference** -- recipe knowledge graph for inferring non-visible ingredients. Trigger: dish identification accuracy is high enough.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **3D depth/LiDAR portion estimation** -- significant engineering investment for iPhone Pro only. Defer until visual estimation limitations are quantified.
- [ ] **Correlation graphs** -- requires stable multi-stream data (nutrition + exercise + weight). Defer until Health integration is mature.
- [ ] **Container weight learning** -- useful but niche. Defer until scale detection proves valuable.
- [ ] **Multi-region database routing** -- USDA covers most needs for English-speaking users. Add AFCD, CoFID, etc. when user base demands it.
- [ ] **Barcode scanning** -- only if photo-first approach proves insufficient for packaged foods.
- [ ] **Micronutrient deep-dive views** -- store the data from day one, but don't build specialized UI until users request it.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Rationale |
|---------|------------|---------------------|----------|-----------|
| Food photo recognition (YOLO) | HIGH | MEDIUM | P1 | Core value prop. Nothing works without this. |
| Gallery scanning (manual) | HIGH | MEDIUM | P1 | Primary differentiator. Even manual trigger is novel. |
| Basic portion estimation | HIGH | MEDIUM | P1 | Without weight/portion, nutrition numbers are meaningless. |
| Nutrition database lookup (USDA) | HIGH | LOW | P1 | Already prototyped. Straightforward API integration. |
| Macro tracking dashboard | HIGH | LOW | P1 | Table stakes UI. Not complex to build. |
| Manual search & edit | HIGH | LOW | P1 | Essential fallback. |
| Recipe saving | MEDIUM | LOW | P1 | Already exists. High reuse value. |
| Basic deduplication (temporal) | HIGH | LOW | P1 | Structural requirement for gallery scanning. |
| EXIF metadata extraction | MEDIUM | LOW | P1 | Needed for dedup and meal context. Low effort. |
| Background gallery scanning | HIGH | MEDIUM | P2 | Passive = zero-effort. But manual trigger validates first. |
| Visual similarity dedup | MEDIUM | MEDIUM | P2 | Improves dedup accuracy but temporal clustering handles 80% of cases. |
| End-of-day notification | MEDIUM | LOW | P2 | Engagement driver. Simple to build after data pipeline exists. |
| Apple Health integration | MEDIUM | MEDIUM | P2 | Expected feature. Opens door to correlation graphs. |
| Scale/weight OCR detection | MEDIUM | MEDIUM | P2 | Accuracy multiplier for users who weigh food. |
| Configurable UX modes | MEDIUM | MEDIUM | P2 | Important for trust building but needs stable AI first. |
| Hidden ingredient inference | MEDIUM | MEDIUM | P2 | Accuracy improvement for mixed dishes. |
| Weight trend tracking | MEDIUM | LOW | P2 | Depends on Health integration for weigh-in data. |
| Correlation graphs | MEDIUM | HIGH | P3 | High value but needs mature multi-stream data. |
| 3D LiDAR estimation | MEDIUM | HIGH | P3 | iPhone Pro only. SnapCalorie validates approach. |
| Container weight learning | LOW | LOW | P3 | Niche feature dependent on scale detection. |
| Multi-region DB routing | LOW | MEDIUM | P3 | USDA covers most users initially. |

**Priority key:**
- P1: Must have for launch -- core pipeline
- P2: Should have, add once core is validated
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | MyFitnessPal | MacroFactor | Foodvisor | Lose It | SnapCalorie | Cal AI | Cronometer | **Our Approach** |
|---------|-------------|-------------|-----------|---------|-------------|--------|------------|-----------------|
| **Food photo recognition** | Premium (Meal Scan, 97% accuracy) | AI Describe (voice, not photo) | Core feature (fast, -47% energy estimation error) | Snap It (photo, needs manual verification) | Core feature (16% error rate) | Core feature (claims 90%, users report ~80%) | No | YOLO on-device, cloud fallback. Photo-first. |
| **Passive gallery scanning** | No | No | No | No | No | No | No | **Primary differentiator.** Background discovery of food photos. |
| **Portion estimation method** | Manual entry | Manual entry | Visual AI estimation | Visual AI estimation | LiDAR depth sensor + visual fallback | Visual AI + phone depth sensor | Manual entry | Visual estimation + scale OCR detection + LiDAR (future). |
| **Nutrition database** | 20M+ crowd-sourced (accuracy issues) | 26.5k NCC verified (research-grade) | European-focused | 50M+ foods | USDA verified | Unspecified | 1.1M verified, 84 micronutrients, lab-analyzed | USDA FDC primary (400k+ verified), multi-region expansion. |
| **Logging speed** | 15 actions per food | 10 actions per food (fastest) | Photo snap + review | Photo snap + manual verify | Photo snap, instant | Photo snap, instant | 10+ actions (manual) | Photo auto-detected from gallery. Target: 2-3 taps to confirm. |
| **Adaptive algorithm** | Static goals | Dynamic TDEE, weekly adjustment (best in class) | Dietitian coaching (premium) | Static goals | No | No | No | Not in scope. Provide data, not coaching. |
| **Barcode scanner** | Premium (was free pre-2022) | Yes | Yes | Yes | No | Yes | Yes | Not in v1. Photo-first approach. |
| **Health platform sync** | Apple Health, Fitbit, Garmin | Apple Health, Google Fit | Apple Health | Apple Health, Fitbit, Google Fit | Apple Health | Apple Health | Apple Health, Fitbit, Google Fit | Apple Health + Google Fit for exercise, weight, nutrition sync. |
| **Deduplication** | N/A (manual logging) | N/A (manual logging) | N/A (manual logging) | N/A (manual logging) | N/A (manual logging) | N/A (manual logging) | N/A (manual logging) | **Novel.** Temporal + visual clustering to prevent duplicate meal entries. |
| **Scale detection** | No | No | No | No | No | No | No | **Novel.** OCR on kitchen scale displays + container tare learning. |
| **Correlation analytics** | Basic (calories in vs out) | TDEE trends, expenditure modifiers | No | Basic weight trend | No | No | Nutrient target vs actual | Cross-domain: nutrition x exercise x weight over time. |
| **Price** | Free + Premium ($80/yr) | $72/yr (no free tier) | Free + Premium | Free + Premium | Free | $2.49/mo or $30/yr | Free + Gold ($50/yr) | TBD. Premium-only likely (no ad-supported free tier). |

### Competitive Positioning Summary

**Where competitors are strong (don't compete directly):**
- MyFitnessPal: massive database size and brand recognition
- MacroFactor: adaptive TDEE algorithm and coaching
- Cronometer: micronutrient depth and data accuracy
- SnapCalorie: 3D depth-based portion estimation research

**Where we differentiate:**
- Passive gallery scanning (nobody does this)
- Intelligent deduplication (novel problem for passive approach)
- Scale/weight detection via OCR (nobody does this)
- Zero-effort UX philosophy (competitors require opening the app)

**Key insight:** Every competitor requires the user to actively open the app and take an action to log food. Our passive approach inverts this: the app discovers food photos the user already took and presents them for review. This addresses the #1 reason people stop tracking (84% report tracking is tedious, 70% abandon within 2 weeks if too complex).

## Sources

### Competitor Products Analyzed
- [MyFitnessPal](https://www.myfitnesspal.com/) -- dominant market share, 220M+ users, massive crowd-sourced database
- [MacroFactor](https://macrofactor.com/) -- adaptive TDEE algorithm, fastest food logger, $72/yr
- [Foodvisor](https://www.foodvisor.io/en/) -- French-origin, AI photo recognition, European food coverage
- [Lose It](https://apps.apple.com/us/app/lose-it-calorie-counter/id297368629) -- 50M+ users, Snap It photo feature, AI voice logging
- [SnapCalorie](https://www.snapcalorie.com/) -- ex-Google AI team, LiDAR depth estimation, 16% caloric error rate
- [Cal AI](https://www.calai.app/) -- 1M+ downloads, phone depth sensor, $30/yr
- [Cronometer](https://cronometer.com/) -- 1.1M verified foods, 84 micronutrients, lab-analyzed sources
- [Fitia](https://fitia.app/) -- nutrition professional-verified entries, AI coach
- [Passio AI](https://www.passio.ai/) -- B2B SDK for food recognition, on-device + cloud, 2.5M+ foods

### Research & Technical Sources
- [Automated Food Weight Estimation using CV and AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644939/) -- YOLO-based weight estimation, 5% error margin (MEDIUM confidence)
- [Food Portion Estimation via 3D Object Scaling (CVPR 2024)](https://arxiv.org/abs/2404.12257) -- reference object scaling for portion estimation
- [SnapCalorie Nutrition5k Research](https://www.snapcalorie.com/blog/snapcalorie-revolutionizing-nutrition-tracking-with-ai/) -- peer-reviewed portion estimation validation
- [YOLO26 Release](https://blog.roboflow.com/yolo26/) -- January 2026, CoreML/TFLite export, 43% faster CPU inference than YOLO11
- [Evaluating AI Food Image Recognition (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11314244/) -- MFP 97% accuracy, Foodvisor -47% energy estimation error
- [Apple HealthKit Documentation](https://developer.apple.com/documentation/healthkit) -- 70+ nutrition data types
- [Apple Health 2026 Redesign](https://9to5mac.com/2026/01/11/apple-health-new-features-and-overhaul-coming-ios-26-4/) -- AI agent, meal tracking, Health+ service incoming
- [Diet App Retention Statistics](https://media.market.us/diet-and-nutrition-apps-statistics/) -- 84% find tracking tedious, 70% abandon in 2 weeks
- [Nutrition API Comparison](https://www.spikeapi.com/blog/top-nutrition-apis-for-developers-2026/) -- USDA FDC vs Nutritionix vs OpenFoodFacts
- [iOS Photo Library Privacy](https://developer.apple.com/documentation/PhotoKit/delivering-an-enhanced-privacy-experience-in-your-photos-app) -- PHPicker, limited library access, privacy constraints

---
*Feature research for: AI-powered food/nutrition tracking mobile apps*
*Researched: 2026-02-12*
