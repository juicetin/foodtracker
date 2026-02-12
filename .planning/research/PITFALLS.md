# Pitfalls Research

**Domain:** AI-powered food tracking with on-device ML, gallery scanning, nutrition estimation
**Researched:** 2026-02-12
**Confidence:** MEDIUM-HIGH (based on academic papers, competitor failures, developer forums, Apple docs)

## Critical Pitfalls

### Pitfall 1: Training Data Bias Destroys Non-Western Food Accuracy

**What goes wrong:**
YOLO models fine-tuned on standard datasets (Food-101, ISIA Food-500) perform well on Western foods but fail dramatically on Asian, Indian, Middle Eastern, and African cuisines. Asian dishes make up only ~30% of the ETHZ Food-101 dataset; African dishes ~1%. Energy is overestimated for Western diets by ~249 calories on average but underestimated for Asian diets by ~363 calories. For an Australian user eating pho, the app could be off by 49% on calories. For pearl milk tea, underestimation can reach 76%.

**Why it happens:**
Teams fine-tune on available English-language datasets without auditing cuisine distribution. They test against the same biased test split and report good aggregate mAP numbers that mask per-cuisine failures. Dishes of the same category present completely different visual features due to regional cooking differences (e.g., "curry" in Japan vs India vs Thailand).

**How to avoid:**
- Audit training data by cuisine region before training. Compute per-cuisine mAP, not just aggregate.
- Supplement Food-101 with region-specific datasets: IndianFood dataset, VIREO Food-172 (Chinese), UEC-FOOD-256 (Japanese). Build a custom annotation set for Australian-specific foods (meat pies, flat whites, Vegemite toast).
- Use stratified evaluation: create a test set with equal representation across target cuisines and report per-region accuracy separately.
- Implement a confidence threshold + cloud fallback: when on-device YOLO confidence is below threshold (e.g., 0.6), route to cloud model with broader training data.

**Warning signs:**
- Aggregate mAP looks great (>80%) but you haven't tested on non-Western food photos
- User complaints cluster around specific food types or ethnic cuisines
- Your training dataset has fewer than 50 images per class for target regional foods

**Phase to address:**
Model training phase. This must be addressed before any user-facing accuracy claims. Build the stratified evaluation framework before fine-tuning begins.

---

### Pitfall 2: Portion/Weight Estimation Errors Compound Into Unusable Calorie Numbers

**What goes wrong:**
Even with perfect food identification, weight estimation from 2D images introduces 10-40% error. A 2D image lacks depth information, so a flat chicken breast and a thick one look identical. Volume estimation errors range from 0.09% to 33%, and calorie estimate errors span 0.10% to 38.3%. On a 2,000-calorie diet, this means 200-600 calories/day of error -- enough to completely undermine weight loss/gain goals. Real user complaints confirm: "60kcal estimate for 260kcal worth of grapes" and "meat estimated about 50% less calories than when weighed."

**Why it happens:**
Teams treat weight estimation as a secondary problem to food identification. The 2D-to-3D volume estimation problem is fundamentally harder than classification. No dimensional reference exists in a typical food photo. Camera angle, plate size, and food density all vary. Bounding box area is a poor proxy for volume -- two foods with identical bounding boxes can differ 3x in weight (a pile of lettuce vs. a pile of rice).

**How to avoid:**
- Do NOT rely solely on bounding-box-area-to-weight regression. This is the most common shortcut and it produces terrible results.
- Implement a reference object system: detect plate/bowl edges and use known plate diameters as reference. Learn common container sizes from user history.
- Prioritize scale integration for users who want accuracy. Bluetooth scale readings are ground truth; image estimation is always a fallback.
- Use food-specific density tables, not a single density constant. Rice, salad, and steak have vastly different weight-to-volume ratios.
- Present calorie estimates as ranges (e.g., "350-450 kcal") rather than false-precision single numbers. Users who see "387 kcal" trust it as exact.
- Allow easy manual weight override and learn from corrections over time.

**Warning signs:**
- Your weight estimation uses a single linear model from bbox area to grams
- You test with food on white plates at consistent angles but users photograph food on varied surfaces
- Calorie estimates consistently under-count (the most dangerous failure mode -- users eat more than tracked)

**Phase to address:**
Weight estimation phase, with scale integration providing ground-truth validation. Do not ship weight estimation without a manual correction UI -- this is table stakes.

---

### Pitfall 3: iOS Photo Library "Limited Access" Breaks Passive Gallery Scanning

**What goes wrong:**
Since iOS 14, users can grant "limited" photo access -- selecting specific photos the app can see rather than the full library. Your passive gallery scanner sees zero new photos unless the user explicitly adds them via the system picker. The app thinks there are no new food photos to analyze, and the entire passive scanning value proposition silently fails. Worse, the deprecated pre-iOS 14 authorization check returns `.authorized` instead of `.limited`, so code that checks the old way thinks it has full access when it doesn't.

**Why it happens:**
Teams build gallery scanning assuming full photo library access, test on their own devices with full access granted, and never exercise the limited access path. Apple's privacy model fundamentally opposes passive background scanning of user photos. PHPickerViewController runs out-of-process -- the app literally cannot see or screenshot the picker content.

**How to avoid:**
- Design for limited access as the primary path, not full access. Assume most privacy-conscious users will grant limited access.
- Use `PHPhotoLibrary.requestAuthorization(for: .readWrite)` with the access-level parameter (not the deprecated no-argument version).
- For passive scanning: monitor `PHPhotoLibraryChangeObserver` for changes to the limited selection. When the user grants access to new photos, scan those.
- Implement a manual "scan these photos" flow as the primary UX, with passive scanning as an enhancement for users who grant full access.
- Add `PHPhotoLibraryPreventAutomaticLimitedAccessAlert` to Info.plist and manage the limited library picker UI yourself at the right moment (not on app launch).
- In Expo/React Native: `expo-media-library` has known issues with limited access. Test with `requestMediaLibraryPermissionsAsync()` and verify the `accessPrivileges` field returns `"limited"` vs `"all"`.

**Warning signs:**
- Your gallery scanning tests only run with full photo library access
- You use the deprecated `PHPhotoLibrary.authorizationStatus()` without the `for:` parameter
- Users report "the app never finds my food photos" but your logs show no errors

**Phase to address:**
Gallery scanning phase. This must be the first thing you prototype -- before building the ML pipeline on top of it. If passive scanning is architecturally impossible for your permission model, you need to know early and redesign around manual photo selection.

---

### Pitfall 4: Hidden Ingredients Make Image-Only Calorie Tracking Fundamentally Inaccurate

**What goes wrong:**
Cooking oils, butter, sauces, marinades, sugar in dressings, and preparation methods are invisible in photos but account for 200-500+ calories per meal. A grilled chicken breast and a pan-fried-in-butter chicken breast look identical. A salad with 2 tbsp ranch dressing adds 150 calories that no vision model can detect. Users who rely on photo-only tracking systematically under-count calories by 15-30%.

**Why it happens:**
Teams focus on the impressive demo -- "point your camera at food and get instant calories!" -- without acknowledging the fundamental physical limitation that a 2D image cannot capture ingredients that aren't visible. Marketing promises outpace technical capability.

**How to avoid:**
- Always prompt users to add cooking method and condiments after AI detection. This is not optional friction -- it is necessary for accuracy.
- Build a "common additions" quick-add UI: after detecting "chicken breast," suggest "add cooking oil?", "add sauce?" with common calorie values.
- For known recipes/meals, let users save the full recipe (including hidden ingredients) and reuse it. Recurring meals are the highest-accuracy path.
- Be transparent about limitations: "AI estimated visible food. Tap to add cooking oils, sauces, or dressings for better accuracy."
- Track a per-user "hidden ingredient correction factor" learned from users who do enter detailed data, and apply it as a suggestion for similar meals.

**Warning signs:**
- Your accuracy testing uses pre-portioned foods without oils/sauces
- User testing shows systematic calorie under-counting vs. weighed/measured ground truth
- Users report losing less weight than expected (the most common real-world signal)

**Phase to address:**
Review/editing UX phase. The AI detection pipeline cannot solve this -- the solution is UX that makes adding invisible ingredients fast and frictionless.

---

### Pitfall 5: CoreML/TFLite Model Deployment Fails Silently on Real Devices

**What goes wrong:**
A YOLO model that performs well in Python produces incorrect bounding boxes, crashes, or runs at 2 FPS after CoreML/TFLite conversion and deployment. Half-precision floating point on iPhone causes accuracy degradation. Specific layer combinations trigger inference engine bugs (e.g., dilated convolutions consuming excessive memory). After macOS/iOS version updates, previously working CoreML exports produce incorrect outputs. iPhone has only 6GB total RAM (iPhone 14 Pro), with ~2-3GB available to your app -- exceeding this causes immediate termination with no crash report.

**Why it happens:**
Teams validate model accuracy in Python (float32, desktop GPU) and assume the converted mobile model performs identically. CoreML's inference engine cannot be tested against every layer combination, so bugs lurk in specific architectures. The conversion process involves quantization and layer translation that can introduce errors invisible in aggregate metrics.

**How to avoid:**
- Convert to CoreML/TFLite early in development -- before full training. Run a quick sanity check that the conversion pipeline works and produces reasonable outputs on-device.
- Validate on-device inference outputs against Python outputs for the same input images. Compare bounding box coordinates numerically, not just visually.
- Test on the oldest supported device (e.g., iPhone 12) not just the latest hardware. Neural Engine capabilities vary across generations.
- Use YOLOv8n (nano) or YOLOv11n for mobile -- not the medium or large variants. Model size directly impacts memory and battery.
- Pin your Ultralytics version and CoreML tools version. A minor version bump can change export behavior.
- In React Native: use `react-native-fast-tflite` for TFLite or build a custom Expo native module for CoreML. The CoreML delegate for TFLite has operation support limitations -- verify your model's operations are supported before committing to an approach.

**Warning signs:**
- You've only tested the model in a Jupyter notebook, never on a physical device
- Inference time exceeds 100ms on target hardware (should be ~30ms for YOLO nano)
- Output bounding boxes are slightly but consistently offset from Python outputs
- App memory usage exceeds 1.5GB during inference

**Phase to address:**
On-device ML integration phase. Do a device deployment spike BEFORE investing in extensive fine-tuning. A model that can't deploy is worthless regardless of its accuracy.

---

### Pitfall 6: Food-to-Nutrition-DB Mapping Is a Harder Problem Than Food Detection

**What goes wrong:**
YOLO detects "rice" but the USDA FoodData Central has 200+ rice entries: white rice cooked, brown rice raw, fried rice, rice pilaf, jasmine rice, etc. The mapping from a detected class label to the correct nutrition DB entry introduces another 10-30% calorie error on top of detection and weight errors. Fuzzy text matching returns incorrect entries (e.g., "chicken curry" matching "curry powder" instead of "chicken curry, restaurant-prepared"). Regional database gaps mean Australian foods like meat pies have no USDA equivalent.

**Why it happens:**
Teams treat nutrition lookup as a simple API call: detect food -> search USDA -> done. But the semantic gap between a visual class label ("pasta") and a nutrition DB entry ("spaghetti, cooked, enriched" vs "fettuccine, cooked" vs "pasta, fresh-refrigerated, plain, cooked") is significant. Different DB entries for the same visual food can vary by 50-100% in calories depending on preparation method and serving state.

**How to avoid:**
- Build a curated mapping table from your YOLO class labels to specific nutrition DB entries (not just a text search). Each YOLO class should map to a default USDA FDC ID with the most common preparation method.
- Use the USDA "Foundation Foods" and "SR Legacy" datasets for per-100g values -- these are research-grade. Avoid "Branded Foods" for default mappings (too much variability).
- Implement a preparation-method modifier system: base food + cooking method = specific nutrition profile. "Chicken breast" + "grilled" = different entry than "chicken breast" + "fried."
- For multi-region support: build an abstraction layer early. The `NutritionPer100g` dataclass pattern from your existing POC is the right approach -- keep it source-agnostic.
- Cache and version your mapping table. USDA updates quarterly; don't let a DB update silently change calorie values for existing food logs.

**Warning signs:**
- Your nutrition lookup is a raw text search against the USDA API with no curated mappings
- The same detected food returns different calorie values on different API calls (due to fuzzy search instability)
- You haven't tested your mapping against regional foods from target markets

**Phase to address:**
Nutrition DB integration phase. This is a data curation problem, not a code problem. Budget significant time for building and validating the mapping table.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single-number calorie display (no ranges) | Cleaner UI, simpler code | Users trust false precision, lose faith when estimates are wrong | Never -- always show ranges or confidence indicators |
| Hardcoded YOLO class-to-nutrition mapping | Fast to build | Every new food class requires a code change, no user customization | MVP only, replace with DB-driven mapping in next phase |
| Skip manual weight override UI | Faster MVP launch | Power users (your most engaged segment) have no recourse for bad estimates | Never -- this is table stakes |
| Full photo library access only | Simpler permissions code | App rejected or rated poorly by privacy-conscious users on iOS 14+ | Never -- limited access is the norm now |
| Single nutrition DB (USDA only) | Simpler integration | Poor accuracy for non-US foods, blocks international expansion | POC only, per existing ADR-004 decision |
| Skip background task testing on real devices | Faster dev cycles | BGProcessingTask behaves completely differently in debug vs production | Never -- iOS background tasks require on-device testing |
| Bundle model in app binary | No download management code | App binary exceeds 100MB, slow App Store downloads, can't update model without app release | MVP only if model is <30MB, otherwise use on-demand download |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| USDA FoodData Central API | Using raw text search and taking first result | Build curated mapping table with specific FDC IDs per food class; use Foundation/SR Legacy datasets, not Branded Foods |
| Apple HealthKit | Writing nutrition data without clear HealthKit UI in the app | Apple rejects apps that use HealthKit APIs but don't clearly surface HealthKit functionality in the UI. Must show health data prominently, include privacy policy, never store health data in iCloud |
| HealthKit (data integrity) | Writing AI-estimated calories as exact values | Apple guidelines prohibit writing inaccurate data to HealthKit. Mark AI-estimated entries with metadata indicating estimation method and confidence level |
| iOS Photo Library (Expo) | Using `expo-image-picker` for gallery scanning | `expo-image-picker` is for one-off selection. Use `expo-media-library` with `getAssetsAsync()` for scanning, and handle the `accessPrivileges: "limited"` case explicitly |
| Bluetooth smart scales | Assuming stable BLE connection during weighing | BLE connections drop. Buffer weight readings, use the last stable reading, handle reconnection gracefully. Android requires location permission for BLE scanning; iOS does not |
| CoreML model updates | Bundling updated model in app binary for each update | Host models externally, download on-demand, compile on-device with `MLModel.compileModel(at:)`. Version with integer (not UUID) so client can compare versions |
| BGProcessingTask | Testing only in Xcode debugger | `BGProcessingTask` timing is entirely system-controlled in production. Set `requiresExternalPower = false` but test on battery. System may defer task for hours or days based on battery and usage patterns |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Running YOLO inference on every gallery photo without throttling | Battery drain >5%/hour in background, thermal throttling, user complaints | Process max 10-20 photos per BGProcessingTask execution. Prioritize by recency. Skip already-processed photos via content hash | Immediate -- users notice battery drain within first day |
| Loading full-resolution images for YOLO inference | 12MP iPhone photos consume ~48MB each in memory. 10 photos = OOM kill | Resize to model input size (640x640) before inference. Use `CGImageSourceCreateThumbnailAtIndex` for efficient downscaling | At 3-5 photos queued. App terminates with no crash log |
| Synchronous nutrition DB lookups per food item | UI freezes for 1-3 seconds per detected food during review | Batch nutrition lookups, cache aggressively (USDA data changes quarterly), pre-fetch common foods on app install | At 3+ food items per meal. Users perceive app as slow |
| Storing full-resolution images with food logs | DB/storage grows 5-10MB per meal logged | Store only thumbnails + references to photo library assets. Never copy full images into app sandbox | At ~200 logged meals (~1-2GB). Users delete app for storage |
| Per-photo cloud fallback without rate limiting | Cloud API costs spike with heavy users (5+ meals/day with multiple photos) | Implement daily cloud inference budget per user. Queue low-confidence items for batch processing | At 100+ active users. $0.01-0.03 per cloud inference adds up fast |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Sending food photos to cloud inference without user consent | Privacy violation, potential GDPR/privacy law breach, App Store rejection | Explicit opt-in for cloud processing. Default to on-device only. Never send photos to third-party APIs without disclosure |
| Storing nutrition/health data unencrypted on device | Health data exposure if device is compromised | Use iOS Keychain for sensitive data, encrypted Core Data for food logs. HealthKit data is already protected by Apple |
| Logging food photo metadata (GPS, timestamps) to analytics | Reveals user location, meal times, dietary patterns to analytics provider | Strip EXIF data before any cloud upload. Anonymize timestamps in analytics. Never send photo metadata to third-party analytics |
| Using USDA DEMO_KEY in production | 1000 req/hour rate limit, no SLA, shared across all DEMO_KEY users | Register for a dedicated API key. Implement server-side caching to reduce API calls. Rate limit is per-IP, so a shared backend IP exhausts it fast |
| Trusting client-side model confidence scores without validation | Adversarial inputs or model bugs could write garbage data to HealthKit | Server-side validation of nutrition data before HealthKit write. Sanity-check ranges (e.g., single meal >3000 kcal triggers review) |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Requiring photo crop before analysis | Adds friction to every meal log, users abandon the flow | Auto-detect food regions. If multiple foods detected, show detection overlay and let user confirm/adjust. Never require manual cropping |
| Showing single calorie number with false precision ("387 kcal") | Users trust the number as exact, lose faith when weight goals don't match | Show ranges ("350-420 kcal") or confidence indicator. Be transparent: "AI estimate -- tap to refine" |
| Making correction flow harder than initial entry | Users who find errors stop correcting, data quality degrades, trust erodes | One-tap to edit any detected food. Inline weight/portion adjustment. "This isn't right" shortcut that opens full edit mode |
| End-of-day notification without actionable content | Notification fatigue, users disable notifications, lose daily habit | Include summary: "3 meals logged, ~1,850 kcal. Missing anything?" with quick-add buttons for common snacks/drinks |
| Auto-logging detected food without user review | Wrong detections appear in food log without user noticing. Calorie totals become untrustworthy | Always show detection results for user confirmation before logging. Passive scanning should create "pending review" items, never auto-confirmed entries |
| Complex multi-step flow to add a simple snack | Users revert to manual apps (MyFitnessPal) for quick entries, use AI tracking only sometimes | Provide text entry, barcode scan, and recent/frequent foods alongside photo capture. The fastest path should be 2 taps for a repeated meal |

## "Looks Done But Isn't" Checklist

- [ ] **Food detection:** Tested on non-Western cuisines (Asian, Indian, Middle Eastern) with per-cuisine accuracy metrics -- not just aggregate mAP
- [ ] **Gallery scanning:** Tested with iOS limited photo access (not just full access). Verified behavior when user has granted access to zero food-related photos
- [ ] **Weight estimation:** Tested with foods off-plate (handheld items, food in bags, restaurant takeout containers) -- not just foods centered on white plates
- [ ] **Nutrition lookup:** Verified that the same detected food always maps to the same nutrition entry (deterministic mapping, not fuzzy search)
- [ ] **Deduplication:** Tested with burst photos, screenshots of food, food-adjacent photos (restaurant menus, grocery receipts) that should NOT be logged
- [ ] **HealthKit integration:** HealthKit permission UI is visible in app without navigating to settings. Privacy policy covers health data usage. App passes the "HealthKit feature must be clearly identified" App Store review requirement
- [ ] **Background processing:** Tested BGProcessingTask on a real device unplugged from power, with the app not in foreground, over a 24-hour period. Verified tasks actually execute
- [ ] **Model deployment:** Compared on-device inference outputs numerically against Python reference outputs for the same test images. Checked on oldest supported iOS version
- [ ] **Cloud fallback:** Tested behavior when device is offline (graceful degradation, not crash). Cloud timeout handling works (doesn't block UI)
- [ ] **Correction flow:** User can correct every AI output: food type, weight, portion count, cooking method, condiments. Corrections are persisted and improve future suggestions

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Biased training data | MEDIUM | Collect region-specific training images (can crowdsource from users with consent), retrain model, ship OTA model update. Does not require app binary update if OTA model delivery is in place |
| Bad weight estimation model | MEDIUM | Add manual weight override UI (quick fix), then improve estimation model. User corrections become training data for improved model |
| Limited photo access not handled | HIGH | Requires architectural change from passive scanning to user-initiated scanning. May need to redesign core UX flow. Earlier detection = lower cost |
| Nutrition DB mapping errors | LOW | Fix mapping table (data change, no code change if mapping is DB-driven). Retroactively correct affected food logs with user notification |
| CoreML conversion failures | HIGH | May require changing model architecture to avoid unsupported layers. Worst case: switch from CoreML to TFLite or vice versa. Test conversion early to avoid this |
| HealthKit App Store rejection | MEDIUM | Add required HealthKit UI elements and privacy policy. Resubmit. Typically 1-2 day turnaround but blocks launch |
| Battery drain from background processing | LOW | Reduce photos-per-batch, add battery level check before processing, set `requiresExternalPower = true` as temporary fix. OTA config change if you have remote config |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Training data bias | Model Training | Per-cuisine mAP report showing >70% accuracy across all target cuisines |
| Weight estimation errors | Weight/Scale Integration | Mean absolute error <20% on a test set of 100 weighed meals. User correction rate <30% |
| iOS limited photo access | Gallery Scanning | Automated test: grant limited access to 5 photos, verify scanner finds exactly 5 and no more |
| Hidden ingredient blindness | Review/Edit UX | User study: participants add cooking method and condiments for >80% of meals within 3 taps |
| CoreML/TFLite deployment | On-Device ML Integration | Numerical comparison: on-device outputs within 5% of Python reference for 95% of test images |
| Food-to-nutrition mapping | Nutrition DB Integration | Deterministic test: same food class always returns same FDC ID. Manual audit of top 100 food mappings |
| HealthKit rejection | Health Platform Integration | Pre-submission App Store review checklist pass. HealthKit UI visible in first 3 screens of app |
| Photo deduplication failures | Gallery Scanning | Test set: 50 burst photos of same meal deduplicate to 1. 50 non-food photos (menus, receipts) are filtered out |
| Battery drain from scanning | Gallery Scanning | 24-hour battery test: background scanning consumes <2% battery with 20 photos/day |
| Cloud fallback cost explosion | On-Device ML Integration | Cost projection spreadsheet: cloud inference cost at 1K, 10K, 100K users with daily cap per user |

## Sources

- [How Accurate Are AI Calorie Counters? -- WhatTheFood](https://whatthefood.io/blog/how-accurate-are-ai-calorie-counters) -- accuracy ranges, dataset bias statistics
- [AI food tracking apps need improvement -- University of Sydney](https://www.sydney.edu.au/news-opinion/news/2024/08/29/ai-food-tracking-apps-need-improvement-to-address-cultural-diversity.html) -- cultural food bias, Asian diet underestimation
- [Common Mistakes in AI Food Tracking -- CounterCal](https://www.countercal.com/blog/common-mistakes-in-ai-food-tracking-and-how-to-avoid-them) -- hidden ingredients, portion estimation
- [Hacker News: AI calorie counting apps](https://news.ycombinator.com/item?id=44220135) -- real user complaints, accuracy failures, UX problems
- [Avoiding Hidden Hazards: ML on iOS -- Ksemianov](https://ksemianov.github.io/articles/ios-ml/) -- CoreML memory limits, inference bugs, half-precision issues
- [Delivering Enhanced Privacy in Photos App -- Apple Developer](https://developer.apple.com/documentation/PhotoKit/delivering-an-enhanced-privacy-experience-in-your-photos-app) -- limited access architecture
- [Managing Photo Library Permission -- Swift Senpai](https://swiftsenpai.com/development/photo-library-permission/) -- iOS permission handling pitfalls
- [AI-based dietary assessment systematic review -- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10836267/) -- Western vs Asian diet estimation bias
- [Automated Food Weight Estimation -- MDPI Sensors](https://www.mdpi.com/1424-8220/24/23/7660) -- volume estimation challenges
- [Vision-Based Food Weight Estimation -- arXiv](https://arxiv.org/html/2405.16478v1) -- 2D-to-3D estimation limitations
- [YOLO Food Detection -- Benny Cheung](https://bennycheung.github.io/yolo-for-real-time-food-detection) -- Food-101 training, class imbalance
- [Image-Based Dietary Assessment Using YOLO -- JMIR 2025](https://formative.jmir.org/2025/1/e70124) -- YOLO portion recognition limitations
- [iOS Background Tasks -- OneUpTime 2026](https://oneuptime.com/blog/post/2026-02-02-ios-background-tasks/view) -- BGProcessingTask constraints
- [Safely Distribute ML Models OTA -- ContextSDK](https://contextsdk.com/blog/safely-distribute-new-machine-learning-models-to-millions-of-iphones-over-the-air) -- model versioning pitfalls
- [Algorithm-based food DB mapping -- Frontiers in Nutrition](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2022.1013516/full) -- fuzzy matching challenges
- [Evaluating AI Food Image Recognition -- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11314244/) -- consumer app accuracy comparison
- [Deep Learning Food Image Recognition Review -- MDPI 2025](https://www.mdpi.com/2076-3417/15/14/7626) -- dataset bias across cuisines
- [react-native-fast-tflite -- GitHub](https://github.com/mrousavy/react-native-fast-tflite) -- React Native TFLite integration
- [Building React Native CoreML with Expo and YOLOv8 -- Medium](https://hietalajulius.medium.com/building-a-react-native-coreml-image-classification-app-with-expo-and-yolov8-a083c7866e85) -- Expo native module approach
- [Perceptual hashing for duplicate detection -- Ben Hoyt](https://benhoyt.com/writings/duplicate-image-detection/) -- hashing threshold selection
- [Effective near-duplicate detection -- ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S0306457325000287) -- pHash limitations, deep learning hybrid

---
*Pitfalls research for: AI-powered food tracking with on-device ML*
*Researched: 2026-02-12*
