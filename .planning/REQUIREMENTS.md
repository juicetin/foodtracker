# Requirements: FoodTracker

**Defined:** 2026-02-12
**Core Value:** Accurate, effortless food tracking from photos you already take — no manual entry, no barcode scanning, just eat, photograph, and review.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### On-Device ML Detection

- [ ] **DET-01**: App classifies gallery photos as food vs non-food using on-device binary classifier
- [ ] **DET-02**: App identifies specific food items in photos using fine-tuned YOLO model with bounding boxes
- [ ] **DET-03**: App estimates portion size from visual cues (plate/bowl size, reference objects)
- [ ] **DET-04**: App detects kitchen scales in photos and reads displayed weight via OCR
- [ ] **DET-05**: App identifies the dish type and infers hidden ingredients from recipe knowledge (e.g. "alfredo" → butter, cream, parmesan)
- [ ] **DET-06**: App uses LiDAR depth sensor on supported devices (iPhone Pro) for more accurate portion estimation, with graceful fallback to visual estimation

### Gallery Scanning & Deduplication

- [ ] **GAL-01**: User can manually trigger a gallery scan to process recent food photos
- [ ] **GAL-02**: App performs background/periodic gallery scanning to discover food photos automatically
- [ ] **GAL-03**: App extracts EXIF metadata (timestamp, GPS location, camera model, focal length) from each photo
- [ ] **GAL-04**: App deduplicates food photos using temporal clustering (photos within 5 minutes at same location = same meal)
- [ ] **GAL-05**: App deduplicates food photos using visual similarity embeddings to catch non-temporal duplicates (different angles, group vs individual shots)
- [ ] **GAL-06**: App prompts user for confirmation when deduplication confidence is low

### Nutrition & Database

- [ ] **NUT-01**: App looks up per-ingredient nutritional data from USDA FoodData Central
- [ ] **NUT-02**: App calculates per-ingredient macro breakdown (calories, protein, carbs, fat)
- [ ] **NUT-03**: App calculates per-ingredient micro nutrient data (stored, not prominently displayed)
- [ ] **NUT-04**: App combines detected ingredients + estimated weights to produce total meal nutrition estimate

### Tracking UI

- [ ] **UI-01**: User can view daily macro tracking dashboard (calories, protein, carbs, fat per meal and total)
- [ ] **UI-02**: User can manually search and add foods when AI detection fails (fast logger — target <10 taps)
- [ ] **UI-03**: User can edit/correct detected ingredients, portions, and quantities after logging
- [ ] **UI-04**: User can save corrected meals as recipes for future reuse
- [ ] **UI-05**: User can reuse saved recipes to log repeat meals
- [ ] **UI-06**: User can view daily and weekly nutrition summaries
- [ ] **UI-07**: Logged meals retain the original photo(s) linked for later review
- [ ] **UI-08**: User can adjust serving size/scale of a logged meal proportionally

### Notifications & UX Modes

- [ ] **UX-01**: User receives configurable end-of-day summary notification (total cals, protein, carbs, fat)
- [ ] **UX-02**: User can switch between UX modes: zero-effort (auto-log, daily summary), confirm-only (review before logging), guided-edit (step-by-step correction)
- [ ] **UX-03**: App defaults to zero-effort mode with edits available after the fact

### Scale & Weight Intelligence

- [ ] **SCL-01**: User can save known container/vessel weights for tare subtraction
- [ ] **SCL-02**: App improves scale reading accuracy over time by learning frequently used container weights
- [ ] **SCL-03**: App intelligently guesses whether a scale reading includes the container or not

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Health Platform Integration

- **HLT-01**: App syncs with Apple Health (HealthKit) to read weight and exercise data
- **HLT-02**: App syncs with Google Fit (Health Connect) to read weight and exercise data
- **HLT-03**: App writes nutrition data to Apple Health / Google Fit
- **HLT-04**: User can view weight trend tracking with smoothing (MacroFactor-style)

### Analytics & Insights

- **ANL-01**: User can view correlation graphs (nutrition vs exercise vs weight over time)
- **ANL-02**: App estimates TDEE based on tracked nutrition and weight changes

### Database Expansion

- **DB-01**: App routes nutrition lookups to region-specific databases (AFCD for AU, CoFID for UK, CIQUAL for FR)
- **DB-02**: App supports branded/packaged food database lookups

### Container Intelligence

- **CTN-01**: App recognizes specific containers across photos and auto-applies tare weight

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| AI coaching / auto-adjust programs | User wants to track, not be told what to eat. MacroFactor does this well already. |
| Social features / sharing | Personal tracking tool, not a social network. Massive engineering investment for no target user value. |
| Meal planning / recommendations | Track what you eat, not what you should eat. Different product entirely. |
| Barcode scanning | Photo-first is the differentiator. Barcode scanning pulls UX toward packaged food logging (MFP's strength). Defer to v2+. |
| Gamification (streaks, badges) | Creates anxiety-driven engagement. Health tracking should reduce stress. Target user is intrinsically motivated. |
| Real-time camera food detection | Users review photos taken earlier, not pointing cameras at food in real time. Battery drain from continuous camera + ML. |
| Micronutrient deep-dive UI | Store micronutrient data but don't build specialized UI. Overwhelming for macro-focused fitness enthusiast. |
| Web app | Mobile-first, on-device ML makes web impractical. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DET-01 | — | Pending |
| DET-02 | — | Pending |
| DET-03 | — | Pending |
| DET-04 | — | Pending |
| DET-05 | — | Pending |
| DET-06 | — | Pending |
| GAL-01 | — | Pending |
| GAL-02 | — | Pending |
| GAL-03 | — | Pending |
| GAL-04 | — | Pending |
| GAL-05 | — | Pending |
| GAL-06 | — | Pending |
| NUT-01 | — | Pending |
| NUT-02 | — | Pending |
| NUT-03 | — | Pending |
| NUT-04 | — | Pending |
| UI-01 | — | Pending |
| UI-02 | — | Pending |
| UI-03 | — | Pending |
| UI-04 | — | Pending |
| UI-05 | — | Pending |
| UI-06 | — | Pending |
| UI-07 | — | Pending |
| UI-08 | — | Pending |
| UX-01 | — | Pending |
| UX-02 | — | Pending |
| UX-03 | — | Pending |
| SCL-01 | — | Pending |
| SCL-02 | — | Pending |
| SCL-03 | — | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 0
- Unmapped: 30

---
*Requirements defined: 2026-02-12*
*Last updated: 2026-02-12 after initial definition*
