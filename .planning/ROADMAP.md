# Roadmap: FoodTracker

## Overview

FoodTracker delivers accurate, effortless food tracking from photos users already take. The roadmap follows a risk-retirement strategy: validate food detection accuracy first (the foundation everything depends on, with a go/no-go decision on YOLO vs LLM as primary method), then prove the passive gallery scanning differentiator works within iOS permission constraints, stand up the multi-region nutrition backend, build the tracking UI, layer on UX intelligence, and finally add scale/weight accuracy enhancers. Each phase delivers a coherent, verifiable capability that unlocks the next.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Food Detection Foundation** - Validate detection accuracy and determine primary detection method (YOLO vs LLM)
- [ ] **Phase 2: Gallery Scanning & Deduplication** - Discover and deduplicate food photos from the user's gallery
- [ ] **Phase 3: Go Backend & Multi-Region Nutrition** - Serve nutrition data from global food composition databases and store food entries
- [ ] **Phase 4: Tracking UI & Entry Management** - Let users view, edit, and manage their food diary
- [ ] **Phase 5: UX Modes & Notifications** - Adapt the tracking experience to user preferences
- [ ] **Phase 6: Scale Detection & Weight Refinement** - Improve portion accuracy with scale OCR and depth sensing

## Phase Details

### Phase 1: Food Detection Foundation
**Goal**: Users can point the app at any food photo and get accurate identification of food items, dish type, and estimated portions -- with YOLO fine-tuning as the preferred on-device approach, but multimodal LLM calls as the primary method if YOLO accuracy proves insufficient. Accuracy is the top priority, above cost, speed, and on-device constraints.
**Depends on**: Nothing (first phase)
**Requirements**: DET-01, DET-02, DET-03, DET-05, DET-07
**Success Criteria** (what must be TRUE):
  1. App correctly classifies a photo as food vs non-food with >90% accuracy on a diverse test set
  2. App draws bounding boxes around individual food items in a photo and labels them with dish/ingredient names, achieving >80% mAP on a held-out test set spanning Western, Asian, and other regional cuisines
  3. App estimates portion size from visual cues (plate size, reference objects) and displays a weight estimate
  4. App infers hidden ingredients from dish identification (e.g. "carbonara" yields egg, pancetta, parmesan, black pepper)
  5. A go/no-go decision has been made on detection approach: YOLO fine-tuning benchmarks are compared against multimodal LLM accuracy on the same test set, and one method is selected as the primary detection path for v1 based on accuracy results
**Decision Point**: After initial YOLO fine-tuning and accuracy benchmarking, compare against multimodal LLM (Gemini/GPT-4o) on the same test images. If YOLO achieves target accuracy (>80% mAP), proceed with on-device YOLO. If not, pivot to LLM as the primary detection method and accept the cost/latency/network trade-offs. This decision gates Phase 2 architecture (on-device vs cloud detection pipeline).
**Plans**: TBD

Plans:
- [ ] 01-01: TBD
- [ ] 01-02: TBD

### Phase 2: Gallery Scanning & Deduplication
**Goal**: The app passively discovers food photos from the user's gallery, extracts context, and deduplicates multi-angle shots into single meal events -- with explicit handling for iOS limited photo access
**Depends on**: Phase 1 (needs food classifier to filter food vs non-food; detection method decision gates pipeline architecture)
**Requirements**: GAL-01, GAL-02, GAL-03, GAL-04, GAL-05, GAL-06
**Success Criteria** (what must be TRUE):
  1. User can manually trigger a gallery scan and see detected food photos appear in the app within seconds
  2. App performs background/periodic scanning and surfaces newly discovered food photos without user intervention (when full library access is granted)
  3. App correctly groups multiple photos of the same meal (different angles, before/during eating) into a single meal event
  4. When deduplication confidence is low, user is prompted to confirm whether photos are the same meal or different meals
  5. Each discovered photo retains its EXIF metadata (timestamp, location) displayed as meal context (e.g. "Tuesday 12:34 PM, near home")
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Go Backend & Multi-Region Nutrition
**Goal**: A Go API serves accurate per-ingredient nutritional data from region-appropriate food composition databases covering North America, Europe, APAC/Oceania, and Asia -- with Open Food Facts as a global fallback -- stores food entries, and routes lookups based on user locale
**Depends on**: Phase 1 (model class labels map to nutrition DB entries)
**Requirements**: NUT-01, NUT-02, NUT-03, NUT-04, NUT-05, NUT-06, NUT-07, NUT-08, NUT-09, NUT-10
**Success Criteria** (what must be TRUE):
  1. App retrieves per-ingredient macro breakdown (calories, protein, carbs, fat) from the backend for any detected food item, sourced from the user's region-appropriate database
  2. App retrieves per-ingredient micronutrient data (vitamins, minerals) stored alongside macro data
  3. When the user views a detected meal, the app displays a combined total nutrition estimate (sum of all detected ingredients weighted by estimated portions)
  4. Backend supports nutrition lookups from at least 4 regional database groups: North America (USDA FDC, CNF), EU (CoFID, CIQUAL), APAC/Oceania (AFCD, NZFCD), and Asia (Japan STFCJ and others available)
  5. Backend routes nutrition queries to the appropriate regional database based on user locale or explicit region preference, falling back to Open Food Facts when no regional match exists
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Tracking UI & Entry Management
**Goal**: Users can view their daily food diary, manually search and add foods, edit AI-detected meals, save and reuse recipes, and review linked photos -- the complete food logging experience
**Depends on**: Phase 2 (gallery scanning provides detected food entries), Phase 3 (backend stores entries and serves nutrition data)
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07, UI-08
**Success Criteria** (what must be TRUE):
  1. User can view a daily dashboard showing per-meal and total macros (calories, protein, carbs, fat)
  2. User can search for and manually add a food item in under 10 taps when AI detection fails
  3. User can edit any detected meal's ingredients, portions, and quantities after logging
  4. User can save a corrected meal as a recipe and reuse that recipe to log future meals in one tap
  5. User can view daily and weekly nutrition summaries and tap any logged meal to see its linked photo(s)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: UX Modes & Notifications
**Goal**: Users can choose how much involvement they want in the tracking process, from fully automatic to step-by-step guided editing, and receive a daily nutrition summary notification
**Depends on**: Phase 4 (needs working tracking UI to apply modes to)
**Requirements**: UX-01, UX-02, UX-03
**Success Criteria** (what must be TRUE):
  1. App defaults to zero-effort mode where detected meals are auto-logged and user can edit after the fact
  2. User can switch to confirm-only mode (review each detected meal before it is logged) or guided-edit mode (step-by-step ingredient correction)
  3. User receives a configurable end-of-day notification summarizing total calories, protein, carbs, and fat
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: Scale Detection & Weight Refinement
**Goal**: Users who weigh their food get dramatically more accurate tracking through automatic scale reading, container weight learning, and depth-based portion estimation
**Depends on**: Phase 1 (detection pipeline), Phase 3 (backend for weight storage), Phase 4 (UI to display scale-detected weights)
**Requirements**: DET-04, DET-06, SCL-01, SCL-02, SCL-03
**Success Criteria** (what must be TRUE):
  1. When a kitchen scale is visible in a food photo, the app reads the displayed weight via OCR and pre-fills the weight field
  2. User can save known container/vessel weights, and the app automatically subtracts tare weight from scale readings
  3. App learns frequently used container weights over time and intelligently guesses whether a scale reading includes the container or not
  4. On iPhone Pro devices with LiDAR, the app uses depth data for more accurate portion estimation, with graceful fallback to visual estimation on other devices
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6
Note: Phase 3 can partially overlap with Phase 2 (no dependency between them).

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 1. Food Detection Foundation | 0/TBD | Not started | - |
| 2. Gallery Scanning & Deduplication | 0/TBD | Not started | - |
| 3. Go Backend & Multi-Region Nutrition | 0/TBD | Not started | - |
| 4. Tracking UI & Entry Management | 0/TBD | Not started | - |
| 5. UX Modes & Notifications | 0/TBD | Not started | - |
| 6. Scale Detection & Weight Refinement | 0/TBD | Not started | - |
