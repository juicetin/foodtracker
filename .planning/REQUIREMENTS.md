# Requirements: FoodTracker

**Defined:** 2026-03-12
**Core Value:** Accurate, effortless food tracking from photos you already take — no manual entry, no barcode scanning, no subscription, just eat, photograph, and review.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Detection

- [x] **DET-01**: User can photograph food and get on-device identification of food items with bounding boxes via YOLO (CoreML/LiteRT)
- [x] **DET-02**: User's device automatically selects the appropriate VLM tier (SmolVLM-256M / Moondream 0.5B / Gemma 3n E2B) based on device capability, downloaded post-install
- [x] **DET-03**: User on a supported device (Pixel 8+, Galaxy S24+, etc.) gets opportunistic Gemini Nano inference for food identification via AICore
- [ ] **DET-04**: User sees inferred hidden ingredients from dish identification via knowledge graph lookup (e.g., "carbonara" -> egg, pancetta, parmesan)
- [x] **DET-05**: User sees confidence indicators (green/yellow/red) on detection results and can manually correct when confidence is low
- [x] **DET-06**: User sees portion estimates based on visual cues (plate size, reference objects, density tables) from the on-device portion estimator

### Data & Storage

- [x] **DAT-01**: All user data (food entries, recipes, preferences, history) is stored locally via op-sqlite with no backend dependency
- [x] **DAT-02**: User has access to a bundled USDA FDC nutrition database (~50-80MB) delivered as fast-follow asset pack, available before first food log
- [x] **DAT-03**: User can download optional regional nutrition databases (AFCD, CoFID, CIQUAL) for non-US food coverage
- [ ] **DAT-04**: User can opt into Google Drive backup/sync via app data folder (cross-platform)
- [ ] **DAT-05**: User on iOS can opt into iCloud backup/sync
- [ ] **DAT-06**: Sync conflicts are resolved via last-write-wins with timestamps, with full edit history retained locally

### Gallery Scanning

- [ ] **GAL-01**: User can manually trigger a gallery scan to discover and process recent food photos
- [ ] **GAL-02**: App performs background/periodic scanning to surface newly discovered food photos without user intervention
- [ ] **GAL-03**: App correctly groups multiple photos of the same meal (temporal clustering within 5-min window + GPS proximity) into a single meal event
- [ ] **GAL-04**: Each discovered photo retains EXIF metadata (timestamp, location) displayed as meal context
- [ ] **GAL-05**: Background scanning works within platform constraints (iOS 30-second BGTask limit, Android WorkManager) using chunked processing blocks

### UI & Diary

- [x] **UI-01**: User can view a daily food diary organized by meal (breakfast/lunch/dinner/snacks) with per-meal and daily macro totals
- [x] **UI-02**: User can search and manually add foods from the bundled USDA database in under 7 taps
- [ ] **UI-03**: User can edit any logged meal's ingredients, portions, and quantities after logging
- [ ] **UI-04**: User can save a corrected meal as a recipe and reuse it in one tap
- [x] **UI-05**: User can create nested recipes (recipes containing other recipes) with expandable detail view
- [x] **UI-06**: When editing a nested recipe, user is prompted whether to modify it only in the parent context or update the original recipe as well
- [ ] **UI-07**: User can view linked photo(s) for any logged meal
- [ ] **UI-08**: User can choose UX mode: zero-effort (auto-log, daily review), confirm-only (review before logging), or guided-edit (step-by-step correction)

### Scale & Weight

- [ ] **SCL-01**: When a kitchen scale is visible in a food photo, the app reads the displayed weight via custom 7-segment TFLite OCR
- [ ] **SCL-02**: User can save known container/vessel weights, and the app auto-subtracts tare weight from scale readings
- [ ] **SCL-03**: App learns frequently used container weights over time

### Notifications & Tracking

- [ ] **NTF-01**: User receives a configurable end-of-day push notification summarizing daily macro totals
- [ ] **NTF-02**: User can import weight data from Apple Health / Google Fit and view smoothed weight trend

### Model Delivery

- [ ] **MDL-01**: Android app delivers ML models via Play for On-Device AI with device targeting by RAM and chipset
- [ ] **MDL-02**: iOS app delivers optional models via On-Demand Resources or Background Assets API

### Backup & Sync

- [x] **BKP-01**: op-sqlite updateHook writes to a change journal table recording table, rowid, operation, and timestamp for every INSERT/UPDATE/DELETE
- [x] **BKP-02**: Manual or automatic backup trigger exports journal entries as a JSON changeset file (incremental diff since last backup)
- [x] **BKP-03**: Periodic full backup via VACUUM INTO creates a clean, portable .db snapshot
- [x] **BKP-04**: Compaction tool replays incremental JSON diffs onto the last full backup to produce a merged full backup
- [x] **BKP-05**: Backup includes ALL user data: food entries, ingredients, dishes, photos, recipes, favourites, OFF cache, preferences
- [x] **BKP-06**: Backup files stored on local device storage accessible via Files app

### ML Pipeline Expansion

- [x] **ML-01**: App uses a custom-trained 335+ class food classifier (EfficientNet-Lite0) replacing the generic AIY Food V1 model, with ImageNet-normalized 224x224 input
- [x] **ML-02**: Detection pipeline uses a food-specific YOLO model (241 food classes) instead of COCO YOLO (10 food classes) for accurate multi-dish separation
- [x] **ML-03**: Classifier covers 700+ food classes spanning all major global cuisines (East Asian, South Asian, Southeast Asian, African, Middle Eastern, European, Americas) via merged training datasets
- [x] **ML-04**: Nutrition database is enriched with Open Food Facts product data (4.4M products), Nutrition5k per-dish calorie data, and curated regional nutrition tables
- [x] **ML-05**: Food labels include multilingual names, cuisine tags, and cultural context from WorldCuisines knowledge base (2,414 dishes, 35+ countries)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Sync Enhancements

- **SYNC-01**: User can sync data via WebDAV (Nextcloud, etc.) for self-hosters
- **SYNC-02**: Row-level sync via PowerSync for multi-device real-time use

### Detection Enhancements

- **DET-07**: Domain-specific model distillation (JDNet-style) for improved food-specific accuracy
- **DET-08**: LiDAR-based depth estimation for portion sizing on iPhone Pro devices

### UI Enhancements

- **UI-09**: Micronutrient deep-dive UI (vitamins, minerals from bundled USDA data)
- **UI-10**: Barcode scanning via Open Food Facts integration
- **UI-11**: Correlation graphs (nutrition vs exercise vs weight over time)

### Health Integration

- **HEALTH-01**: Full bidirectional Apple Health / Google Fit sync (nutrition data export, exercise import)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cloud-based AI fallback | Breaks zero-cost guarantee. May revisit if data shows material accuracy gaps. |
| AI coaching / adaptive TDEE | Different product. Compete on friction, not coaching. |
| Real-time camera detection | Battery drain + thermal throttling. Photo review, not live camera. |
| Social features | Requires server. Personal tracking tool. |
| Meal planning | Different product. Track, not prescribe. |
| Gamification | Anxiety-driven engagement. Target user is intrinsically motivated. |
| Web app | On-device ML makes web impractical. |
| Backend server | Local-first per ADR-005. |
| Subscription | Zero-cost is core differentiator. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DET-01 | Phase 2 | Complete |
| DET-02 | Phase 5 | Complete |
| DET-03 | Phase 5 | Complete |
| DET-04 | Phase 5 | Pending |
| DET-05 | Phase 2 | Complete |
| DET-06 | Phase 2 | Complete |
| DAT-01 | Phase 1 | Complete |
| DAT-02 | Phase 1 | Complete |
| DAT-03 | Phase 1 | Complete |
| DAT-04 | Phase 6 | Pending |
| DAT-05 | Phase 6 | Pending |
| DAT-06 | Phase 6 | Pending |
| GAL-01 | Phase 4 | Pending |
| GAL-02 | Phase 4 | Pending |
| GAL-03 | Phase 4 | Pending |
| GAL-04 | Phase 4 | Pending |
| GAL-05 | Phase 4 | Pending |
| UI-01 | Phase 3 | Complete |
| UI-02 | Phase 3 | Complete |
| UI-03 | Phase 3 | Pending |
| UI-04 | Phase 3 | Pending |
| UI-05 | Phase 3 | Complete |
| UI-06 | Phase 3 | Complete |
| UI-07 | Phase 3 | Pending |
| UI-08 | Phase 3 | Pending |
| SCL-01 | Phase 5 | Pending |
| SCL-02 | Phase 5 | Pending |
| SCL-03 | Phase 5 | Pending |
| NTF-01 | Phase 5 | Pending |
| NTF-02 | Phase 5 | Pending |
| MDL-01 | Phase 6 | Pending |
| MDL-02 | Phase 6 | Pending |

| ML-01 | Phase 2.2 | Complete |
| ML-02 | Phase 2.3 | Complete |
| ML-03 | Phase 2.4 | Complete |
| ML-04 | Phase 2.5 | Complete |
| ML-05 | Phase 2.5 | Complete |

| BKP-01 | Phase 3.6 | Complete |
| BKP-02 | Phase 3.6 | Complete |
| BKP-03 | Phase 3.6 | Complete |
| BKP-04 | Phase 3.6 | Complete |
| BKP-05 | Phase 3.6 | Complete |
| BKP-06 | Phase 3.6 | Complete |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after v1.0 roadmap creation*
