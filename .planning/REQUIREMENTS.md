# Requirements: FoodTracker

**Defined:** 2026-03-12
**Core Value:** Accurate, effortless food tracking from photos you already take — no manual entry, no barcode scanning, no subscription, just eat, photograph, and review.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Detection

- [x] **DET-01**: User can photograph food and get on-device identification of food items with bounding boxes via YOLO (CoreML/LiteRT)
- [x] **DET-02**: User's device automatically selects the appropriate VLM tier (SmolVLM-256M / Moondream 0.5B / Gemma 3n E2B) based on device capability, downloaded post-install
- [x] **DET-03**: User on a supported device (Pixel 8+, Galaxy S24+, etc.) gets opportunistic Gemini Nano inference for food identification via AICore
- [x] **DET-04**: User sees inferred hidden ingredients from dish identification via knowledge graph lookup (e.g., "carbonara" -> egg, pancetta, parmesan)
- [x] **DET-05**: DROPPED -- Gemini Nano does not produce confidence scores; VLM-only pipeline makes confidence indicators obsolete
- [x] **DET-06**: DROPPED -- Gemini Nano provides gram estimates directly; portionBridge and visual portion estimation unused

### Data & Storage

- [x] **DAT-01**: All user data (food entries, recipes, preferences, history) is stored locally via op-sqlite with no backend dependency
- [x] **DAT-02**: User has access to a bundled USDA FDC nutrition database (~50-80MB) delivered as fast-follow asset pack, available before first food log
- [x] **DAT-03**: User can download optional regional nutrition databases (AFCD, CoFID, CIQUAL) for non-US food coverage
- [x] **DAT-04**: User can opt into Google Drive backup/sync via app data folder (cross-platform)
- [x] **DAT-05**: User on iOS can opt into iCloud backup/sync
- [x] **DAT-06**: Sync conflicts are resolved via last-write-wins with timestamps, with full edit history retained locally

### Gallery Scanning

- [x] **GAL-01**: User can manually trigger a gallery scan to discover and process recent food photos
- [x] **GAL-02**: App performs background/periodic scanning to surface newly discovered food photos without user intervention
- [x] **GAL-03**: App correctly groups multiple photos of the same meal (temporal clustering within 5-min window + GPS proximity) into a single meal event
- [x] **GAL-04**: Each discovered photo retains EXIF metadata (timestamp, location) displayed as meal context
- [x] **GAL-05**: Background scanning works within platform constraints (iOS 30-second BGTask limit, Android WorkManager) using chunked processing blocks

### UI & Diary

- [x] **UI-01**: User can view a daily food diary organized by meal (breakfast/lunch/dinner/snacks) with per-meal and daily macro totals
- [x] **UI-02**: User can search and manually add foods from the bundled USDA database in under 7 taps
- [x] **UI-03**: User can edit any logged meal's ingredients, portions, and quantities after logging
- [x] **UI-04**: User can save a corrected meal as a recipe and reuse it in one tap
- [x] **UI-05**: User can create nested recipes (recipes containing other recipes) with expandable detail view
- [x] **UI-06**: When editing a nested recipe, user is prompted whether to modify it only in the parent context or update the original recipe as well
- [x] **UI-07**: User can view linked photo(s) for any logged meal
- [x] **UI-08**: User can choose UX mode: zero-effort (auto-log, daily review), confirm-only (review before logging), or guided-edit (step-by-step correction)

### Scale & Weight

- [x] **SCL-01**: When a kitchen scale is visible in a food photo, the app reads the displayed weight via custom 7-segment TFLite OCR
- [x] **SCL-02**: User can save known container/vessel weights, and the app auto-subtracts tare weight from scale readings
- [x] **SCL-03**: App learns frequently used container weights over time

### Notifications & Tracking

- [x] **NTF-01**: User receives a configurable end-of-day push notification summarizing daily macro totals
- [x] **NTF-02**: User can import weight data from Apple Health / Google Fit and view smoothed weight trend

### Model Delivery

- [x] **MDL-01**: Android app delivers ML models via Play for On-Device AI with device targeting by RAM and chipset
- [x] **MDL-02**: iOS app delivers optional models via On-Demand Resources or Background Assets API

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

### On-Device Embedding

- [x] **EMB-01**: MiniLM-L6-v2 exported as TFLite INT8 with mean pooling and L2 normalization baked into the graph, producing 384-dim normalized float32 vectors
- [x] **EMB-02**: WordPiece vocabulary (30522 tokens) extracted from HuggingFace tokenizer and bundled as JSON asset
- [x] **EMB-03**: Pure-JS WordPiece tokenizer handles lowercasing, punctuation splitting, subword splitting with ## prefixes, [CLS]/[SEP] special tokens, attention mask generation
- [x] **EMB-04**: EmbeddingService loads TFLite model via react-native-fast-tflite with lazy initialization on first detection flow
- [x] **EMB-05**: Vec search path in vlmPipeline activates when embedding service is ready, enabling semantic USDA food matching alongside BM25

### UX Redesign

- [ ] **UX-01**: Diary-first home screen with remaining calories display and P/C/F macro progress bars, replacing separate Home + Diary tabs
- [ ] **UX-02**: Date navigation with swipe gestures, arrow buttons, and calendar picker modal on date tap
- [ ] **UX-03**: Meal group headers (Breakfast/Lunch/Dinner/Snacks) with tap expand/collapse and long-press menu, replacing time-period grouping
- [ ] **UX-04**: Food item tap opens bottom sheet detail (read-only macros, ingredients, expandable sections) instead of navigating to full screen
- [ ] **UX-05**: Food item long press opens context menu with Copy to clipboard, Copy to another day, Move to other meal, Save as favorite, Delete
- [ ] **UX-06**: Unified Add Food screen with search bar containing camera, voice, and barcode icons
- [ ] **UX-07**: Quick access tabs (Recent, Frequent, Favorites, My Recipes) on Add Food screen with personal food history
- [ ] **UX-08**: Barcode scanning integrated into Add Food screen search bar (always visible)
- [ ] **UX-09**: Voice input hint for food description via keyboard voice button
- [ ] **UX-10**: AI Photo Scan results display with per-dish ingredient breakdown and macros
- [ ] **UX-11**: Copy entry to another day and move entry to different meal type operations
- [ ] **UX-12**: Item Detail Bottom Sheet with expandable micronutrients, nutrition source, and view photo sections
- [ ] **QA-01**: Fix long press diary item crash (replaces broken handler with new context menu)
- [ ] **QA-02**: Fix re-log tap behavior (tap now opens detail bottom sheet, not re-log action)
- [ ] **QA-03**: Remove third toggle view on diary items (keep two states: summary + ingredients)
- [ ] **QA-06**: Barcode option always visible on add food screen

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
| DET-04 | Phase 5 | Complete |
| DET-05 | Phase 7.1 | Dropped |
| DET-06 | Phase 7.1 | Dropped |
| DAT-01 | Phase 1 | Complete |
| DAT-02 | Phase 1 | Complete |
| DAT-03 | Phase 1 | Complete |
| DAT-04 | Phase 6 | Complete |
| DAT-05 | Phase 6 | Complete |
| DAT-06 | Phase 6 | Complete |
| GAL-01 | Phase 4 | Complete |
| GAL-02 | Phase 4 | Complete |
| GAL-03 | Phase 4 | Complete |
| GAL-04 | Phase 4 | Complete |
| GAL-05 | Phase 4 | Complete |
| UI-01 | Phase 3 | Complete |
| UI-02 | Phase 3 | Complete |
| UI-03 | Phase 3 | Complete |
| UI-04 | Phase 3 | Complete |
| UI-05 | Phase 3 | Complete |
| UI-06 | Phase 3 | Complete |
| UI-07 | Phase 3 | Complete |
| UI-08 | Phase 3 | Complete |
| SCL-01 | Phase 5 | Complete |
| SCL-02 | Phase 5 | Complete |
| SCL-03 | Phase 5 | Complete |
| NTF-01 | Phase 5 | Complete |
| NTF-02 | Phase 5 | Complete |
| MDL-01 | Phase 6 | Complete |
| MDL-02 | Phase 6 | Complete |

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

| EMB-01 | Phase 8 | Planned |
| EMB-02 | Phase 8 | Planned |
| EMB-03 | Phase 8 | Planned |
| EMB-04 | Phase 8 | Planned |
| EMB-05 | Phase 8 | Planned |

| UX-01 | Phase 9 | Planned |
| UX-02 | Phase 9 | Planned |
| UX-03 | Phase 9 | Planned |
| UX-04 | Phase 9 | Planned |
| UX-05 | Phase 9 | Planned |
| UX-06 | Phase 9 | Planned |
| UX-07 | Phase 9 | Planned |
| UX-08 | Phase 9 | Planned |
| UX-09 | Phase 9 | Planned |
| UX-10 | Phase 9 | Planned |
| UX-11 | Phase 9 | Planned |
| UX-12 | Phase 9 | Planned |
| QA-01 | Phase 9 | Planned |
| QA-02 | Phase 9 | Planned |
| QA-03 | Phase 9 | Planned |
| QA-06 | Phase 9 | Planned |

**Coverage:**
- v1 requirements: 60 total
- Mapped to phases: 60
- Unmapped: 0

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-23 after Phase 9 planning*
