# FoodTracker

## What This Is

An AI-powered food tracking app for iOS and Android that passively scans the user's photo gallery to detect food images, identify dishes and ingredients, estimate weights, and calculate nutritional breakdowns — with near-zero manual effort. Built for health-conscious fitness enthusiasts who want accurate macro tracking without the friction of manual food logging.

## Core Value

Accurate, effortless food tracking from photos you already take — no manual entry, no barcode scanning, just eat, photograph, and review.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Express backend with REST API for food log CRUD — existing
- ✓ PostgreSQL database with food entries, ingredients, photos schema — existing
- ✓ Recipe management (create from entry, search, reuse) — existing
- ✓ Retrospective ingredient editing with modification history — existing
- ✓ Multi-region food composition database configuration (AFCD, USDA, CoFID, CIQUAL, Open Food Facts) — existing
- ✓ Food density lookup table for weight estimation — existing
- ✓ YOLO-based food detection POC notebook with end-to-end pipeline — existing
- ✓ Mobile app scaffold with navigation, state management (Zustand), photo picker — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] On-device gallery scanning to discover food photos (background, periodic, and manual modes)
- [ ] EXIF metadata extraction (timestamp, location, camera info) for meal context
- [ ] Intelligent photo deduplication (multiple angles, group meals, before/during/after eating)
- [ ] On-device food detection using YOLO with cloud fallback for accuracy
- [ ] Dish identification → ingredient inference (hidden ingredients from known recipes)
- [ ] Scale/weight detection via OCR with container weight learning over time
- [ ] Global recipe and ingredient database (all cuisines, branded products, home cooking)
- [ ] Per-ingredient macro and micro nutrient breakdown
- [ ] Configurable end-of-day summary notification (total cals, protein, carbs, fat)
- [ ] Editable recipes with serving size adjustment
- [ ] Image retention linked to ingredients for later review and editing
- [ ] Health platform integration (Apple Health, Google Fit) for exercise and weight data
- [ ] Correlation graphs (nutrition vs exercise vs weight over time)
- [ ] Configurable UX modes: zero-effort (default), confirm-only, guided-edit
- [ ] User-managed container weights for improved scale accuracy

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- AI coaching/auto-adjust programs — user wants to track, not be told what to eat
- Social features — personal tracking tool, not a social network
- Meal planning/recommendations — track what you eat, not what you should eat
- Barcode scanning — may add later, but photo-first approach is the differentiator
- Web app — mobile-first, on-device ML makes web impractical

## Context

**Prior work:** Existing Express/TS backend with PostgreSQL, React Native + Expo mobile scaffold, Google ADK agent structure, and a Python YOLO food detection POC notebook. The backend is planned for migration to Go.

**Competitive landscape:** MyFitnessPal and MacroFactor are the main alternatives the user has tried. Both require significant manual effort. Foodvisor and Lose It have photo features but lack the passive gallery scanning approach.

**Technical validation:** The YOLO food detection POC (`spike/food-detection-poc/`) validates that traditional ML can detect food items with bounding boxes. The pretrained COCO model knows ~10 food classes; fine-tuning on Food-101/ISIA-500 is needed for production accuracy.

**Key insight from research:** On-device YOLO runs in ~30ms with near-zero cost. Cloud LLMs cost $0.01-0.03/image and take 2-5 seconds. Hybrid approach (on-device first, cloud fallback for low confidence) gives best accuracy-to-cost ratio.

## Constraints

- **Accuracy first:** Detection and nutrition estimates must be trustworthy enough that users don't second-guess them. Without accuracy, nothing else matters.
- **Backend language:** Go for production backend (strongly typed preference). Python for ML training/prototyping only.
- **Mobile framework:** Open to native (Swift/Kotlin) if on-device ML performance demands it. React Native + Expo is current but not locked in.
- **Agent framework:** ADK is no longer required. Choose the best available technology for the functional and non-functional requirements.
- **Cloud fallback:** Acceptable for steps that can't achieve sufficient accuracy on-device today. Minimize data sent off-device.
- **Database coverage:** Nutrition database must cover global cuisines, branded products, and home cooking — not just Western food.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| YOLO over multimodal LLMs for food detection | Cost ($0 on-device), speed (30ms), spatial accuracy (bounding boxes), privacy | — Pending (POC validates approach) |
| Python for ML spike/POC only | ML ecosystem is Python-first; production serving via ONNX/CoreML/TFLite or Go microservice | ✓ Good |
| Go for production backend | Strongly typed, performant, team preference | — Pending |
| Gallery scanning over in-app camera | Removes friction — user photographs food naturally, app discovers photos later | — Pending |
| Accuracy > cost/speed/offline | Cloud fallback acceptable to ensure estimates are trustworthy | — Pending |
| ADK removed as requirement | Choose best agent framework based on functional needs, not vendor lock-in | — Pending |
| Open to native mobile | If on-device ML is significantly better served by Swift/Kotlin, willing to move from React Native | — Pending |

---
*Last updated: 2026-02-12 after initialization*
