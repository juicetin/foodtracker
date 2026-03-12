# Phase 2: On-Device Detection Pipeline - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete on-device food detection pipeline: YOLO training completion, model export to CoreML (iOS) and LiteRT (Android), mobile ML integration via react-native-fast-tflite, inference router, and detection results UI. Users photograph food and see identified items with bounding boxes, confidence indicators, portion estimates, and can correct wrong detections before logging. Single photo per detection — multi-photo meal grouping is Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Detection result presentation
- Confidence-colored bounding box outlines with floating label chips (food name + confidence %) above each box
- Box colors map to confidence: green (≥80%), yellow (50-79%), red (<50%)
- Tapping a bounding box or list item opens a detail card via @gorhom/bottom-sheet showing: food name, confidence, portion estimate, macros preview, edit button
- Annotated photo is interactive: pinch-to-zoom and pan (react-native-gesture-handler already installed)
- Summary bar below photo: "4 items detected • ~650 cal • 45g protein" — quick-glance aggregate
- Simple spinner with "Detecting foods..." text during inference — no animated scanning effect, no inference timing shown
- Results persist on screen until user taps "Log Meal" or dismisses — no auto-timeout
- "Log Meal" is a floating action button (FAB) at bottom-right with item count badge

### Item removal
- X button on each bounding box label chip for dismissal
- Swipe-to-dismiss also works as power-user shortcut (both patterns available)
- **Removal must be undoable** — show undo toast after dismissal, item restored if tapped within timeout
- Manual "add missed item" deferred to Phase 3 (UI-02 manual food search)

### Confidence & correction flow
- Threshold split: green ≥80%, yellow 50-79%, red <50%
- All detected items auto-included in the meal regardless of confidence — red items just have a visual flag encouraging review, not excluded
- Correction: tap bounding box → bottom sheet → tap food name → search/replace from nutrition DB (Phase 1 bundled USDA + regional DBs)
- Local correction history stored in SQLite — over time, suggest user's preferred label when similar detections recur (all on-device)

### Portion estimation display
- Grams as primary unit, descriptive as secondary: "~150g (1 medium)"
- Portion adjustment via slider in the detail card: 0.5x to 3x of estimated portion, macros update in real-time
- When no reliable estimate available: use USDA standard serving size as fallback, show subtle "estimated" badge
- Reference object scaling: when standard dinner plate (~26cm) or common utensils are detected in the photo, use them to calibrate portion size (portion_estimator.py already supports this)

### Multi-item meal handling
- All bounding boxes visible on photo simultaneously, with scrollable item list below
- Tapping either box or list item opens the same detail bottom sheet; tapping one highlights the other to show connection
- Items sorted by confidence (highest first) — no grouping by tier
- Single photo per detection in this phase (multi-photo grouping is Phase 4)
- Meal type auto-detected from time of day: before 10am = breakfast, 10am-2pm = lunch, 2pm-5pm = snack, 5pm-9pm = dinner, 9pm+ = snack. User can change in summary bar.

### Claude's Discretion
- YOLO training hyperparameters and augmentation strategy
- CoreML/LiteRT export pipeline implementation details
- react-native-fast-tflite integration approach
- Inference router architecture (binary → detect → classify pipeline orchestration)
- Model pack format and metadata within the existing pack system
- Detail card layout and spacing
- Exact slider behavior and haptic feedback
- Undo toast duration and animation
- List item row design
- Cross-highlight animation between box and list

</decisions>

<specifics>
## Specific Ideas

- User explicitly wants item removal to be undoable — undo toast pattern, not destructive delete
- Detection results must feel trustworthy: confidence colors provide immediate visual signal without requiring user to read numbers
- Slider for portion adjustment (0.5x-3x) instead of text input — fast, one-handed, real-time macro updates
- Reference object calibration (plate size, utensils) when detected — leverages existing portion_estimator.py capability
- Correction learning over time is in scope for this phase — store corrections locally in SQLite for future suggestions

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `training/train_binary.py`, `train_detect.py`, `train_classify.py`: YOLO 3-stage training pipeline ready, needs completion (datasets not yet merged)
- `training/configs/food-detect.yaml`, `food-binary.yaml`, `food-classify.yaml`: Dataset configs (class lists TBD after auto-labeling)
- `training/datasets/scripts/auto_label.py`, `merge_datasets.py`, `download_datasets.py`: Dataset acquisition pipeline
- `training/portion_estimator.py`: Portion estimation module with reference object scaling support
- `training/benchmark.py`: VLM benchmark (PaliGemma 2 3B) — validated, not directly used in Phase 2
- `apps/mobile/src/services/packs/packManager.ts`: Generic pack manager with `type: 'nutrition' | 'model'` — ready for ML model packs
- `apps/mobile/src/services/packs/types.ts`: Pack types already support model delivery
- `@gorhom/bottom-sheet 5.2`: Bottom sheet for detail cards
- `react-native-reanimated 4.2` + `react-native-gesture-handler 2.30`: Animation and gestures for interactive photo, swipe-to-dismiss

### Established Patterns
- Zustand for state management (thin cache over SQLite per Phase 1 decisions)
- op-sqlite for local data via `db/client.ts` canonical location
- Write-first-then-refresh pattern for store mutations
- Soft-delete via isDeleted flag
- expo-file-system v19 class-based API for file operations
- Raw SQL via op-sqlite for nutrition queries (separate from drizzle schema)

### Integration Points
- PackManager: model packs downloaded/cached through same R2 system as nutrition DBs
- Nutrition DB: detection → food name → nutrition lookup via `nutritionService.ts` and `regionalResolver.ts`
- `db/schema.ts`: New tables needed for correction history, detection results cache
- Bottom tab navigation: detection screen as new tab or triggered from photo picker

</code_context>

<deferred>
## Deferred Ideas

- Manual food search and add when detection misses items — Phase 3 (UI-02)
- Multi-photo meal grouping (overhead + close-up merged detections) — Phase 4
- Animated scanning effect during inference — may revisit if simple spinner feels too basic
- Step-through review mode for individual items — may revisit in Phase 5

</deferred>

---

*Phase: 02-on-device-detection-pipeline*
*Context gathered: 2026-03-12*
