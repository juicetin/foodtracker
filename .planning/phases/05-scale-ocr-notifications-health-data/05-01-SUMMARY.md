---
phase: "05"
plan: "01"
subsystem: detection-notifications
tags: [hidden-ingredients, knowledge-graph, push-notifications, preferences]
dependency_graph:
  requires: [knowledge-graph-service, gemini-nano-pipeline]
  provides: [kg-ingredient-enrichment, daily-macro-notifications, notification-preferences]
  affects: [vlm-pipeline, preferences-store, app-config]
tech_stack:
  added: [expo-notifications]
  patterns: [kg-enrichment-pipeline, daily-trigger-notification, cancel-before-reschedule]
key_files:
  created:
    - apps/mobile/src/services/detection/hiddenIngredientsService.ts
    - apps/mobile/src/services/detection/__tests__/hiddenIngredientsService.test.ts
    - apps/mobile/src/services/notifications/notificationService.ts
    - apps/mobile/src/services/notifications/__tests__/notificationService.test.ts
    - apps/mobile/src/__mocks__/expo-notifications.ts
  modified:
    - apps/mobile/src/services/vlm/vlmPipeline.ts
    - apps/mobile/src/services/knowledge-graph/knowledgeGraphService.ts
    - apps/mobile/src/services/knowledge-graph/index.ts
    - apps/mobile/src/types/index.ts
    - apps/mobile/src/store/usePreferencesStore.ts
    - apps/mobile/app.json
    - apps/mobile/jest.config.js
decisions:
  - "getRecipeIngredients made public on KnowledgeGraphService (was private) so hiddenIngredientsService can access ingredient lists"
  - "KG enrichment runs after VLM identification in vlmPipeline.scanFood() for both live and mock paths"
  - "expo-notifications DailyTriggerInput for reliable daily scheduling with cancel-before-reschedule pattern"
  - "Notification defaults: 9 PM, disabled — enabled during onboarding (not in this plan)"
metrics:
  duration: "5min"
  completed: "2026-03-20T17:31:46Z"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 9
  tests_passing: 9
---

# Phase 05 Plan 01: Hidden Ingredients + Daily Notifications Summary

KG ingredient enrichment for dishes without VLM ingredients, plus configurable daily macro push notifications via expo-notifications with DailyTriggerInput.

## Task Results

### Task 1: Hidden ingredients service + VLM pipeline integration

**TDD:** RED (4 failing tests) -> GREEN (implementation + tests pass)

Created `hiddenIngredientsService.ts` with `enrichDishesWithKgIngredients()` that fills empty ingredient arrays from KG recipe data. For each dish with no VLM-provided ingredients, it chains `searchDish -> getCanonicalRecipe -> getRecipeIngredients` and maps KG ingredients to ScannedIngredient with `kgInferred=true` flag. VLM-provided ingredients are never overwritten.

Wired into `vlmPipeline.ts` `scanFood()` after VLM identification, ensuring all dishes get KG enrichment opportunity.

**Commits:** `8f84eb73` (RED), `3b3ac6f4` (GREEN)

### Task 2: Daily macro notification service + preferences extension

**TDD:** RED (5 failing tests) -> GREEN (implementation + tests pass)

Created `notificationService.ts` with `scheduleDailyNotification`, `cancelDailyNotification`, `buildMacroSummaryBody`, `requestNotificationPermission`, and `rescheduleWithFreshContent`. Uses cancel-before-reschedule pattern for clean notification management.

Extended `usePreferencesStore` with `notificationsEnabled` (default: false), `notificationHour` (default: 21), `notificationMinute` (default: 0), and their setters.

Added `expo-notifications` to app.json plugins and jest moduleNameMapper.

**Commits:** `44337aca` (RED), `102f21b1` (GREEN)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Made getRecipeIngredients public on KnowledgeGraphService**
- **Found during:** Task 1
- **Issue:** `getRecipeIngredients` was private; hiddenIngredientsService needs direct access to ingredient lists
- **Fix:** Changed visibility from `private` to `public`, exported `IngredientResult` type
- **Files modified:** knowledgeGraphService.ts, knowledge-graph/index.ts

## Verification

All 9 tests pass across both test suites:
- `hiddenIngredientsService.test.ts`: 4/4
- `notificationService.test.ts`: 5/5

## Self-Check: PASSED

All 5 created files verified on disk. All 4 commits verified in git log.
