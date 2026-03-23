---
phase: 09-ux-redesign
plan: 01
subsystem: navigation, services
tags: [navigation, types, services, insights]
dependency_graph:
  requires: []
  provides: [MainTabNavigator-4tabs, mealGroups-service, copyMoveService, InsightsScreen, AddFood-route]
  affects: [DiaryScreen, DetectionScreen, ProfileScreen]
tech_stack:
  added: [expo-haptics, expo-clipboard]
  patterns: [meal-group-constants, copy-move-service, tab-restructure]
key_files:
  created:
    - apps/mobile/src/services/diary/mealGroups.ts
    - apps/mobile/src/services/diary/copyMoveService.ts
    - apps/mobile/src/screens/InsightsScreen.tsx
  modified:
    - apps/mobile/src/types/index.ts
    - apps/mobile/src/navigation/MainTabNavigator.tsx
    - apps/mobile/src/navigation/RootNavigator.tsx
    - apps/mobile/src/navigation/types.ts
    - apps/mobile/src/screens/index.ts
    - apps/mobile/package.json
decisions:
  - "FAB diameter 56px (up from 52px) per UI-SPEC"
  - "Unknown meal types fallback to snacks group in loadEntriesGroupedByMeal"
  - "AddFoodPlaceholder inline in RootNavigator (not separate file) since temporary"
metrics:
  duration: 3min
  completed: "2026-03-23T10:50:12Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 10
requirements:
  - UX-01
  - UX-03
  - UX-11
  - QA-01
---

# Phase 09 Plan 01: Foundation - Types, Services, Navigation Summary

Diary-first 4-tab navigation with meal group service layer and InsightsScreen trend extraction from DiaryScreen.

## What Was Built

### Task 1: Dependencies, Types, and Service Layer
- Installed expo-haptics and expo-clipboard
- Added `AddFood: { mealType?: string }` to RootStackParamList
- Changed MainTabParamList to Today/Add/Insights/Profile
- Created mealGroups.ts: MEAL_GROUPS constant, MealGroup type, MEAL_GROUP_CONFIG with labels/icons, loadEntriesGroupedByMeal (groups entries by meal type with snack normalization), computeMealGroupTotals
- Created copyMoveService.ts: copyEntryToDate (deep copies entry with dishes and ingredients), moveEntryToMeal (updates meal_type), copyAllEntriesFromDate (bulk copy with optional meal filter)

### Task 2: Navigation Restructure and InsightsScreen
- Restructured MainTabNavigator from Home/Detect/Diary/Profile to Today/Add/Insights/Profile
- Today tab uses existing DiaryScreen (Plan 02 will replace with DiaryHomeScreen)
- FAB center button (56px diameter, #16A34A) navigates to AddFood route
- Created InsightsScreen with trend range selector (7D/14D/30D/All), calorie bar chart, stats row (avg kcal, on target %, days logged, streak), macro averages
- Added AddFood placeholder screen to RootNavigator stack
- Updated navigation types (TodayScreenNavigationProp, InsightsScreenNavigationProp)

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 82afcc1a | Install deps, define types and service layer |
| 2 | 55086670 | Restructure navigation and create InsightsScreen |

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- TypeScript compiles without errors (`npx tsc --noEmit` passes)
- All acceptance criteria met for both tasks
