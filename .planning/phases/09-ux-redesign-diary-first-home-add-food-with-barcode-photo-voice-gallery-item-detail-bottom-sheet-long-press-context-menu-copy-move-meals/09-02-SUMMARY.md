---
phase: 09-ux-redesign
plan: 02
subsystem: diary-ui
tags: [ui, diary, components, navigation]
dependency_graph:
  requires: [09-01]
  provides: [DiaryHomeScreen, MacroSummaryHeader, DateNavigator, CalendarPicker, MealGroupSection, MealGroupHeader, FoodItemCard]
  affects: [MainTabNavigator, screens/index]
tech_stack:
  added: []
  patterns: [meal-type-grouping, collapsible-sections, calendar-modal, swipe-gesture-navigation]
key_files:
  created:
    - apps/mobile/src/components/diary/MacroSummaryHeader.tsx
    - apps/mobile/src/components/diary/DateNavigator.tsx
    - apps/mobile/src/components/diary/MealGroupHeader.tsx
    - apps/mobile/src/components/diary/FoodItemCard.tsx
    - apps/mobile/src/components/diary/MealGroupSection.tsx
    - apps/mobile/src/components/diary/CalendarPicker.tsx
    - apps/mobile/src/screens/DiaryHomeScreen.tsx
  modified:
    - apps/mobile/src/components/diary/index.ts
    - apps/mobile/src/screens/index.ts
    - apps/mobile/src/navigation/MainTabNavigator.tsx
    - apps/mobile/src/screens/DiaryScreen.tsx
decisions:
  - "DiaryScreen.tsx imports changed to direct file imports (not barrel) since old exports removed from barrel"
metrics:
  duration: 5min
  completed: "2026-03-23T10:57:00Z"
---

# Phase 09 Plan 02: Diary-First Home Screen Summary

Diary-first home screen with 6 new diary components, CalendarPicker modal, and DiaryHomeScreen wired as Today tab replacing old time-period grouped DiaryScreen.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Build diary interaction components | baabd5c7 | MacroSummaryHeader, DateNavigator, MealGroupHeader, FoodItemCard, MealGroupSection, index.ts |
| 2 | Build CalendarPicker modal | 8c7818d7 | CalendarPicker.tsx, index.ts |
| 3 | Create DiaryHomeScreen and wire as Today tab | 0a79cbc4 | DiaryHomeScreen.tsx, screens/index.ts, MainTabNavigator.tsx |

## What Was Built

### MacroSummaryHeader
Remaining calories display (32px, red when over goal) with three horizontal P/C/F progress bars. Protein (#3B82F6), Carbs (#D97706), Fat (#059669) each with label and "Xg / Yg" values. 8px bars clamped at 100%.

### DateNavigator
Arrow buttons + date label. Shows "Today" when on current date. Tap date opens CalendarPicker. Min 44px touch targets. Right arrow disabled/dimmed when on today.

### MealGroupHeader
Expand/collapse with chevron, meal icon, label, calorie subtotal, "+" add food button (#16A34A). 500ms delayLongPress for future context menu (Plan 04).

### FoodItemCard
Compact row replacing ExpandableEntryCard (QA-03: no more toggle states). 40x40 photo thumbnail, dish name, calories, time, P/C/F macro pills. Long-press ref flag suppresses subsequent onPress.

### MealGroupSection
Collapsible container with Reanimated withTiming 250ms animation. Empty state shows "+ Add Food" text. Groups entries under MealGroupHeader.

### CalendarPicker
Modal with transparent backdrop. Custom 42-cell month grid. Month navigation arrows. Selected day: green circle (#16A34A). Today: underlined green text. Other-month days dimmed (#D1D5DB).

### DiaryHomeScreen
Unified diary-first home replacing separate Home+Diary screens. Meal-type grouping (Breakfast/Lunch/Dinner/Snacks). Swipe navigation via Gesture.Pan. State hooks prepared for Plan 04 bottom sheets (selectedEntryId, contextMenuEntry, menuMealGroup). Background #F5F5F5.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed DiaryScreen barrel imports**
- **Found during:** Task 1
- **Issue:** Removing StickyMacroHeader/TimePeriodSection/SearchBar from barrel broke DiaryScreen.tsx imports
- **Fix:** Changed DiaryScreen to use direct file imports instead of barrel
- **Files modified:** apps/mobile/src/screens/DiaryScreen.tsx
- **Commit:** baabd5c7

## Verification

- TypeScript compiles without errors after all 3 tasks
- All acceptance criteria met (component exports, color codes, props, patterns)
