---
phase: 09-ux-redesign
plan: 04
subsystem: diary-sheets
tags: [ui, bottom-sheet, context-menu, interaction, copy-move]
dependency_graph:
  requires: [09-02]
  provides: [ItemDetailSheet, ContextMenuSheet, MealGroupMenuSheet]
  affects: [DiaryHomeScreen]
tech_stack:
  added: []
  patterns: [bottom-sheet-overlay, haptic-feedback, context-menu, calendar-picker-reuse]
key_files:
  created:
    - apps/mobile/src/components/sheets/ItemDetailSheet.tsx
    - apps/mobile/src/components/sheets/ContextMenuSheet.tsx
    - apps/mobile/src/components/sheets/MealGroupMenuSheet.tsx
  modified:
    - apps/mobile/src/screens/DiaryHomeScreen.tsx
decisions:
  - "Calendar picker reused for copy-to-day and copy-from-date via datePickerContext state pattern"
  - "Sheets rendered outside GestureDetector to avoid gesture conflicts"
  - "Delete action shows confirmation alert then soft-deletes via useFoodLogStore.deleteEntry"
  - "Move to meal uses Alert action sheet with filtered meal options"
metrics:
  duration: 3min
  completed: "2026-03-23T10:58:00Z"
---

# Phase 09 Plan 04: Item Detail, Context Menu, and Meal Group Bottom Sheets Summary

Three bottom sheet overlays wired into DiaryHomeScreen: tap opens read-only detail sheet, long-press opens context menu with copy/move/favorite/delete, header long-press opens meal group menu with copy/template actions.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Build three bottom sheet components | f3b07df4 | ItemDetailSheet.tsx, ContextMenuSheet.tsx, MealGroupMenuSheet.tsx |
| 2 | Wire sheets into DiaryHomeScreen with action handlers | d9c64a33 | DiaryHomeScreen.tsx |

## What Was Built

### ItemDetailSheet
Read-only bottom sheet at 50%/90% snap points. Shows dish name with favourite/edit/delete icons, time logged, total macros (32px cal + P/C/F pills), ingredient list with per-ingredient name/amount/macros, expandable sections for micronutrients (fiber/sugar), nutrition source (USDA/OFF), and photo. "+ Add Ingredient" button navigates to full edit screen.

### ContextMenuSheet
Long-press action sheet at 35% snap point with haptic feedback. Five actions: Copy to Clipboard (formats and copies via expo-clipboard), Copy to Another Day (opens CalendarPicker then calls copyEntryToDate), Move to Other Meal (Alert action sheet then calls moveEntryToMeal), Save as Favorite (calls addFavourite), Delete (confirmation alert then soft-delete).

### MealGroupMenuSheet
Header long-press action sheet at 28% snap point with haptic feedback. Three actions: Copy from Another Day (opens CalendarPicker then calls copyAllEntriesFromDate), Copy Yesterday's Meal (calls copyAllEntriesFromDate with yesterday), Save as Meal Template (placeholder alert).

### DiaryHomeScreen Integration
All three sheets rendered outside GestureDetector to avoid gesture conflicts. Calendar picker reused for copy date selection via datePickerContext state. Action handlers implement full copy/move/delete/favorite workflows with confirmation alerts and data refresh.

### QA Fixes
- QA-01 (long press crash): Fixed by properly wiring long-press to ContextMenuSheet instead of unhandled state
- QA-02 (tap re-logs directly): Fixed by wiring tap to ItemDetailSheet read-only view instead of direct action

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- TypeScript compiles without errors after both tasks
- All acceptance criteria met for both tasks
