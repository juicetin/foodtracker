/**
 * Meal group constants, types, and grouped entry loading.
 *
 * Defines the four meal groups (breakfast, lunch, dinner, snacks) with
 * display config and provides functions to load diary entries grouped
 * by meal and compute per-group macro totals.
 */

import { loadEntriesForDate, type DiaryEntry } from './diaryQueries';
import { DEFAULT_BOUNDARIES } from './timePeriods';

// ---------------------------------------------------------------------------
// Constants & Types
// ---------------------------------------------------------------------------

export const MEAL_GROUPS = ['breakfast', 'lunch', 'dinner', 'snacks'] as const;

export type MealGroup = typeof MEAL_GROUPS[number];

export const MEAL_GROUP_CONFIG: Record<MealGroup, { label: string; icon: string }> = {
  breakfast: { label: 'Breakfast', icon: 'sunny-outline' },
  lunch: { label: 'Lunch', icon: 'partly-sunny-outline' },
  dinner: { label: 'Dinner', icon: 'moon-outline' },
  snacks: { label: 'Snacks', icon: 'cafe-outline' },
};

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/**
 * Load all entries for a date, grouped by meal type.
 * Returns a Map with all 4 meal groups (empty arrays for groups with no entries).
 * Normalizes 'snack' -> 'snacks'.
 */
export function loadEntriesGroupedByMeal(dateStr: string): Map<MealGroup, DiaryEntry[]> {
  const entries = loadEntriesForDate(dateStr, DEFAULT_BOUNDARIES);

  const grouped = new Map<MealGroup, DiaryEntry[]>();
  for (const group of MEAL_GROUPS) {
    grouped.set(group, []);
  }

  for (const entry of entries) {
    // Normalize 'snack' to 'snacks'
    const mealType = entry.mealType === 'snack' ? 'snacks' : entry.mealType;
    const group = MEAL_GROUPS.includes(mealType as MealGroup)
      ? (mealType as MealGroup)
      : 'snacks'; // fallback unknown meal types to snacks
    grouped.get(group)!.push(entry);
  }

  return grouped;
}

/**
 * Sum macro totals for an array of diary entries.
 */
export function computeMealGroupTotals(
  entries: DiaryEntry[],
): { calories: number; protein: number; carbs: number; fat: number } {
  return entries.reduce(
    (acc, e) => ({
      calories: acc.calories + e.totalCalories,
      protein: acc.protein + e.totalProtein,
      carbs: acc.carbs + e.totalCarbs,
      fat: acc.fat + e.totalFat,
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0 },
  );
}
