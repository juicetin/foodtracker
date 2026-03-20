/**
 * Diary SQL queries: entries by date with timePeriod, week presence, day totals.
 *
 * Extracted from DiaryScreen.tsx and extended with time-period assignment.
 */

import { opsqlite } from '../../../db/client';
import {
  assignTimePeriod,
  type TimePeriod,
  type TimePeriodBoundary,
  DEFAULT_BOUNDARIES,
} from './timePeriods';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DiaryDish {
  id: string;
  name: string;
  cuisine: string | null;
}

export interface DiaryEntry {
  id: string;
  timePeriod: TimePeriod;
  mealType: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  notes: string | null;
  createdAt: string;
  photoUri: string | null;
  dishes: DiaryDish[];
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/**
 * Load all non-deleted entries for a given date, each annotated with a
 * `timePeriod` derived from `created_at`.
 */
export function loadEntriesForDate(
  dateStr: string,
  boundaries: TimePeriodBoundary = DEFAULT_BOUNDARIES,
): DiaryEntry[] {
  const entryRows = opsqlite.executeSync(
    `SELECT id, meal_type, total_calories, total_protein, total_carbs, total_fat, notes, created_at
     FROM food_entries
     WHERE entry_date = ? AND is_deleted = 0
     ORDER BY created_at ASC`,
    [dateStr],
  ).rows as Array<Record<string, unknown>>;

  return entryRows.map((row) => {
    const entryId = row.id as string;
    const createdAt = row.created_at as string;

    const photoRows = opsqlite.executeSync(
      'SELECT uri FROM photos WHERE entry_id = ? LIMIT 1',
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    const dishRows = opsqlite.executeSync(
      'SELECT id, name, cuisine FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at',
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    return {
      id: entryId,
      timePeriod: assignTimePeriod(createdAt, boundaries),
      mealType: row.meal_type as string,
      totalCalories: (row.total_calories as number) ?? 0,
      totalProtein: (row.total_protein as number) ?? 0,
      totalCarbs: (row.total_carbs as number) ?? 0,
      totalFat: (row.total_fat as number) ?? 0,
      notes: (row.notes as string) ?? null,
      createdAt,
      photoUri: photoRows.length > 0 ? (photoRows[0].uri as string) : null,
      dishes: dishRows.map((d) => ({
        id: d.id as string,
        name: d.name as string,
        cuisine: (d.cuisine as string) ?? null,
      })),
    };
  });
}

/**
 * Load a 7-day entry presence map (Mon-Sun) centered on the given date.
 * Returns Map<dateStr, entryCount>.
 */
export function loadWeekEntryPresence(centerDate: string): Map<string, number> {
  // Find Monday of the week containing centerDate
  const d = new Date(centerDate + 'T12:00:00');
  const dayOfWeek = d.getDay(); // 0=Sun, 1=Mon, ...
  const mondayOffset = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
  const monday = new Date(d);
  monday.setDate(d.getDate() + mondayOffset);
  const mondayStr = dateToStr(monday);

  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);
  const sundayStr = dateToStr(sunday);

  const rows = opsqlite.executeSync(
    `SELECT entry_date, COUNT(*) as count
     FROM food_entries
     WHERE entry_date BETWEEN ? AND ? AND is_deleted = 0
     GROUP BY entry_date`,
    [mondayStr, sundayStr],
  ).rows as Array<Record<string, unknown>>;

  const result = new Map<string, number>();
  for (const row of rows) {
    result.set(row.entry_date as string, row.count as number);
  }
  return result;
}

/**
 * Sum macro totals from an array of diary entries.
 */
export function computeDayTotals(
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

// ---------------------------------------------------------------------------
// Utility functions (extracted from DiaryScreen)
// ---------------------------------------------------------------------------

/** Today's date as YYYY-MM-DD. */
export function getTodayDateStr(): string {
  return new Date().toISOString().split('T')[0];
}

/** Format a Date object to YYYY-MM-DD. */
export function dateToStr(d: Date): string {
  return d.toISOString().split('T')[0];
}

/** Human-readable date label: 'Today', 'Yesterday', or formatted date. */
export function formatDateLabel(dateStr: string): string {
  const today = getTodayDateStr();
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = dateToStr(yesterday);

  if (dateStr === today) return 'Today';
  if (dateStr === yesterdayStr) return 'Yesterday';

  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}
