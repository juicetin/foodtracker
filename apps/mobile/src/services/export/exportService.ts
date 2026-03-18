/**
 * Export service — generate CSV and JSON from food diary data.
 *
 * Exports food entries, recipes, and favourites.
 * Supports date range filtering.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ExportEntry {
  date: string;
  mealType: string;
  foodName: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  notes: string | null;
}

export interface ExportRecipe {
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface ExportFavourite {
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  timesUsed: number;
}

export interface ExportOptions {
  startDate?: string; // YYYY-MM-DD
  endDate?: string;   // YYYY-MM-DD
  format: 'csv' | 'json';
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

export function loadExportEntries(startDate?: string, endDate?: string): ExportEntry[] {
  let query = `
    SELECT fe.entry_date, fe.meal_type, fe.total_calories, fe.total_protein,
           fe.total_carbs, fe.total_fat, fe.notes,
           GROUP_CONCAT(sd.name, ', ') AS dish_names
    FROM food_entries fe
    LEFT JOIN scanned_dishes sd ON sd.entry_id = fe.id
    WHERE fe.is_deleted = 0`;

  const params: unknown[] = [];

  if (startDate) {
    query += ' AND fe.entry_date >= ?';
    params.push(startDate);
  }
  if (endDate) {
    query += ' AND fe.entry_date <= ?';
    params.push(endDate);
  }

  query += ' GROUP BY fe.id ORDER BY fe.entry_date, fe.created_at';

  try {
    const rows = opsqlite.execute(query, params).rows as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      date: r.entry_date as string,
      mealType: r.meal_type as string,
      foodName: (r.dish_names as string) || (r.notes as string) || 'Logged meal',
      calories: Math.round((r.total_calories as number) ?? 0),
      protein: Math.round((r.total_protein as number) ?? 0),
      carbs: Math.round((r.total_carbs as number) ?? 0),
      fat: Math.round((r.total_fat as number) ?? 0),
      fiber: 0, // TODO: sum from ingredients
      notes: (r.notes as string) ?? null,
    }));
  } catch {
    return [];
  }
}

export function loadExportRecipes(): ExportRecipe[] {
  try {
    const rows = opsqlite.execute(
      'SELECT name, total_calories, total_protein, total_carbs, total_fat FROM custom_recipes ORDER BY name',
    ).rows as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      name: r.name as string,
      calories: Math.round((r.total_calories as number) ?? 0),
      protein: Math.round((r.total_protein as number) ?? 0),
      carbs: Math.round((r.total_carbs as number) ?? 0),
      fat: Math.round((r.total_fat as number) ?? 0),
    }));
  } catch {
    return [];
  }
}

export function loadExportFavourites(): ExportFavourite[] {
  try {
    const rows = opsqlite.execute(
      'SELECT name, total_calories, total_protein, total_carbs, total_fat, times_used FROM favourite_meals ORDER BY times_used DESC',
    ).rows as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      name: r.name as string,
      calories: Math.round((r.total_calories as number) ?? 0),
      protein: Math.round((r.total_protein as number) ?? 0),
      carbs: Math.round((r.total_carbs as number) ?? 0),
      fat: Math.round((r.total_fat as number) ?? 0),
      timesUsed: (r.times_used as number) ?? 0,
    }));
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// CSV generation
// ---------------------------------------------------------------------------

function escapeCsv(value: string | null | number): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function generateCsv(
  entries: ExportEntry[],
  recipes: ExportRecipe[],
  favourites: ExportFavourite[],
): string {
  const lines: string[] = [];

  // Food entries
  lines.push('date,meal_type,food_name,calories,protein_g,carbs_g,fat_g,fiber_g,notes');
  for (const e of entries) {
    lines.push([
      e.date,
      e.mealType,
      escapeCsv(e.foodName),
      e.calories,
      e.protein,
      e.carbs,
      e.fat,
      e.fiber,
      escapeCsv(e.notes),
    ].join(','));
  }

  // Recipes
  if (recipes.length > 0) {
    lines.push('');
    lines.push('# Recipes');
    lines.push('name,calories,protein_g,carbs_g,fat_g');
    for (const r of recipes) {
      lines.push([escapeCsv(r.name), r.calories, r.protein, r.carbs, r.fat].join(','));
    }
  }

  // Favourites
  if (favourites.length > 0) {
    lines.push('');
    lines.push('# Favourites');
    lines.push('name,calories,protein_g,carbs_g,fat_g,times_used');
    for (const f of favourites) {
      lines.push([escapeCsv(f.name), f.calories, f.protein, f.carbs, f.fat, f.timesUsed].join(','));
    }
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// JSON generation
// ---------------------------------------------------------------------------

export function generateJson(
  entries: ExportEntry[],
  recipes: ExportRecipe[],
  favourites: ExportFavourite[],
): string {
  return JSON.stringify({
    app: 'Tastimate',
    version: '1.0.0',
    exportedAt: new Date().toISOString(),
    entries,
    recipes,
    favourites,
  }, null, 2);
}
