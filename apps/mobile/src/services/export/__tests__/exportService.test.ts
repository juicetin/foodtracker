/**
 * Export service tests — generate CSV and JSON from food diary data.
 */

import {
  generateCsv,
  generateJson,
  type ExportEntry,
  type ExportOptions,
} from '../exportService';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SAMPLE_ENTRIES: ExportEntry[] = [
  {
    date: '2026-03-15',
    mealType: 'breakfast',
    foodName: 'Oatmeal with Berries',
    calories: 350,
    protein: 12,
    carbs: 55,
    fat: 8,
    fiber: 6,
    notes: 'Quick breakfast',
  },
  {
    date: '2026-03-15',
    mealType: 'lunch',
    foodName: 'Grilled Chicken Salad',
    calories: 520,
    protein: 42,
    carbs: 15,
    fat: 32,
    fiber: 4,
    notes: null,
  },
  {
    date: '2026-03-16',
    mealType: 'dinner',
    foodName: 'Pasta Bolognese, Garlic Bread',
    calories: 780,
    protein: 35,
    carbs: 90,
    fat: 28,
    fiber: 5,
    notes: 'Restaurant meal',
  },
];

const SAMPLE_RECIPES = [
  { name: 'Chicken Fried Rice', calories: 600, protein: 30, carbs: 70, fat: 18 },
];

const SAMPLE_FAVOURITES = [
  { name: 'Morning Smoothie', calories: 280, protein: 15, carbs: 40, fat: 6, timesUsed: 12 },
];

// ---------------------------------------------------------------------------
// generateCsv
// ---------------------------------------------------------------------------

describe('generateCsv', () => {
  it('generates CSV with header row', () => {
    const csv = generateCsv(SAMPLE_ENTRIES, [], []);
    const lines = csv.split('\n');
    expect(lines[0]).toBe('date,meal_type,food_name,calories,protein_g,carbs_g,fat_g,fiber_g,notes');
  });

  it('generates correct number of data rows', () => {
    const csv = generateCsv(SAMPLE_ENTRIES, [], []);
    const lines = csv.split('\n').filter(Boolean);
    // 1 header + 3 data + empty sections
    expect(lines.length).toBeGreaterThanOrEqual(4);
  });

  it('escapes commas in food names', () => {
    const entries: ExportEntry[] = [{
      date: '2026-03-15',
      mealType: 'lunch',
      foodName: 'Rice, Beans, and Chicken',
      calories: 500,
      protein: 30,
      carbs: 60,
      fat: 12,
      fiber: 8,
      notes: null,
    }];
    const csv = generateCsv(entries, [], []);
    expect(csv).toContain('"Rice, Beans, and Chicken"');
  });

  it('handles null notes', () => {
    const csv = generateCsv(SAMPLE_ENTRIES, [], []);
    // Second entry has null notes — should be empty string in CSV
    const lines = csv.split('\n');
    const lunchLine = lines.find((l) => l.includes('Grilled Chicken'));
    expect(lunchLine).toBeTruthy();
    expect(lunchLine!.endsWith(',') || lunchLine!.endsWith('""')).toBe(true);
  });

  it('includes recipes section when provided', () => {
    const csv = generateCsv([], SAMPLE_RECIPES, []);
    expect(csv).toContain('Recipes');
    expect(csv).toContain('Chicken Fried Rice');
  });

  it('includes favourites section when provided', () => {
    const csv = generateCsv([], [], SAMPLE_FAVOURITES);
    expect(csv).toContain('Favourites');
    expect(csv).toContain('Morning Smoothie');
  });
});

// ---------------------------------------------------------------------------
// generateJson
// ---------------------------------------------------------------------------

describe('generateJson', () => {
  it('returns valid JSON string', () => {
    const json = generateJson(SAMPLE_ENTRIES, SAMPLE_RECIPES, SAMPLE_FAVOURITES);
    const parsed = JSON.parse(json);
    expect(parsed).toBeTruthy();
  });

  it('includes all sections', () => {
    const json = generateJson(SAMPLE_ENTRIES, SAMPLE_RECIPES, SAMPLE_FAVOURITES);
    const parsed = JSON.parse(json);
    expect(parsed.entries).toHaveLength(3);
    expect(parsed.recipes).toHaveLength(1);
    expect(parsed.favourites).toHaveLength(1);
  });

  it('includes metadata', () => {
    const json = generateJson(SAMPLE_ENTRIES, [], []);
    const parsed = JSON.parse(json);
    expect(parsed.exportedAt).toBeTruthy();
    expect(parsed.app).toBe('Tastimate');
    expect(parsed.version).toBeTruthy();
  });

  it('includes entry details', () => {
    const json = generateJson(SAMPLE_ENTRIES, [], []);
    const parsed = JSON.parse(json);
    expect(parsed.entries[0].date).toBe('2026-03-15');
    expect(parsed.entries[0].mealType).toBe('breakfast');
    expect(parsed.entries[0].calories).toBe(350);
  });

  it('returns empty arrays when no data', () => {
    const json = generateJson([], [], []);
    const parsed = JSON.parse(json);
    expect(parsed.entries).toEqual([]);
    expect(parsed.recipes).toEqual([]);
    expect(parsed.favourites).toEqual([]);
  });
});
