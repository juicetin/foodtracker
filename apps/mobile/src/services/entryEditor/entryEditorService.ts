/**
 * Entry editor service — CRUD operations on logged food entries.
 *
 * Provides ingredient-level editing: update weights/names, add/remove ingredients,
 * update dish names, and recalculate entry totals.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface IngredientUpdate {
  entryId: string;
  dishId: string | null;
  name: string;
  amountG: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// ---------------------------------------------------------------------------
// Ingredient operations
// ---------------------------------------------------------------------------

/**
 * Update an ingredient's weight (amount_g) and proportionally rescale nutrition.
 * Sets user_modified = 1.
 */
export function updateIngredientWeight(ingredientId: string, newAmountG: number): void {
  // Get current values to calculate scale factor
  const rows = opsqlite.executeSync(
    'SELECT id, original_amount_g, calories, protein, carbs, fat, fiber FROM ingredients WHERE id = ?',
    [ingredientId],
  ).rows as Array<Record<string, unknown>>;

  if (rows.length === 0) return;

  const row = rows[0];
  const originalG = (row.original_amount_g as number) || 0;

  if (originalG > 0) {
    const scale = newAmountG / originalG;
    const baseCal = (row.calories as number) || 0;
    const basePro = (row.protein as number) || 0;
    const baseCarb = (row.carbs as number) || 0;
    const baseFat = (row.fat as number) || 0;
    const baseFiber = (row.fiber as number) || 0;

    // Nutrition values in DB are stored at original_amount_g scale
    // We need to store them at the new scale: (base / oldScale) * newScale
    // But since base IS at originalG, just multiply by newAmountG/originalG
    opsqlite.executeSync(
      `UPDATE ingredients
       SET amount_g = ?, calories = ?, protein = ?, carbs = ?, fat = ?, fiber = ?,
           user_modified = 1, updated_at = datetime('now')
       WHERE id = ?`,
      [
        newAmountG,
        baseCal * scale,
        basePro * scale,
        baseCarb * scale,
        baseFat * scale,
        baseFiber * scale,
        ingredientId,
      ],
    );
  } else {
    // No original amount — just update the weight without scaling
    opsqlite.executeSync(
      `UPDATE ingredients SET amount_g = ?, user_modified = 1, updated_at = datetime('now') WHERE id = ?`,
      [newAmountG, ingredientId],
    );
  }
}

/**
 * Update an ingredient's name.
 */
export function updateIngredientName(ingredientId: string, newName: string): void {
  opsqlite.executeSync(
    `UPDATE ingredients SET name = ?, user_modified = 1, updated_at = datetime('now') WHERE id = ?`,
    [newName, ingredientId],
  );
}

/**
 * Remove an ingredient from the database.
 */
export function removeIngredient(ingredientId: string): void {
  opsqlite.executeSync('DELETE FROM ingredients WHERE id = ?', [ingredientId]);
}

/**
 * Add a new ingredient to an entry (optionally linked to a dish).
 * Returns the generated ingredient ID.
 */
export function addIngredient(data: IngredientUpdate): string {
  const id = generateId();
  const now = new Date().toISOString();

  opsqlite.executeSync(
    `INSERT INTO ingredients
      (id, entry_id, dish_id, name, quantity, unit, amount_g, original_amount_g,
       calories, protein, carbs, fat, fiber, user_modified, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, 'g', ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)`,
    [
      id,
      data.entryId,
      data.dishId,
      data.name,
      data.amountG, // quantity
      data.amountG, // amount_g
      data.amountG, // original_amount_g
      data.calories,
      data.protein,
      data.carbs,
      data.fat,
      data.fiber,
      now,
      now,
    ],
  );

  return id;
}

// ---------------------------------------------------------------------------
// Dish operations
// ---------------------------------------------------------------------------

/**
 * Update a dish's display name.
 */
export function updateDishName(dishId: string, newName: string): void {
  opsqlite.executeSync(
    `UPDATE scanned_dishes SET name = ? WHERE id = ?`,
    [newName, dishId],
  );
}

// ---------------------------------------------------------------------------
// Entry totals
// ---------------------------------------------------------------------------

/**
 * Recalculate and persist an entry's total macros from its ingredients.
 */
export function recalculateEntryTotals(entryId: string): void {
  const rows = opsqlite.executeSync(
    `SELECT
       COALESCE(SUM(calories), 0) as calories,
       COALESCE(SUM(protein), 0) as protein,
       COALESCE(SUM(carbs), 0) as carbs,
       COALESCE(SUM(fat), 0) as fat
     FROM ingredients WHERE entry_id = ?`,
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const totals = rows[0] ?? { calories: 0, protein: 0, carbs: 0, fat: 0 };

  opsqlite.executeSync(
    `UPDATE food_entries
     SET total_calories = ?, total_protein = ?, total_carbs = ?, total_fat = ?,
         updated_at = datetime('now')
     WHERE id = ?`,
    [
      (totals.calories as number) ?? 0,
      (totals.protein as number) ?? 0,
      (totals.carbs as number) ?? 0,
      (totals.fat as number) ?? 0,
      entryId,
    ],
  );
}
