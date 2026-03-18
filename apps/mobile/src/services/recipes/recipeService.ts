/**
 * Recipe service — CRUD for custom recipes and their ingredients.
 *
 * Recipes are stored in custom_recipes + recipe_ingredients tables.
 * Can be logged as food entries for one-tap meal re-logging.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RecipeInput {
  name: string;
  description?: string | null;
}

export interface RecipeIngredientInput {
  recipeId: string;
  name: string;
  quantity: number;
  unit: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface RecipeSummary {
  id: string;
  name: string;
  description: string | null;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  timesUsed: number;
  lastUsedAt: string | null;
  createdAt: string;
}

export interface RecipeIngredient {
  id: string;
  name: string;
  quantity: number;
  unit: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface RecipeDetail extends RecipeSummary {
  ingredients: RecipeIngredient[];
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

function recalculateRecipeTotals(recipeId: string): void {
  const rows = opsqlite.execute(
    `SELECT
       COALESCE(SUM(calories), 0) as calories,
       COALESCE(SUM(protein), 0) as protein,
       COALESCE(SUM(carbs), 0) as carbs,
       COALESCE(SUM(fat), 0) as fat
     FROM recipe_ingredients WHERE recipe_id = ?`,
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  const t = rows[0] ?? { calories: 0, protein: 0, carbs: 0, fat: 0 };

  opsqlite.execute(
    `UPDATE custom_recipes
     SET total_calories = ?, total_protein = ?, total_carbs = ?, total_fat = ?,
         updated_at = datetime('now')
     WHERE id = ?`,
    [
      (t.calories as number) ?? 0,
      (t.protein as number) ?? 0,
      (t.carbs as number) ?? 0,
      (t.fat as number) ?? 0,
      recipeId,
    ],
  );
}

function mapRecipeRow(r: Record<string, unknown>): RecipeSummary {
  return {
    id: r.id as string,
    name: r.name as string,
    description: (r.description as string) ?? null,
    totalCalories: (r.total_calories as number) ?? 0,
    totalProtein: (r.total_protein as number) ?? 0,
    totalCarbs: (r.total_carbs as number) ?? 0,
    totalFat: (r.total_fat as number) ?? 0,
    timesUsed: (r.times_used as number) ?? 0,
    lastUsedAt: (r.last_used_at as string) ?? null,
    createdAt: r.created_at as string,
  };
}

function mapIngredientRow(r: Record<string, unknown>): RecipeIngredient {
  return {
    id: r.id as string,
    name: r.name as string,
    quantity: (r.quantity as number) ?? 0,
    unit: (r.unit as string) ?? 'g',
    calories: (r.calories as number) ?? 0,
    protein: (r.protein as number) ?? 0,
    carbs: (r.carbs as number) ?? 0,
    fat: (r.fat as number) ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Recipe CRUD
// ---------------------------------------------------------------------------

/** Create a new recipe. Returns its ID. */
export function createRecipe(input: RecipeInput): string {
  const id = generateId();
  const now = new Date().toISOString();

  opsqlite.execute(
    `INSERT INTO custom_recipes (id, name, description, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?)`,
    [id, input.name, input.description ?? null, now, now],
  );

  return id;
}

/** Load all recipes (sorted by most recently used, then newest). */
export function loadRecipes(limit: number = 50): RecipeSummary[] {
  try {
    const rows = opsqlite.execute(
      `SELECT * FROM custom_recipes
       ORDER BY last_used_at DESC NULLS LAST, created_at DESC
       LIMIT ?`,
      [limit],
    ).rows as Array<Record<string, unknown>>;

    return rows.map(mapRecipeRow);
  } catch {
    return [];
  }
}

/** Load a single recipe with all its ingredients. */
export function loadRecipe(recipeId: string): RecipeDetail | null {
  const recipeRows = opsqlite.execute(
    'SELECT * FROM custom_recipes WHERE id = ?',
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  if (recipeRows.length === 0) return null;

  const ingRows = opsqlite.execute(
    'SELECT * FROM recipe_ingredients WHERE recipe_id = ? ORDER BY created_at',
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  return {
    ...mapRecipeRow(recipeRows[0]),
    ingredients: ingRows.map(mapIngredientRow),
  };
}

/** Delete a recipe (cascade deletes ingredients). */
export function deleteRecipe(recipeId: string): void {
  opsqlite.execute('DELETE FROM custom_recipes WHERE id = ?', [recipeId]);
}

// ---------------------------------------------------------------------------
// Recipe ingredient CRUD
// ---------------------------------------------------------------------------

/** Add an ingredient to a recipe. Returns ingredient ID. */
export function addRecipeIngredient(input: RecipeIngredientInput): string {
  const id = generateId();
  const now = new Date().toISOString();

  opsqlite.execute(
    `INSERT INTO recipe_ingredients (id, recipe_id, name, quantity, unit, calories, protein, carbs, fat, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [id, input.recipeId, input.name, input.quantity, input.unit, input.calories, input.protein, input.carbs, input.fat, now],
  );

  recalculateRecipeTotals(input.recipeId);
  return id;
}

/** Remove an ingredient from a recipe. */
export function removeRecipeIngredient(ingredientId: string, recipeId: string): void {
  opsqlite.execute('DELETE FROM recipe_ingredients WHERE id = ?', [ingredientId]);
  recalculateRecipeTotals(recipeId);
}

/** Update fields of a recipe ingredient. */
export function updateRecipeIngredient(
  ingredientId: string,
  recipeId: string,
  updates: Partial<Omit<RecipeIngredientInput, 'recipeId'>>,
): void {
  const sets: string[] = [];
  const values: unknown[] = [];

  if (updates.name !== undefined) { sets.push('name = ?'); values.push(updates.name); }
  if (updates.quantity !== undefined) { sets.push('quantity = ?'); values.push(updates.quantity); }
  if (updates.unit !== undefined) { sets.push('unit = ?'); values.push(updates.unit); }
  if (updates.calories !== undefined) { sets.push('calories = ?'); values.push(updates.calories); }
  if (updates.protein !== undefined) { sets.push('protein = ?'); values.push(updates.protein); }
  if (updates.carbs !== undefined) { sets.push('carbs = ?'); values.push(updates.carbs); }
  if (updates.fat !== undefined) { sets.push('fat = ?'); values.push(updates.fat); }

  if (sets.length === 0) return;

  values.push(ingredientId);
  opsqlite.execute(
    `UPDATE recipe_ingredients SET ${sets.join(', ')} WHERE id = ?`,
    values,
  );

  recalculateRecipeTotals(recipeId);
}

// ---------------------------------------------------------------------------
// Log recipe as food entry
// ---------------------------------------------------------------------------

/** Log a recipe as a new food entry with the given meal type. */
export function logRecipeAsEntry(recipeId: string, mealType: string): void {
  const recipe = loadRecipe(recipeId);
  if (!recipe) return;

  const entryId = generateId();
  const now = new Date().toISOString();
  const entryDate = new Date().toISOString().split('T')[0];

  // Create food entry
  opsqlite.execute(
    `INSERT INTO food_entries (id, meal_type, entry_date, total_calories, total_protein, total_carbs, total_fat, notes, created_at, updated_at, is_synced, is_deleted)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)`,
    [entryId, mealType, entryDate, recipe.totalCalories, recipe.totalProtein, recipe.totalCarbs, recipe.totalFat, `Recipe: ${recipe.name}`, now, now],
  );

  // Create scanned dish
  const dishId = generateId();
  opsqlite.execute(
    'INSERT INTO scanned_dishes (id, entry_id, name, portion_scale, created_at) VALUES (?, ?, ?, 1, ?)',
    [dishId, entryId, recipe.name, now],
  );

  // Copy ingredients
  for (const ing of recipe.ingredients) {
    opsqlite.execute(
      `INSERT INTO ingredients (id, entry_id, dish_id, name, quantity, unit, amount_g, original_amount_g, calories, protein, carbs, fat, user_modified, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
      [generateId(), entryId, dishId, ing.name, ing.quantity, ing.unit, ing.quantity, ing.quantity, ing.calories, ing.protein, ing.carbs, ing.fat, now, now],
    );
  }

  // Update usage stats
  opsqlite.execute(
    `UPDATE custom_recipes SET times_used = times_used + 1, last_used_at = datetime('now') WHERE id = ?`,
    [recipeId],
  );
}
