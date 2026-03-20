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
  servings: number;
  photoUri: string | null;
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
  const rows = opsqlite.executeSync(
    `SELECT
       COALESCE(SUM(calories), 0) as calories,
       COALESCE(SUM(protein), 0) as protein,
       COALESCE(SUM(carbs), 0) as carbs,
       COALESCE(SUM(fat), 0) as fat
     FROM recipe_ingredients WHERE recipe_id = ?`,
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  const t = rows[0] ?? { calories: 0, protein: 0, carbs: 0, fat: 0 };

  opsqlite.executeSync(
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
    servings: (r.servings as number) ?? 1,
    photoUri: (r.photo_uri as string) ?? null,
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

  opsqlite.executeSync(
    `INSERT INTO custom_recipes (id, name, description, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?)`,
    [id, input.name, input.description ?? null, now, now],
  );

  return id;
}

/** Load all recipes (sorted by most recently used, then newest). */
export function loadRecipes(limit: number = 50): RecipeSummary[] {
  try {
    const rows = opsqlite.executeSync(
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
  const recipeRows = opsqlite.executeSync(
    'SELECT * FROM custom_recipes WHERE id = ?',
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  if (recipeRows.length === 0) return null;

  const ingRows = opsqlite.executeSync(
    'SELECT * FROM recipe_ingredients WHERE recipe_id = ? ORDER BY created_at',
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  return {
    ...mapRecipeRow(recipeRows[0]),
    ingredients: ingRows.map(mapIngredientRow),
  };
}

/** Update a recipe's name. */
export function updateRecipeName(recipeId: string, name: string): void {
  opsqlite.executeSync(
    `UPDATE custom_recipes SET name = ?, updated_at = datetime('now') WHERE id = ?`,
    [name, recipeId],
  );
}

/** Delete a recipe (cascade deletes ingredients). */
export function deleteRecipe(recipeId: string): void {
  opsqlite.executeSync('DELETE FROM custom_recipes WHERE id = ?', [recipeId]);
}

// ---------------------------------------------------------------------------
// Recipe ingredient CRUD
// ---------------------------------------------------------------------------

/** Add an ingredient to a recipe. Returns ingredient ID. */
export function addRecipeIngredient(input: RecipeIngredientInput): string {
  const id = generateId();
  const now = new Date().toISOString();

  opsqlite.executeSync(
    `INSERT INTO recipe_ingredients (id, recipe_id, name, quantity, unit, calories, protein, carbs, fat, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [id, input.recipeId, input.name, input.quantity, input.unit, input.calories, input.protein, input.carbs, input.fat, now],
  );

  recalculateRecipeTotals(input.recipeId);
  return id;
}

/** Remove an ingredient from a recipe. */
export function removeRecipeIngredient(ingredientId: string, recipeId: string): void {
  opsqlite.executeSync('DELETE FROM recipe_ingredients WHERE id = ?', [ingredientId]);
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
  opsqlite.executeSync(
    `UPDATE recipe_ingredients SET ${sets.join(', ')} WHERE id = ?`,
    values,
  );

  recalculateRecipeTotals(recipeId);
}

// ---------------------------------------------------------------------------
// Log recipe as food entry
// ---------------------------------------------------------------------------

/** Log a recipe as a new food entry with the given meal type. */
export function logRecipeAsEntry(recipeId: string, mealType: string, servingCount: number = 1): void {
  const recipe = loadRecipe(recipeId);
  if (!recipe) return;

  const entryId = generateId();
  const now = new Date().toISOString();
  const entryDate = new Date().toISOString().split('T')[0];
  const scale = servingCount;

  // Create food entry with source_recipe_id linkage and scaled totals
  opsqlite.executeSync(
    `INSERT INTO food_entries (id, meal_type, entry_date, total_calories, total_protein, total_carbs, total_fat, notes, source_recipe_id, created_at, updated_at, is_synced, is_deleted)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)`,
    [entryId, mealType, entryDate, recipe.totalCalories * scale, recipe.totalProtein * scale, recipe.totalCarbs * scale, recipe.totalFat * scale, `Recipe: ${recipe.name}`, recipeId, now, now],
  );

  // Create scanned dish
  const dishId = generateId();
  opsqlite.executeSync(
    'INSERT INTO scanned_dishes (id, entry_id, name, portion_scale, created_at) VALUES (?, ?, ?, 1, ?)',
    [dishId, entryId, recipe.name, now],
  );

  // Copy ingredients (scaled by servingCount)
  for (const ing of recipe.ingredients) {
    opsqlite.executeSync(
      `INSERT INTO ingredients (id, entry_id, dish_id, name, quantity, unit, amount_g, original_amount_g, calories, protein, carbs, fat, user_modified, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
      [generateId(), entryId, dishId, ing.name, ing.quantity * scale, ing.unit, ing.quantity * scale, ing.quantity * scale, ing.calories * scale, ing.protein * scale, ing.carbs * scale, ing.fat * scale, now, now],
    );
  }

  // Update usage stats
  opsqlite.executeSync(
    `UPDATE custom_recipes SET times_used = times_used + 1, last_used_at = datetime('now') WHERE id = ?`,
    [recipeId],
  );
}

// ---------------------------------------------------------------------------
// Save entry as recipe
// ---------------------------------------------------------------------------

/** Create a recipe from an existing food entry, copying all dishes and ingredients. */
export function saveEntryAsRecipe(entryId: string, recipeName: string, servings: number = 1): string {
  const recipeId = generateId();
  const now = new Date().toISOString();

  // Fetch dishes, ingredients, and photos from the entry
  const dishes = opsqlite.executeSync(
    'SELECT * FROM scanned_dishes WHERE entry_id = ?',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const ingredients = opsqlite.executeSync(
    'SELECT * FROM ingredients WHERE entry_id = ?',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const photos = opsqlite.executeSync(
    'SELECT * FROM photos WHERE entry_id = ?',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  // Determine photo_uri from first photo
  const photoUri = photos.length > 0
    ? ((photos[0].local_path as string) || (photos[0].uri as string) || null)
    : null;

  // Create recipe
  opsqlite.executeSync(
    `INSERT INTO custom_recipes (id, name, source_entry_id, servings, photo_uri, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [recipeId, recipeName, entryId, servings, photoUri, now, now],
  );

  // Copy ingredients into recipe_ingredients
  for (const ing of ingredients) {
    opsqlite.executeSync(
      `INSERT INTO recipe_ingredients (id, recipe_id, name, quantity, unit, calories, protein, carbs, fat, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        generateId(), recipeId,
        ing.name as string,
        (ing.quantity as number) ?? 0,
        (ing.unit as string) ?? 'g',
        (ing.calories as number) ?? 0,
        (ing.protein as number) ?? 0,
        (ing.carbs as number) ?? 0,
        (ing.fat as number) ?? 0,
        now,
      ],
    );
  }

  // Copy photos into recipe_photos
  for (let i = 0; i < photos.length; i++) {
    const photo = photos[i];
    const localPath = (photo.local_path as string) || (photo.uri as string);
    opsqlite.executeSync(
      `INSERT INTO recipe_photos (id, recipe_id, local_path, is_primary, created_at)
       VALUES (?, ?, ?, ?, ?)`,
      [generateId(), recipeId, localPath, i === 0 ? 1 : 0, now],
    );
  }

  recalculateRecipeTotals(recipeId);
  return recipeId;
}

// ---------------------------------------------------------------------------
// Search recipes
// ---------------------------------------------------------------------------

/** Search recipes by name, sorted by usage frequency. */
export function searchRecipes(query: string, limit: number = 10): RecipeSummary[] {
  if (!query || !query.trim()) return [];

  const rows = opsqlite.executeSync(
    `SELECT * FROM custom_recipes
     WHERE name LIKE ? COLLATE NOCASE
     ORDER BY times_used DESC, last_used_at DESC NULLS LAST
     LIMIT ?`,
    [`%${query.trim()}%`, limit],
  ).rows as Array<Record<string, unknown>>;

  return rows.map(mapRecipeRow);
}

// ---------------------------------------------------------------------------
// Recipe versioning
// ---------------------------------------------------------------------------

/**
 * Update a recipe with versioning support.
 * - 'update-all': replaces ingredients in-place and cascades to linked food_entries
 * - 'save-as-new': forks as a new recipe, original stays unchanged
 * Returns the recipe ID (original for update-all, new for save-as-new).
 */
export function updateRecipeWithVersioning(
  recipeId: string,
  ingredients: RecipeIngredientInput[],
  mode: 'update-all' | 'save-as-new',
): string {
  if (mode === 'save-as-new') {
    // Load original recipe for name
    const original = loadRecipe(recipeId);
    const newName = original ? `${original.name} (edited)` : 'Recipe (edited)';
    const newId = generateId();
    const now = new Date().toISOString();

    opsqlite.executeSync(
      `INSERT INTO custom_recipes (id, name, created_at, updated_at)
       VALUES (?, ?, ?, ?)`,
      [newId, newName, now, now],
    );

    for (const ing of ingredients) {
      opsqlite.executeSync(
        `INSERT INTO recipe_ingredients (id, recipe_id, name, quantity, unit, calories, protein, carbs, fat, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [generateId(), newId, ing.name, ing.quantity, ing.unit, ing.calories, ing.protein, ing.carbs, ing.fat, now],
      );
    }

    recalculateRecipeTotals(newId);
    return newId;
  }

  // mode === 'update-all'
  // 1. Replace recipe ingredients
  opsqlite.executeSync('DELETE FROM recipe_ingredients WHERE recipe_id = ?', [recipeId]);
  const now = new Date().toISOString();
  for (const ing of ingredients) {
    opsqlite.executeSync(
      `INSERT INTO recipe_ingredients (id, recipe_id, name, quantity, unit, calories, protein, carbs, fat, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [generateId(), recipeId, ing.name, ing.quantity, ing.unit, ing.calories, ing.protein, ing.carbs, ing.fat, now],
    );
  }
  recalculateRecipeTotals(recipeId);

  // 2. Cascade to linked food_entries
  const linkedEntries = opsqlite.executeSync(
    'SELECT id FROM food_entries WHERE source_recipe_id = ?',
    [recipeId],
  ).rows as Array<Record<string, unknown>>;

  for (const entry of linkedEntries) {
    const entryId = entry.id as string;

    // Only update entries where user has NOT manually modified any ingredient
    const modCount = opsqlite.executeSync(
      'SELECT COUNT(*) as cnt FROM ingredients WHERE entry_id = ? AND user_modified = 1',
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    if ((modCount[0]?.cnt as number) > 0) continue;

    // Delete old entry ingredients
    opsqlite.executeSync('DELETE FROM ingredients WHERE entry_id = ?', [entryId]);

    // Get dish ID for this entry
    const dishRows = opsqlite.executeSync(
      'SELECT id FROM scanned_dishes WHERE entry_id = ? LIMIT 1',
      [entryId],
    ).rows as Array<Record<string, unknown>>;
    const dishId = dishRows.length > 0 ? (dishRows[0].id as string) : null;

    // Insert new ingredients
    for (const ing of ingredients) {
      opsqlite.executeSync(
        `INSERT INTO ingredients (id, entry_id, dish_id, name, quantity, unit, amount_g, original_amount_g, calories, protein, carbs, fat, user_modified, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
        [generateId(), entryId, dishId, ing.name, ing.quantity, ing.unit, ing.quantity, ing.quantity, ing.calories, ing.protein, ing.carbs, ing.fat, now, now],
      );
    }

    // Recalculate entry totals
    const totals = opsqlite.executeSync(
      `SELECT COALESCE(SUM(calories), 0) as calories, COALESCE(SUM(protein), 0) as protein,
              COALESCE(SUM(carbs), 0) as carbs, COALESCE(SUM(fat), 0) as fat
       FROM ingredients WHERE entry_id = ?`,
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    const t = totals[0] ?? { calories: 0, protein: 0, carbs: 0, fat: 0 };
    opsqlite.executeSync(
      `UPDATE food_entries SET total_calories = ?, total_protein = ?, total_carbs = ?, total_fat = ?, updated_at = datetime('now') WHERE id = ?`,
      [(t.calories as number) ?? 0, (t.protein as number) ?? 0, (t.carbs as number) ?? 0, (t.fat as number) ?? 0, entryId],
    );
  }

  return recipeId;
}
