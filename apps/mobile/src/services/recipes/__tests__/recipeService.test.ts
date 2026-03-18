/**
 * Recipe service tests — CRUD for custom recipes and recipe ingredients.
 */

import {
  createRecipe,
  loadRecipes,
  loadRecipe,
  addRecipeIngredient,
  removeRecipeIngredient,
  updateRecipeIngredient,
  deleteRecipe,
  logRecipeAsEntry,
  type RecipeInput,
  type RecipeIngredientInput,
} from '../recipeService';

// ---------------------------------------------------------------------------
// Mock opsqlite
// ---------------------------------------------------------------------------

const mockExecute = jest.fn();
const mockInsert = jest.fn().mockReturnValue({ values: jest.fn().mockReturnValue(Promise.resolve()) });

jest.mock('../../../../db/client', () => ({
  opsqlite: {
    execute: (...args: unknown[]) => mockExecute(...args),
  },
  userDb: {
    insert: () => mockInsert(),
  },
}));

beforeEach(() => {
  mockExecute.mockReset();
  mockInsert.mockClear();
});

// ---------------------------------------------------------------------------
// createRecipe
// ---------------------------------------------------------------------------

describe('createRecipe', () => {
  it('inserts a recipe row and returns the generated ID', () => {
    const input: RecipeInput = {
      name: 'Chicken Fried Rice',
      description: 'Quick weeknight dinner',
    };

    const id = createRecipe(input);

    expect(id).toBeTruthy();
    expect(id.length).toBeGreaterThan(10); // UUID
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO custom_recipes'),
      expect.arrayContaining([id, 'Chicken Fried Rice', 'Quick weeknight dinner']),
    );
  });

  it('handles null description', () => {
    const id = createRecipe({ name: 'Simple Salad' });
    expect(id).toBeTruthy();
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO custom_recipes'),
      expect.arrayContaining([id, 'Simple Salad', null]),
    );
  });
});

// ---------------------------------------------------------------------------
// loadRecipes
// ---------------------------------------------------------------------------

describe('loadRecipes', () => {
  it('returns list of recipes sorted by last used', () => {
    mockExecute.mockReturnValueOnce({
      rows: [
        { id: 'r1', name: 'Recipe A', description: null, total_calories: 500, total_protein: 30, total_carbs: 50, total_fat: 15, times_used: 3, last_used_at: '2026-03-19', created_at: '2026-03-18' },
        { id: 'r2', name: 'Recipe B', description: 'Desc', total_calories: 300, total_protein: 20, total_carbs: 30, total_fat: 10, times_used: 1, last_used_at: null, created_at: '2026-03-17' },
      ],
    });

    const recipes = loadRecipes();

    expect(recipes).toHaveLength(2);
    expect(recipes[0].id).toBe('r1');
    expect(recipes[0].name).toBe('Recipe A');
    expect(recipes[0].totalCalories).toBe(500);
  });

  it('returns empty array on error', () => {
    mockExecute.mockImplementationOnce(() => { throw new Error('DB error'); });
    expect(loadRecipes()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// loadRecipe (single recipe with ingredients)
// ---------------------------------------------------------------------------

describe('loadRecipe', () => {
  it('returns recipe with ingredients', () => {
    // First call: recipe row
    mockExecute.mockReturnValueOnce({
      rows: [{ id: 'r1', name: 'Pasta', description: null, total_calories: 600, total_protein: 25, total_carbs: 80, total_fat: 15, times_used: 0, last_used_at: null, created_at: '2026-03-18' }],
    });
    // Second call: ingredients
    mockExecute.mockReturnValueOnce({
      rows: [
        { id: 'ri1', name: 'Spaghetti', quantity: 200, unit: 'g', calories: 300, protein: 10, carbs: 60, fat: 2 },
        { id: 'ri2', name: 'Tomato Sauce', quantity: 100, unit: 'g', calories: 50, protein: 2, carbs: 10, fat: 1 },
      ],
    });

    const recipe = loadRecipe('r1');

    expect(recipe).not.toBeNull();
    expect(recipe!.name).toBe('Pasta');
    expect(recipe!.ingredients).toHaveLength(2);
    expect(recipe!.ingredients[0].name).toBe('Spaghetti');
    expect(recipe!.ingredients[1].calories).toBe(50);
  });

  it('returns null for non-existent recipe', () => {
    mockExecute.mockReturnValueOnce({ rows: [] });
    expect(loadRecipe('nonexistent')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// addRecipeIngredient
// ---------------------------------------------------------------------------

describe('addRecipeIngredient', () => {
  it('inserts ingredient and updates recipe totals', () => {
    const input: RecipeIngredientInput = {
      recipeId: 'r1',
      name: 'Olive Oil',
      quantity: 15,
      unit: 'g',
      calories: 120,
      protein: 0,
      carbs: 0,
      fat: 14,
    };

    // Mock for recalculate totals SELECT
    mockExecute.mockReturnValueOnce(undefined); // INSERT
    mockExecute.mockReturnValueOnce({
      rows: [{ calories: 420, protein: 30, carbs: 50, fat: 29 }],
    });
    mockExecute.mockReturnValueOnce(undefined); // UPDATE totals

    const id = addRecipeIngredient(input);

    expect(id).toBeTruthy();
    // Should have called INSERT, SELECT SUM, UPDATE
    expect(mockExecute).toHaveBeenCalledTimes(3);
  });
});

// ---------------------------------------------------------------------------
// removeRecipeIngredient
// ---------------------------------------------------------------------------

describe('removeRecipeIngredient', () => {
  it('deletes ingredient and recalculates totals', () => {
    // Mock for recalculate SELECT
    mockExecute.mockReturnValueOnce(undefined); // DELETE
    mockExecute.mockReturnValueOnce({
      rows: [{ calories: 300, protein: 20, carbs: 40, fat: 10 }],
    });
    mockExecute.mockReturnValueOnce(undefined); // UPDATE totals

    removeRecipeIngredient('ri1', 'r1');

    expect(mockExecute).toHaveBeenCalledTimes(3);
    expect(mockExecute.mock.calls[0][0]).toContain('DELETE FROM recipe_ingredients');
  });
});

// ---------------------------------------------------------------------------
// updateRecipeIngredient
// ---------------------------------------------------------------------------

describe('updateRecipeIngredient', () => {
  it('updates ingredient fields', () => {
    mockExecute.mockReturnValueOnce(undefined); // UPDATE
    mockExecute.mockReturnValueOnce({
      rows: [{ calories: 500, protein: 30, carbs: 60, fat: 15 }],
    });
    mockExecute.mockReturnValueOnce(undefined); // UPDATE totals

    updateRecipeIngredient('ri1', 'r1', {
      name: 'Brown Rice',
      quantity: 200,
      calories: 250,
      protein: 5,
      carbs: 50,
      fat: 2,
    });

    expect(mockExecute.mock.calls[0][0]).toContain('UPDATE recipe_ingredients');
  });
});

// ---------------------------------------------------------------------------
// deleteRecipe
// ---------------------------------------------------------------------------

describe('deleteRecipe', () => {
  it('deletes recipe (cascade deletes ingredients)', () => {
    deleteRecipe('r1');

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('DELETE FROM custom_recipes'),
      ['r1'],
    );
  });
});

// ---------------------------------------------------------------------------
// logRecipeAsEntry
// ---------------------------------------------------------------------------

describe('logRecipeAsEntry', () => {
  it('creates a food entry from a recipe', () => {
    // Mock loadRecipe
    mockExecute.mockReturnValueOnce({
      rows: [{ id: 'r1', name: 'Pasta', description: null, total_calories: 600, total_protein: 25, total_carbs: 80, total_fat: 15, times_used: 2, last_used_at: null, created_at: '2026-03-18' }],
    });
    mockExecute.mockReturnValueOnce({
      rows: [
        { id: 'ri1', name: 'Spaghetti', quantity: 200, unit: 'g', calories: 300, protein: 10, carbs: 60, fat: 2 },
      ],
    });
    // Mock: INSERT food_entry, INSERT scanned_dish, INSERT ingredient, UPDATE times_used
    mockExecute.mockReturnValue(undefined);

    logRecipeAsEntry('r1', 'dinner');

    // Should insert food_entry, dish, ingredient(s), and update usage
    expect(mockExecute.mock.calls.length).toBeGreaterThanOrEqual(4);
  });
});
