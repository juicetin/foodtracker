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
  saveEntryAsRecipe,
  searchRecipes,
  updateRecipeWithVersioning,
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
    executeSync: (...args: unknown[]) => mockExecute(...args),
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

  it('sets source_recipe_id on the food entry', () => {
    // Mock loadRecipe
    mockExecute.mockReturnValueOnce({
      rows: [{ id: 'r1', name: 'Pasta', description: null, total_calories: 600, total_protein: 25, total_carbs: 80, total_fat: 15, times_used: 2, last_used_at: null, created_at: '2026-03-18', servings: 1, photo_uri: null }],
    });
    mockExecute.mockReturnValueOnce({
      rows: [
        { id: 'ri1', name: 'Spaghetti', quantity: 200, unit: 'g', calories: 300, protein: 10, carbs: 60, fat: 2 },
      ],
    });
    mockExecute.mockReturnValue(undefined);

    logRecipeAsEntry('r1', 'dinner');

    // The INSERT INTO food_entries should include source_recipe_id
    const insertCall = mockExecute.mock.calls.find(
      (c: unknown[]) => typeof c[0] === 'string' && (c[0] as string).includes('INSERT INTO food_entries'),
    );
    expect(insertCall).toBeDefined();
    expect(insertCall![0]).toContain('source_recipe_id');
    expect(insertCall![1]).toContain('r1');
  });

  it('scales nutrition by servingCount', () => {
    mockExecute.mockReturnValueOnce({
      rows: [{ id: 'r1', name: 'Pasta', description: null, total_calories: 600, total_protein: 25, total_carbs: 80, total_fat: 15, times_used: 0, last_used_at: null, created_at: '2026-03-18', servings: 1, photo_uri: null }],
    });
    mockExecute.mockReturnValueOnce({
      rows: [
        { id: 'ri1', name: 'Spaghetti', quantity: 200, unit: 'g', calories: 300, protein: 10, carbs: 60, fat: 2 },
      ],
    });
    mockExecute.mockReturnValue(undefined);

    logRecipeAsEntry('r1', 'lunch', 2);

    // food_entries total should be 2x
    const insertCall = mockExecute.mock.calls.find(
      (c: unknown[]) => typeof c[0] === 'string' && (c[0] as string).includes('INSERT INTO food_entries'),
    );
    expect(insertCall).toBeDefined();
    // total_calories should be 600*2 = 1200
    expect(insertCall![1]).toContain(1200);
  });
});

// ---------------------------------------------------------------------------
// saveEntryAsRecipe
// ---------------------------------------------------------------------------

describe('saveEntryAsRecipe', () => {
  it('creates a recipe from entry dishes and ingredients', () => {
    // Mock calls in order:
    // 1. scanned_dishes, 2. ingredients, 3. photos,
    // 4. INSERT custom_recipe, 5-6. INSERT recipe_ingredients (x2),
    // 7. INSERT recipe_photo, 8. recalculate SELECT, 9. recalculate UPDATE
    mockExecute
      .mockReturnValueOnce({ rows: [{ id: 'd1', name: 'Ramen', cuisine: 'Japanese' }] })
      .mockReturnValueOnce({ rows: [
        { id: 'i1', name: 'Noodles', quantity: 200, unit: 'g', calories: 300, protein: 8, carbs: 60, fat: 2 },
        { id: 'i2', name: 'Broth', quantity: 300, unit: 'ml', calories: 50, protein: 5, carbs: 3, fat: 1 },
      ] })
      .mockReturnValueOnce({ rows: [{ id: 'p1', uri: 'file://photo1.jpg', local_path: '/path/photo1.jpg' }] })
      .mockReturnValueOnce(undefined) // INSERT custom_recipe
      .mockReturnValueOnce(undefined) // INSERT recipe_ingredient 1
      .mockReturnValueOnce(undefined) // INSERT recipe_ingredient 2
      .mockReturnValueOnce(undefined) // INSERT recipe_photo
      .mockReturnValueOnce({ rows: [{ calories: 350, protein: 13, carbs: 63, fat: 3 }] }) // recalculate SELECT
      .mockReturnValueOnce(undefined); // recalculate UPDATE

    const recipeId = saveEntryAsRecipe('entry-1', 'My Ramen', 2);

    expect(recipeId).toBeTruthy();
    const recipeInsert = mockExecute.mock.calls.find(
      (c: unknown[]) => typeof c[0] === 'string' && (c[0] as string).includes('INSERT INTO custom_recipes'),
    );
    expect(recipeInsert).toBeDefined();
    expect(recipeInsert![1]).toContain('My Ramen');
    expect(recipeInsert![1]).toContain(2); // servings
  });

  it('copies ingredient nutrition faithfully', () => {
    mockExecute
      .mockReturnValueOnce({ rows: [{ id: 'd1', name: 'Bowl', cuisine: null }] })
      .mockReturnValueOnce({ rows: [{ id: 'i1', name: 'Rice', quantity: 150, unit: 'g', calories: 200, protein: 4, carbs: 45, fat: 0.5 }] })
      .mockReturnValueOnce({ rows: [] }) // no photos
      .mockReturnValueOnce(undefined) // INSERT custom_recipe
      .mockReturnValueOnce(undefined) // INSERT recipe_ingredient
      .mockReturnValueOnce({ rows: [{ calories: 200, protein: 4, carbs: 45, fat: 0.5 }] }) // recalculate SELECT
      .mockReturnValueOnce(undefined); // recalculate UPDATE

    saveEntryAsRecipe('entry-2', 'Rice Bowl');

    const ingredientInsert = mockExecute.mock.calls.find(
      (c: unknown[]) => typeof c[0] === 'string' && (c[0] as string).includes('INSERT INTO recipe_ingredients'),
    );
    expect(ingredientInsert).toBeDefined();
    expect(ingredientInsert![1]).toContain(200); // calories
    expect(ingredientInsert![1]).toContain(4);   // protein
  });
});

// ---------------------------------------------------------------------------
// searchRecipes
// ---------------------------------------------------------------------------

describe('searchRecipes', () => {
  it('returns matching recipes sorted by times_used DESC', () => {
    mockExecute.mockReturnValueOnce({
      rows: [
        { id: 'r1', name: 'Pasta Carbonara', description: null, total_calories: 700, total_protein: 30, total_carbs: 80, total_fat: 25, times_used: 5, last_used_at: '2026-03-19', created_at: '2026-03-10', servings: 1, photo_uri: null },
        { id: 'r2', name: 'Pasta Bolognese', description: null, total_calories: 650, total_protein: 35, total_carbs: 75, total_fat: 20, times_used: 2, last_used_at: '2026-03-18', created_at: '2026-03-11', servings: 2, photo_uri: null },
      ],
    });

    const results = searchRecipes('pasta');

    expect(results).toHaveLength(2);
    expect(results[0].name).toBe('Pasta Carbonara');
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('LIKE'),
      expect.arrayContaining(['%pasta%', 10]),
    );
  });

  it('returns empty array for empty query', () => {
    const results = searchRecipes('');
    expect(results).toEqual([]);
    expect(mockExecute).not.toHaveBeenCalled();
  });

  it('returns empty array for whitespace-only query', () => {
    const results = searchRecipes('   ');
    expect(results).toEqual([]);
    expect(mockExecute).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// updateRecipeWithVersioning
// ---------------------------------------------------------------------------

describe('updateRecipeWithVersioning', () => {
  const newIngredients: RecipeIngredientInput[] = [
    { recipeId: 'r1', name: 'Whole Wheat Pasta', quantity: 200, unit: 'g', calories: 280, protein: 12, carbs: 55, fat: 2 },
    { recipeId: 'r1', name: 'Pesto', quantity: 50, unit: 'g', calories: 150, protein: 3, carbs: 5, fat: 14 },
  ];

  it('save-as-new creates a new recipe, leaves original unchanged', () => {
    mockExecute
      // loadRecipe: recipe row
      .mockReturnValueOnce({ rows: [{ id: 'r1', name: 'Pasta', description: null, total_calories: 600, total_protein: 25, total_carbs: 80, total_fat: 15, times_used: 3, last_used_at: '2026-03-19', created_at: '2026-03-10', servings: 1, photo_uri: null }] })
      // loadRecipe: ingredients
      .mockReturnValueOnce({ rows: [] })
      // INSERT new custom_recipe
      .mockReturnValueOnce(undefined)
      // INSERT recipe_ingredient 1
      .mockReturnValueOnce(undefined)
      // INSERT recipe_ingredient 2
      .mockReturnValueOnce(undefined)
      // recalculate SELECT
      .mockReturnValueOnce({ rows: [{ calories: 430, protein: 15, carbs: 60, fat: 16 }] })
      // recalculate UPDATE
      .mockReturnValueOnce(undefined);

    const newId = updateRecipeWithVersioning('r1', newIngredients, 'save-as-new');

    expect(newId).not.toBe('r1');
    const recipeInsert = mockExecute.mock.calls.find(
      (c: unknown[]) => typeof c[0] === 'string' && (c[0] as string).includes('INSERT INTO custom_recipes'),
    );
    expect(recipeInsert).toBeDefined();
    expect(recipeInsert![1]).toContain('Pasta (edited)');
  });

  it('update-all deletes old ingredients, inserts new, and updates linked entries', () => {
    // DELETE old ingredients
    mockExecute.mockReturnValueOnce(undefined);
    // INSERT new ingredients (x2)
    mockExecute.mockReturnValueOnce(undefined);
    mockExecute.mockReturnValueOnce(undefined);
    // recalculate SELECT
    mockExecute.mockReturnValueOnce({ rows: [{ calories: 430, protein: 15, carbs: 60, fat: 16 }] });
    // recalculate UPDATE
    mockExecute.mockReturnValueOnce(undefined);
    // Find linked food_entries
    mockExecute.mockReturnValueOnce({
      rows: [{ id: 'fe1' }],
    });
    // For fe1: check user_modified count
    mockExecute.mockReturnValueOnce({ rows: [{ cnt: 0 }] });
    // Delete old ingredients for fe1
    mockExecute.mockReturnValueOnce(undefined);
    // Get dish for fe1
    mockExecute.mockReturnValueOnce({ rows: [{ id: 'dish1' }] });
    // INSERT new ingredients for fe1 (x2)
    mockExecute.mockReturnValueOnce(undefined);
    mockExecute.mockReturnValueOnce(undefined);
    // Recalculate entry totals
    mockExecute.mockReturnValueOnce({ rows: [{ calories: 430, protein: 15, carbs: 60, fat: 16 }] });
    mockExecute.mockReturnValueOnce(undefined);

    const resultId = updateRecipeWithVersioning('r1', newIngredients, 'update-all');

    expect(resultId).toBe('r1');
    // Should have called DELETE on recipe_ingredients
    const deleteCall = mockExecute.mock.calls.find(
      (c: unknown[]) => typeof c[0] === 'string' && (c[0] as string).includes('DELETE FROM recipe_ingredients'),
    );
    expect(deleteCall).toBeDefined();
  });
});
