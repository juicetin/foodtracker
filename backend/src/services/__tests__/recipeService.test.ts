import 'dotenv/config';
import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import pool from '../../db/client.js';
import * as recipeService from '../recipeService.js';
import * as foodEntryService from '../foodEntryService.js';

const testUserId = '00000000-0000-0000-0000-000000000002';

beforeAll(async () => {
  await pool.query(`
    INSERT INTO users (id, email, name, region)
    VALUES ($1, 'recipe@test.com', 'Recipe User', 'AU')
    ON CONFLICT (email) DO UPDATE SET id = $1
  `, [testUserId]);
});

afterAll(async () => {
  await pool.query('DELETE FROM custom_recipes WHERE user_id = $1', [testUserId]);
  await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
  await pool.end();
});

beforeEach(async () => {
  await pool.query('DELETE FROM custom_recipes WHERE user_id = $1', [testUserId]);
  await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
});

describe('Recipe Service', () => {
  describe('createRecipeFromEntry', () => {
    it('should create a recipe from a food entry', async () => {
      // Create a food entry first
      const entry = await foodEntryService.createFoodEntry({
        userId: testUserId,
        mealType: 'lunch',
        entryDate: new Date(),
        photos: [{ uri: 'file:///meal.jpg', width: 1920, height: 1080 }],
        ingredients: [
          { name: 'Chicken', quantity: 150, unit: 'g', calories: 165, protein: 31, carbs: 0, fat: 3.6, databaseSource: 'AFCD' },
          { name: 'Rice', quantity: 100, unit: 'g', calories: 130, protein: 2.7, carbs: 28, fat: 0.3, databaseSource: 'AFCD' },
        ],
      });

      // Create recipe from the entry
      const recipe = await recipeService.createRecipeFromEntry(
        testUserId,
        entry.id,
        'Chicken & Rice Bowl',
        'My favorite lunch meal'
      );

      expect(recipe).toBeDefined();
      expect(recipe.id).toBeDefined();
      expect(recipe.name).toBe('Chicken & Rice Bowl');
      expect(recipe.userId).toBe(testUserId);
      expect(recipe.sourceEntryId).toBe(entry.id);
      expect(recipe.totalCalories).toBe(295); // 165 + 130
      expect(recipe.ingredients.length).toBe(2);
    });

    it('should copy photos from the entry', async () => {
      const entry = await foodEntryService.createFoodEntry({
        userId: testUserId,
        mealType: 'dinner',
        entryDate: new Date(),
        photos: [
          { uri: 'file:///photo1.jpg', gcsUrl: 'https://storage/photo1.jpg', width: 1920, height: 1080 },
        ],
        ingredients: [
          { name: 'Salad', quantity: 200, unit: 'g', calories: 50, protein: 2, carbs: 10, fat: 0.5, databaseSource: 'AFCD' },
        ],
      });

      const recipe = await recipeService.createRecipeFromEntry(
        testUserId,
        entry.id,
        'Garden Salad'
      );

      expect(recipe.photos).toBeDefined();
      expect(recipe.photos!.length).toBe(1);
      expect(recipe.photos![0].gcsUrl).toBe('https://storage/photo1.jpg');
    });
  });

  describe('getRecipes', () => {
    it('should get all recipes for a user', async () => {
      const entry = await foodEntryService.createFoodEntry({
        userId: testUserId,
        mealType: 'lunch',
        entryDate: new Date(),
        photos: [],
        ingredients: [
          { name: 'Pasta', quantity: 100, unit: 'g', calories: 150, protein: 5, carbs: 30, fat: 1, databaseSource: 'AFCD' },
        ],
      });

      await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Simple Pasta');
      await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Quick Lunch');

      const recipes = await recipeService.getRecipes(testUserId);
      expect(recipes.length).toBeGreaterThanOrEqual(2);
    });

    it('should order recipes by most recently used', async () => {
      const entry = await foodEntryService.createFoodEntry({
        userId: testUserId,
        mealType: 'lunch',
        entryDate: new Date(),
        photos: [],
        ingredients: [{ name: 'Test', quantity: 100, unit: 'g', calories: 100, protein: 5, carbs: 10, fat: 1, databaseSource: 'AFCD' }],
      });

      const recipe1 = await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Recipe 1');
      const recipe2 = await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Recipe 2');

      // Use recipe1
      await recipeService.useRecipe(recipe1.id, testUserId);

      const recipes = await recipeService.getRecipes(testUserId);
      expect(recipes[0].name).toBe('Recipe 1'); // Should be first (most recently used)
      expect(recipes[0].timesUsed).toBe(1);
    });
  });

  describe('useRecipe', () => {
    it('should create a new entry from a recipe', async () => {
      const entry = await foodEntryService.createFoodEntry({
        userId: testUserId,
        mealType: 'lunch',
        entryDate: new Date(),
        photos: [],
        ingredients: [
          { name: 'Tuna', quantity: 120, unit: 'g', calories: 140, protein: 30, carbs: 0, fat: 1, databaseSource: 'AFCD' },
          { name: 'Avocado', quantity: 50, unit: 'g', calories: 80, protein: 1, carbs: 4, fat: 7.5, databaseSource: 'AFCD' },
        ],
      });

      const recipe = await recipeService.createRecipeFromEntry(
        testUserId,
        entry.id,
        'Tuna Avocado Bowl'
      );

      const newEntry = await recipeService.useRecipe(recipe.id, testUserId, {
        mealType: 'dinner',
        entryDate: new Date(),
      });

      expect(newEntry).toBeDefined();
      expect(newEntry.mealType).toBe('dinner');
      expect(newEntry.totalCalories).toBe(220); // 140 + 80
      expect(newEntry.ingredients!.length).toBe(2);

      // Check that usage was tracked
      const updatedRecipe = await recipeService.getRecipe(recipe.id);
      expect(updatedRecipe?.timesUsed).toBe(1);
      expect(updatedRecipe?.lastUsedAt).toBeDefined();
    });
  });

  describe('searchRecipes', () => {
    it('should search recipes by name', async () => {
      const entry = await foodEntryService.createFoodEntry({
        userId: testUserId,
        mealType: 'lunch',
        entryDate: new Date(),
        photos: [],
        ingredients: [{ name: 'Test', quantity: 100, unit: 'g', calories: 100, protein: 5, carbs: 10, fat: 1, databaseSource: 'AFCD' }],
      });

      await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Chicken Salad');
      await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Beef Stew');
      await recipeService.createRecipeFromEntry(testUserId, entry.id, 'Chicken Curry');

      const results = await recipeService.searchRecipes(testUserId, 'chicken');
      expect(results.length).toBe(2);
      expect(results.every(r => r.name.toLowerCase().includes('chicken'))).toBe(true);
    });
  });
});
