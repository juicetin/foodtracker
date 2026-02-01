import 'dotenv/config';
import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import request from 'supertest';
import express from 'express';
import cors from 'cors';
import pool from '../../db/client.js';
import foodLogsRouter from '../../routes/foodLogs.js';
import recipesRouter from '../../routes/recipes.js';

// Create test app
const app = express();
app.use(cors());
app.use(express.json());
app.use('/api/food-logs', foodLogsRouter);
app.use('/api/recipes', recipesRouter);

const testUserId = '00000000-0000-0000-0000-000000000004';

beforeAll(async () => {
  await pool.query(`
    INSERT INTO users (id, email, name, region)
    VALUES ($1, 'recipes-e2e@test.com', 'Recipes E2E User', 'AU')
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

describe('Recipes API (E2E)', () => {
  describe('POST /api/recipes', () => {
    it('should create a recipe from a food entry', async () => {
      // First create a food entry
      const entryResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        })
        .expect(200)
        .expect('Content-Type', /json/);

      const entryId = entryResponse.body.entry.id;

      // Create recipe from it
      const response = await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entryId,
          name: 'My Favorite Lunch',
          description: 'Healthy and delicious',
        })
        .expect(200);

      expect(response.body.id).toBeDefined();
      expect(response.body.name).toBe('My Favorite Lunch');
      expect(response.body.description).toBe('Healthy and delicious');
      expect(response.body.userId).toBe(testUserId);
      expect(response.body.sourceEntryId).toBe(entryId);
      expect(response.body.ingredients).toBeDefined();
      expect(response.body.ingredients.length).toBeGreaterThan(0);
    });

    it('should return 400 if required fields are missing', async () => {
      const response = await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          name: 'Test Recipe',
        })
        .expect(400);

      expect(response.body.error).toBe('userId, entryId, and name are required');
    });
  });

  describe('GET /api/recipes', () => {
    it('should get all recipes for a user', async () => {
      // Create an entry and recipe
      const entryResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'dinner',
        });

      const entryId = entryResponse.body.entry.id;

      await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entryId,
          name: 'Dinner Recipe',
        });

      // Get recipes
      const response = await request(app)
        .get('/api/recipes')
        .query({ userId: testUserId })
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBeGreaterThanOrEqual(1);
      expect(response.body[0].name).toBe('Dinner Recipe');
    });

    it('should return 400 if userId is missing', async () => {
      const response = await request(app)
        .get('/api/recipes')
        .expect(400);

      expect(response.body.error).toBe('User ID is required');
    });
  });

  describe('GET /api/recipes/search', () => {
    it('should search recipes by name', async () => {
      // Create entries and recipes
      const entry1 = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        });

      await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entry1.body.entry.id,
          name: 'Chicken Salad',
        });

      await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entry1.body.entry.id,
          name: 'Beef Stew',
        });

      // Search for chicken
      const response = await request(app)
        .get('/api/recipes/search')
        .query({ userId: testUserId, q: 'chicken' })
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(1);
      expect(response.body[0].name).toBe('Chicken Salad');
    });
  });

  describe('POST /api/recipes/:id/use', () => {
    it('should create a new entry from a recipe', async () => {
      // Create entry and recipe
      const entryResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        });

      const recipeResponse = await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entryResponse.body.entry.id,
          name: 'Quick Lunch',
        });

      const recipeId = recipeResponse.body.id;

      // Use the recipe
      const response = await request(app)
        .post(`/api/recipes/${recipeId}/use`)
        .send({
          userId: testUserId,
          mealType: 'dinner',
        })
        .expect(200);

      expect(response.body.id).toBeDefined();
      expect(response.body.mealType).toBe('dinner');
      expect(response.body.ingredients).toBeDefined();
      expect(response.body.notes).toContain('From recipe: Quick Lunch');

      // Verify recipe usage was tracked
      const recipesResponse = await request(app)
        .get('/api/recipes')
        .query({ userId: testUserId })
        .expect(200);

      const usedRecipe = recipesResponse.body.find((r: any) => r.id === recipeId);
      expect(usedRecipe.timesUsed).toBe(1);
      expect(usedRecipe.lastUsedAt).toBeDefined();
    });
  });

  describe('DELETE /api/recipes/:id', () => {
    it('should delete a recipe', async () => {
      // Create entry and recipe
      const entryResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'snack',
        });

      const recipeResponse = await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entryResponse.body.entry.id,
          name: 'Snack Recipe',
        });

      const recipeId = recipeResponse.body.id;

      // Delete it
      await request(app)
        .delete(`/api/recipes/${recipeId}`)
        .expect(200);

      // Verify it's gone
      const recipesResponse = await request(app)
        .get('/api/recipes')
        .query({ userId: testUserId })
        .expect(200);

      expect(recipesResponse.body.find((r: any) => r.id === recipeId)).toBeUndefined();
    });
  });

  describe('Complete recipe workflow', () => {
    it('should handle full recipe lifecycle', async () => {
      // 1. Create a food entry from photos
      const entryResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [{ id: '1', uri: 'file:///meal.jpg', timestamp: new Date().toISOString() }],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        })
        .expect(200);

      const entryId = entryResponse.body.entry.id;
      const originalCalories = entryResponse.body.entry.totalCalories;

      // 2. Save it as a recipe
      const recipeResponse = await request(app)
        .post('/api/recipes')
        .send({
          userId: testUserId,
          entryId: entryId,
          name: 'Chicken & Rice Bowl',
          description: 'My go-to healthy lunch',
        })
        .expect(200);

      const recipeId = recipeResponse.body.id;
      expect(recipeResponse.body.totalCalories).toBe(originalCalories);

      // 3. Search for it
      const searchResponse = await request(app)
        .get('/api/recipes/search')
        .query({ userId: testUserId, q: 'chicken' })
        .expect(200);

      expect(searchResponse.body.length).toBeGreaterThanOrEqual(1);

      // 4. Use it to create a new entry
      const useResponse = await request(app)
        .post(`/api/recipes/${recipeId}/use`)
        .send({
          userId: testUserId,
          mealType: 'dinner',
        })
        .expect(200);

      expect(useResponse.body.totalCalories).toBe(originalCalories);

      // 5. Use it again (test usage tracking)
      await request(app)
        .post(`/api/recipes/${recipeId}/use`)
        .send({
          userId: testUserId,
          mealType: 'lunch',
        })
        .expect(200);

      // 6. Verify usage count
      const recipesResponse = await request(app)
        .get('/api/recipes')
        .query({ userId: testUserId })
        .expect(200);

      const recipe = recipesResponse.body.find((r: any) => r.id === recipeId);
      expect(recipe.timesUsed).toBe(2);
      expect(recipe.lastUsedAt).toBeDefined();

      // 7. List all food entries (should have 3: original + 2 from recipe)
      const logsResponse = await request(app)
        .get('/api/food-logs')
        .query({ userId: testUserId })
        .expect(200);

      expect(logsResponse.body.length).toBe(3);

      // 8. Delete the recipe
      await request(app)
        .delete(`/api/recipes/${recipeId}`)
        .expect(200);

      // 9. Verify deletion
      const finalRecipesResponse = await request(app)
        .get('/api/recipes')
        .query({ userId: testUserId })
        .expect(200);

      expect(finalRecipesResponse.body.find((r: any) => r.id === recipeId)).toBeUndefined();
    });
  });
});
