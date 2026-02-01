import 'dotenv/config';
import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import request from 'supertest';
import express from 'express';
import cors from 'cors';
import pool from '../../db/client.js';
import foodLogsRouter from '../../routes/foodLogs.js';

// Create test app
const app = express();
app.use(cors());
app.use(express.json());
app.use('/api/food-logs', foodLogsRouter);

const testUserId = '00000000-0000-0000-0000-000000000003';

beforeAll(async () => {
  await pool.query(`
    INSERT INTO users (id, email, name, region)
    VALUES ($1, 'e2e@test.com', 'E2E User', 'AU')
    ON CONFLICT (email) DO UPDATE SET id = $1
  `, [testUserId]);
});

afterAll(async () => {
  await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
  await pool.end();
});

beforeEach(async () => {
  await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
});

describe('Food Logs API (E2E)', () => {
  describe('POST /api/food-logs/process', () => {
    it('should process photos and create a food entry', async () => {
      const response = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [
            { id: '1', uri: 'file:///photo1.jpg', timestamp: new Date().toISOString() },
          ],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        })
        .expect(200);

      expect(response.body.entry).toBeDefined();
      expect(response.body.entry.id).toBeDefined();
      expect(response.body.entry.userId).toBe(testUserId);
      expect(response.body.entry.mealType).toBe('lunch');
      expect(response.body.entry.totalCalories).toBeGreaterThan(0);
      expect(response.body.entry.ingredients).toBeDefined();
      expect(response.body.entry.ingredients.length).toBeGreaterThan(0);
      expect(response.body.processingTime).toBeDefined();
    });

    it('should return 400 if photos are missing', async () => {
      const response = await request(app)
        .post('/api/food-logs/process')
        .send({
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        })
        .expect(400);

      expect(response.body.error).toBe('Photos are required');
    });

    it('should return 400 if userId is missing', async () => {
      const response = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [{ id: '1', uri: 'file:///photo.jpg', timestamp: new Date().toISOString() }],
          userRegion: 'AU',
          mealType: 'lunch',
        })
        .expect(400);

      expect(response.body.error).toBe('User ID is required');
    });
  });

  describe('GET /api/food-logs', () => {
    it('should get all food logs for a user', async () => {
      // Create some entries first
      await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'breakfast',
        });

      await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        });

      const response = await request(app)
        .get('/api/food-logs')
        .query({ userId: testUserId })
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBeGreaterThanOrEqual(2);
    });

    it('should return 400 if userId is missing', async () => {
      const response = await request(app)
        .get('/api/food-logs')
        .expect(400);

      expect(response.body.error).toBe('User ID is required');
    });
  });

  describe('GET /api/food-logs/:id', () => {
    it('should get a specific food entry', async () => {
      // Create an entry
      const createResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [{ id: '1', uri: 'file:///photo.jpg', timestamp: new Date().toISOString() }],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'dinner',
        });

      const entryId = createResponse.body.entry.id;

      // Get it
      const response = await request(app)
        .get(`/api/food-logs/${entryId}`)
        .expect(200);

      expect(response.body.id).toBe(entryId);
      expect(response.body.userId).toBe(testUserId);
      expect(response.body.mealType).toBe('dinner');
      expect(response.body.ingredients).toBeDefined();
      expect(response.body.photos).toBeDefined();
    });

    it('should return 404 for non-existent entry', async () => {
      const response = await request(app)
        .get('/api/food-logs/00000000-0000-0000-0000-999999999999')
        .expect(404);

      expect(response.body.error).toBe('Entry not found');
    });
  });

  describe('PUT /api/food-logs/ingredients/:id', () => {
    it('should update an ingredient and recalculate totals', async () => {
      // Create an entry
      const createResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        });

      const entry = createResponse.body.entry;

      // Debug: log the response if ingredients are missing
      if (!entry.ingredients || entry.ingredients.length === 0) {
        console.log('Entry:', JSON.stringify(entry, null, 2));
        throw new Error('No ingredients in entry');
      }

      const ingredientId = entry.ingredients[0].id;
      const originalQuantity = entry.ingredients[0].quantity;

      // Update ingredient
      const response = await request(app)
        .put(`/api/food-logs/ingredients/${ingredientId}`)
        .send({ quantity: originalQuantity * 2 })
        .expect(200);

      expect(response.body.quantity).toBe(originalQuantity * 2);
      expect(response.body.userModified).toBe(true);

      // Verify entry totals were updated
      const updatedEntry = await request(app)
        .get(`/api/food-logs/${entry.id}`)
        .expect(200);

      // Should be roughly double (with some rounding)
      expect(updatedEntry.body.totalCalories).toBeGreaterThan(entry.totalCalories * 1.8);
    });
  });

  describe('DELETE /api/food-logs/:id', () => {
    it('should delete a food entry', async () => {
      // Create an entry
      const createResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'snack',
        });

      const entryId = createResponse.body.entry.id;

      // Delete it
      await request(app)
        .delete(`/api/food-logs/${entryId}`)
        .expect(200);

      // Verify it's gone
      await request(app)
        .get(`/api/food-logs/${entryId}`)
        .expect(404);
    });
  });

  describe('End-to-end flow', () => {
    it('should handle complete user journey', async () => {
      // 1. Process photos
      const processResponse = await request(app)
        .post('/api/food-logs/process')
        .send({
          photos: [
            { id: '1', uri: 'file:///meal.jpg', timestamp: new Date().toISOString() },
          ],
          userId: testUserId,
          userRegion: 'AU',
          mealType: 'lunch',
        })
        .expect(200);

      const entryId = processResponse.body.entry.id;
      expect(entryId).toBeDefined();

      // 2. Get the entry
      const getResponse = await request(app)
        .get(`/api/food-logs/${entryId}`)
        .expect(200);

      expect(getResponse.body.id).toBe(entryId);

      // 3. Update an ingredient
      const ingredientId = getResponse.body.ingredients[0].id;
      await request(app)
        .put(`/api/food-logs/ingredients/${ingredientId}`)
        .send({ quantity: 200 })
        .expect(200);

      // 4. List all entries
      const listResponse = await request(app)
        .get('/api/food-logs')
        .query({ userId: testUserId })
        .expect(200);

      expect(listResponse.body.length).toBeGreaterThanOrEqual(1);

      // 5. Delete the entry
      await request(app)
        .delete(`/api/food-logs/${entryId}`)
        .expect(200);

      // 6. Verify deletion
      await request(app)
        .get(`/api/food-logs/${entryId}`)
        .expect(404);
    });
  });
});
