import 'dotenv/config';
import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import pool from '../../db/client.js';
import * as foodEntryService from '../foodEntryService.js';
const testUserId = '00000000-0000-0000-0000-000000000001';
beforeAll(async () => {
    // Create test user
    await pool.query(`
    INSERT INTO users (id, email, name, region)
    VALUES ($1, 'test@test.com', 'Test User', 'AU')
    ON CONFLICT (email) DO UPDATE SET id = $1
  `, [testUserId]);
});
afterAll(async () => {
    // Clean up
    await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
    await pool.end();
});
beforeEach(async () => {
    // Clean up before each test
    await pool.query('DELETE FROM food_entries WHERE user_id = $1', [testUserId]);
});
describe('Food Entry Service', () => {
    describe('createFoodEntry', () => {
        it('should create a food entry with photos and ingredients', async () => {
            const entry = await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'lunch',
                entryDate: new Date('2026-02-02'),
                photos: [
                    { uri: 'file:///photo1.jpg', width: 1920, height: 1080 },
                ],
                ingredients: [
                    {
                        name: 'Chicken Breast',
                        quantity: 150,
                        unit: 'g',
                        calories: 165,
                        protein: 31,
                        carbs: 0,
                        fat: 3.6,
                        databaseSource: 'AFCD',
                    },
                ],
                notes: 'Test meal',
            });
            expect(entry).toBeDefined();
            expect(entry.id).toBeDefined();
            expect(entry.userId).toBe(testUserId);
            expect(entry.mealType).toBe('lunch');
            expect(entry.totalCalories).toBe(165);
            expect(entry.totalProtein).toBe(31);
        });
        it('should calculate totals from ingredients', async () => {
            const entry = await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'dinner',
                entryDate: new Date(),
                photos: [],
                ingredients: [
                    { name: 'Rice', quantity: 100, unit: 'g', calories: 130, protein: 2.7, carbs: 28, fat: 0.3, databaseSource: 'AFCD' },
                    { name: 'Broccoli', quantity: 100, unit: 'g', calories: 35, protein: 2.8, carbs: 7, fat: 0.4, databaseSource: 'AFCD' },
                ],
            });
            expect(entry.totalCalories).toBe(165);
            expect(entry.totalProtein).toBe(5.5);
            expect(entry.totalCarbs).toBe(35);
            expect(entry.totalFat).toBe(0.7);
        });
    });
    describe('getFoodEntries', () => {
        it('should get all entries for a user', async () => {
            await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'breakfast',
                entryDate: new Date(),
                photos: [],
                ingredients: [],
            });
            await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'lunch',
                entryDate: new Date(),
                photos: [],
                ingredients: [],
            });
            const entries = await foodEntryService.getFoodEntries(testUserId);
            expect(entries.length).toBeGreaterThanOrEqual(2);
        });
        it('should filter entries by date range', async () => {
            const today = new Date('2026-02-02');
            const yesterday = new Date('2026-02-01');
            await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'lunch',
                entryDate: today,
                photos: [],
                ingredients: [],
            });
            await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'lunch',
                entryDate: yesterday,
                photos: [],
                ingredients: [],
            });
            const entries = await foodEntryService.getFoodEntries(testUserId, {
                startDate: today,
                endDate: today,
            });
            expect(entries.length).toBe(1);
            const entryDate = new Date(entries[0].entryDate);
            expect(entryDate.getFullYear()).toBe(2026);
            expect(entryDate.getMonth()).toBe(1); // February is month 1
        });
    });
    describe('updateIngredient', () => {
        it('should update ingredient and recalculate totals', async () => {
            const entry = await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'lunch',
                entryDate: new Date(),
                photos: [],
                ingredients: [
                    { name: 'Chicken', quantity: 100, unit: 'g', calories: 110, protein: 20, carbs: 0, fat: 2.4, databaseSource: 'AFCD' },
                ],
            });
            const ingredientId = entry.ingredients[0].id;
            const updated = await foodEntryService.updateIngredient(ingredientId, {
                quantity: 150, // Changed from 100g to 150g
            });
            expect(updated.quantity).toBe(150);
            expect(updated.userModified).toBe(true);
            // Check that entry totals were recalculated
            const updatedEntry = await foodEntryService.getFoodEntry(entry.id);
            expect(updatedEntry?.totalCalories).toBe(165); // 110 * 1.5
        });
        it('should track modification history', async () => {
            const entry = await foodEntryService.createFoodEntry({
                userId: testUserId,
                mealType: 'lunch',
                entryDate: new Date(),
                photos: [],
                ingredients: [
                    { name: 'Rice', quantity: 100, unit: 'g', calories: 130, protein: 2.7, carbs: 28, fat: 0.3, databaseSource: 'AFCD' },
                ],
            });
            const ingredientId = entry.ingredients[0].id;
            await foodEntryService.updateIngredient(ingredientId, { quantity: 150 });
            const history = await foodEntryService.getModificationHistory(ingredientId);
            expect(history.length).toBeGreaterThan(0);
            expect(history[0].fieldName).toBe('quantity');
            expect(history[0].oldValue).toBe('100.00');
            expect(history[0].newValue).toBe('150');
        });
    });
});
