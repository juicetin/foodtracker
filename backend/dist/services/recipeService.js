import pool, { transaction } from '../db/client.js';
import { getFoodEntry, createFoodEntry } from './foodEntryService.js';
/**
 * Create a custom recipe from an existing food entry
 */
export async function createRecipeFromEntry(userId, entryId, name, description) {
    return transaction(async (client) => {
        // Get the source entry
        const entry = await getFoodEntry(entryId);
        if (!entry) {
            throw new Error('Food entry not found');
        }
        // Create the recipe
        const recipeResult = await client.query(`INSERT INTO custom_recipes (user_id, name, description, source_entry_id, total_calories, total_protein, total_carbs, total_fat)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       RETURNING *`, [userId, name, description, entryId, entry.totalCalories, entry.totalProtein, entry.totalCarbs, entry.totalFat]);
        const recipe = recipeResult.rows[0];
        // Copy ingredients
        const ingredients = [];
        for (const ing of entry.ingredients || []) {
            const ingResult = await client.query(`INSERT INTO recipe_ingredients (recipe_id, name, quantity, unit, calories, protein, carbs, fat)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         RETURNING *`, [recipe.id, ing.name, ing.quantity, ing.unit, ing.calories, ing.protein, ing.carbs, ing.fat]);
            ingredients.push(ingResult.rows[0]);
        }
        // Copy photos
        const photos = [];
        for (const photo of entry.photos || []) {
            if (photo.gcs_url) {
                const photoResult = await client.query(`INSERT INTO recipe_photos (recipe_id, gcs_url, is_primary)
           VALUES ($1, $2, $3)
           RETURNING *`, [recipe.id, photo.gcs_url, photos.length === 0] // First photo is primary
                );
                photos.push({
                    id: photoResult.rows[0].id,
                    gcsUrl: photoResult.rows[0].gcs_url,
                    isPrimary: photoResult.rows[0].is_primary,
                });
            }
        }
        return {
            id: recipe.id,
            userId: recipe.user_id,
            name: recipe.name,
            description: recipe.description,
            sourceEntryId: recipe.source_entry_id,
            totalCalories: parseFloat(recipe.total_calories),
            totalProtein: parseFloat(recipe.total_protein),
            totalCarbs: parseFloat(recipe.total_carbs),
            totalFat: parseFloat(recipe.total_fat),
            timesUsed: recipe.times_used,
            lastUsedAt: recipe.last_used_at,
            createdAt: recipe.created_at,
            updatedAt: recipe.updated_at,
            ingredients: ingredients.map(ing => ({
                id: ing.id,
                name: ing.name,
                quantity: parseFloat(ing.quantity),
                unit: ing.unit,
                calories: parseFloat(ing.calories),
                protein: parseFloat(ing.protein),
                carbs: parseFloat(ing.carbs),
                fat: parseFloat(ing.fat),
            })),
            photos,
        };
    });
}
/**
 * Get all recipes for a user
 */
export async function getRecipes(userId) {
    const result = await pool.query(`SELECT * FROM custom_recipes
     WHERE user_id = $1
     ORDER BY last_used_at DESC NULLS LAST, created_at DESC`, [userId]);
    const recipes = await Promise.all(result.rows.map(async (recipe) => {
        const [ingredientsResult, photosResult] = await Promise.all([
            pool.query('SELECT * FROM recipe_ingredients WHERE recipe_id = $1', [recipe.id]),
            pool.query('SELECT * FROM recipe_photos WHERE recipe_id = $1', [recipe.id]),
        ]);
        return {
            id: recipe.id,
            userId: recipe.user_id,
            name: recipe.name,
            description: recipe.description,
            sourceEntryId: recipe.source_entry_id,
            totalCalories: parseFloat(recipe.total_calories),
            totalProtein: parseFloat(recipe.total_protein),
            totalCarbs: parseFloat(recipe.total_carbs),
            totalFat: parseFloat(recipe.total_fat),
            timesUsed: recipe.times_used,
            lastUsedAt: recipe.last_used_at,
            createdAt: recipe.created_at,
            updatedAt: recipe.updated_at,
            ingredients: ingredientsResult.rows.map(ing => ({
                id: ing.id,
                name: ing.name,
                quantity: parseFloat(ing.quantity),
                unit: ing.unit,
                calories: parseFloat(ing.calories),
                protein: parseFloat(ing.protein),
                carbs: parseFloat(ing.carbs),
                fat: parseFloat(ing.fat),
            })),
            photos: photosResult.rows.map(photo => ({
                id: photo.id,
                gcsUrl: photo.gcs_url,
                isPrimary: photo.is_primary,
            })),
        };
    }));
    return recipes;
}
/**
 * Get a single recipe by ID
 */
export async function getRecipe(recipeId) {
    const result = await pool.query('SELECT * FROM custom_recipes WHERE id = $1', [recipeId]);
    if (result.rows.length === 0) {
        return null;
    }
    const recipe = result.rows[0];
    const [ingredientsResult, photosResult] = await Promise.all([
        pool.query('SELECT * FROM recipe_ingredients WHERE recipe_id = $1', [recipeId]),
        pool.query('SELECT * FROM recipe_photos WHERE recipe_id = $1', [recipeId]),
    ]);
    return {
        id: recipe.id,
        userId: recipe.user_id,
        name: recipe.name,
        description: recipe.description,
        sourceEntryId: recipe.source_entry_id,
        totalCalories: parseFloat(recipe.total_calories),
        totalProtein: parseFloat(recipe.total_protein),
        totalCarbs: parseFloat(recipe.total_carbs),
        totalFat: parseFloat(recipe.total_fat),
        timesUsed: recipe.times_used,
        lastUsedAt: recipe.last_used_at,
        createdAt: recipe.created_at,
        updatedAt: recipe.updated_at,
        ingredients: ingredientsResult.rows.map(ing => ({
            id: ing.id,
            name: ing.name,
            quantity: parseFloat(ing.quantity),
            unit: ing.unit,
            calories: parseFloat(ing.calories),
            protein: parseFloat(ing.protein),
            carbs: parseFloat(ing.carbs),
            fat: parseFloat(ing.fat),
        })),
        photos: photosResult.rows.map(photo => ({
            id: photo.id,
            gcsUrl: photo.gcs_url,
            isPrimary: photo.is_primary,
        })),
    };
}
/**
 * Use a recipe to create a new food entry
 */
export async function useRecipe(recipeId, userId, options) {
    const recipe = await getRecipe(recipeId);
    if (!recipe) {
        throw new Error('Recipe not found');
    }
    // Create a new food entry from the recipe
    const entry = await createFoodEntry({
        userId,
        mealType: options?.mealType || 'lunch',
        entryDate: options?.entryDate || new Date(),
        photos: [], // No photos from recipe
        ingredients: recipe.ingredients.map(ing => ({
            name: ing.name,
            quantity: ing.quantity,
            unit: ing.unit,
            calories: ing.calories,
            protein: ing.protein,
            carbs: ing.carbs,
            fat: ing.fat,
            databaseSource: 'RECIPE',
        })),
        notes: `From recipe: ${recipe.name}`,
    });
    // Update recipe usage stats
    await pool.query(`UPDATE custom_recipes
     SET times_used = times_used + 1,
         last_used_at = CURRENT_TIMESTAMP
     WHERE id = $1`, [recipeId]);
    return entry;
}
/**
 * Search recipes by name
 */
export async function searchRecipes(userId, query) {
    const result = await pool.query(`SELECT * FROM custom_recipes
     WHERE user_id = $1
       AND name ILIKE $2
     ORDER BY last_used_at DESC NULLS LAST, created_at DESC`, [userId, `%${query}%`]);
    const recipes = await Promise.all(result.rows.map(async (recipe) => {
        const [ingredientsResult, photosResult] = await Promise.all([
            pool.query('SELECT * FROM recipe_ingredients WHERE recipe_id = $1', [recipe.id]),
            pool.query('SELECT * FROM recipe_photos WHERE recipe_id = $1', [recipe.id]),
        ]);
        return {
            id: recipe.id,
            userId: recipe.user_id,
            name: recipe.name,
            description: recipe.description,
            sourceEntryId: recipe.source_entry_id,
            totalCalories: parseFloat(recipe.total_calories),
            totalProtein: parseFloat(recipe.total_protein),
            totalCarbs: parseFloat(recipe.total_carbs),
            totalFat: parseFloat(recipe.total_fat),
            timesUsed: recipe.times_used,
            lastUsedAt: recipe.last_used_at,
            createdAt: recipe.created_at,
            updatedAt: recipe.updated_at,
            ingredients: ingredientsResult.rows.map(ing => ({
                id: ing.id,
                name: ing.name,
                quantity: parseFloat(ing.quantity),
                unit: ing.unit,
                calories: parseFloat(ing.calories),
                protein: parseFloat(ing.protein),
                carbs: parseFloat(ing.carbs),
                fat: parseFloat(ing.fat),
            })),
            photos: photosResult.rows.map(photo => ({
                id: photo.id,
                gcsUrl: photo.gcs_url,
                isPrimary: photo.is_primary,
            })),
        };
    }));
    return recipes;
}
/**
 * Delete a recipe
 */
export async function deleteRecipe(recipeId) {
    await pool.query('DELETE FROM custom_recipes WHERE id = $1', [recipeId]);
}
