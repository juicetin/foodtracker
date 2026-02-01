import express from 'express';
import * as recipeService from '../services/recipeService.js';
const router = express.Router();
/**
 * GET /api/recipes
 * Get all recipes for a user
 */
router.get('/', async (req, res) => {
    try {
        const { userId } = req.query;
        if (!userId || typeof userId !== 'string') {
            return res.status(400).json({ error: 'User ID is required' });
        }
        const recipes = await recipeService.getRecipes(userId);
        res.json(recipes);
    }
    catch (error) {
        console.error('Error fetching recipes:', error);
        res.status(500).json({
            error: error instanceof Error ? error.message : 'Failed to fetch recipes',
        });
    }
});
/**
 * GET /api/recipes/search
 * Search recipes by name
 */
router.get('/search', async (req, res) => {
    try {
        const { userId, q } = req.query;
        if (!userId || typeof userId !== 'string') {
            return res.status(400).json({ error: 'User ID is required' });
        }
        if (!q || typeof q !== 'string') {
            return res.status(400).json({ error: 'Search query is required' });
        }
        const recipes = await recipeService.searchRecipes(userId, q);
        res.json(recipes);
    }
    catch (error) {
        console.error('Error searching recipes:', error);
        res.status(500).json({
            error: error instanceof Error ? error.message : 'Failed to search recipes',
        });
    }
});
/**
 * POST /api/recipes
 * Create a recipe from a food entry
 */
router.post('/', async (req, res) => {
    try {
        const { userId, entryId, name, description } = req.body;
        if (!userId || !entryId || !name) {
            return res.status(400).json({ error: 'userId, entryId, and name are required' });
        }
        const recipe = await recipeService.createRecipeFromEntry(userId, entryId, name, description);
        res.json(recipe);
    }
    catch (error) {
        console.error('Error creating recipe:', error);
        res.status(500).json({
            error: error instanceof Error ? error.message : 'Failed to create recipe',
        });
    }
});
/**
 * POST /api/recipes/:id/use
 * Use a recipe to create a new food entry
 */
router.post('/:id/use', async (req, res) => {
    try {
        const { id } = req.params;
        const { userId, mealType, entryDate } = req.body;
        if (!userId) {
            return res.status(400).json({ error: 'User ID is required' });
        }
        const entry = await recipeService.useRecipe(id, userId, {
            mealType,
            entryDate: entryDate ? new Date(entryDate) : undefined,
        });
        res.json(entry);
    }
    catch (error) {
        console.error('Error using recipe:', error);
        res.status(500).json({
            error: error instanceof Error ? error.message : 'Failed to use recipe',
        });
    }
});
/**
 * DELETE /api/recipes/:id
 * Delete a recipe
 */
router.delete('/:id', async (req, res) => {
    try {
        const { id } = req.params;
        await recipeService.deleteRecipe(id);
        res.json({ success: true });
    }
    catch (error) {
        console.error('Error deleting recipe:', error);
        res.status(500).json({
            error: error instanceof Error ? error.message : 'Failed to delete recipe',
        });
    }
});
export default router;
