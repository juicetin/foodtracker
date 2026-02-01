import express from 'express';
import { processFoodPhotos, getFoodLogs } from '../services/aiService.js';
import { getFoodEntry, updateIngredient, deleteFoodEntry } from '../services/foodEntryService.js';

const router = express.Router();

/**
 * POST /api/food-logs/process
 * Process food photos with AI and create a food entry
 */
router.post('/process', async (req, res) => {
  try {
    const { photos, userId, userRegion, mealType } = req.body;

    if (!photos || !Array.isArray(photos)) {
      return res.status(400).json({ error: 'Photos are required' });
    }

    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' });
    }

    console.log(`Processing ${photos.length} photos for user ${userId}`);

    const result = await processFoodPhotos({
      photos,
      userId,
      userRegion: userRegion || 'AU',
      mealType: mealType || 'lunch',
    });

    res.json(result);
  } catch (error) {
    console.error('Error processing photos:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to process photos',
    });
  }
});

/**
 * GET /api/food-logs
 * Get food logs for a user
 */
router.get('/', async (req, res) => {
  try {
    const { userId } = req.query;

    if (!userId || typeof userId !== 'string') {
      return res.status(400).json({ error: 'User ID is required' });
    }

    const logs = await getFoodLogs(userId);
    res.json(logs);
  } catch (error) {
    console.error('Error fetching food logs:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to fetch food logs',
    });
  }
});

/**
 * GET /api/food-logs/:id
 * Get a specific food entry
 */
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const entry = await getFoodEntry(id);

    if (!entry) {
      return res.status(404).json({ error: 'Entry not found' });
    }

    res.json(entry);
  } catch (error) {
    console.error('Error fetching food entry:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to fetch entry',
    });
  }
});

/**
 * PUT /api/food-logs/ingredients/:id
 * Update an ingredient (retrospective editing)
 */
router.put('/ingredients/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updates = req.body;

    const updated = await updateIngredient(id, updates);
    res.json(updated);
  } catch (error) {
    console.error('Error updating ingredient:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to update ingredient',
    });
  }
});

/**
 * DELETE /api/food-logs/:id
 * Delete a food log entry
 */
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    await deleteFoodEntry(id);
    res.json({ success: true });
  } catch (error) {
    console.error('Error deleting food log:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to delete food log',
    });
  }
});

export default router;
