import pool, { transaction } from '../db/client.js';
/**
 * Create a food entry with photos and ingredients
 */
export async function createFoodEntry(params) {
    return transaction(async (client) => {
        // Calculate totals
        const totals = params.ingredients.reduce((acc, ing) => ({
            calories: acc.calories + ing.calories,
            protein: acc.protein + (ing.protein || 0),
            carbs: acc.carbs + (ing.carbs || 0),
            fat: acc.fat + (ing.fat || 0),
        }), { calories: 0, protein: 0, carbs: 0, fat: 0 });
        // Insert entry
        const entryResult = await client.query(`INSERT INTO food_entries (user_id, meal_type, entry_date, total_calories, total_protein, total_carbs, total_fat, notes)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       RETURNING *`, [
            params.userId,
            params.mealType,
            params.entryDate,
            totals.calories,
            totals.protein,
            totals.carbs,
            totals.fat,
            params.notes,
        ]);
        const entry = entryResult.rows[0];
        // Insert photos
        const photos = [];
        for (const photo of params.photos) {
            const photoResult = await client.query(`INSERT INTO photos (entry_id, uri, gcs_url, width, height, latitude, longitude)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         RETURNING *`, [entry.id, photo.uri, photo.gcsUrl, photo.width, photo.height, photo.latitude, photo.longitude]);
            photos.push(photoResult.rows[0]);
        }
        // Insert ingredients
        const ingredients = [];
        for (const ing of params.ingredients) {
            const ingResult = await client.query(`INSERT INTO ingredients (
          entry_id, name, quantity, unit, calories, protein, carbs, fat, fiber, sugar,
          ai_confidence, bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height,
          database_source, database_id, original_quantity
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        RETURNING *`, [
                entry.id,
                ing.name,
                ing.quantity,
                ing.unit,
                ing.calories,
                ing.protein || 0,
                ing.carbs || 0,
                ing.fat || 0,
                ing.fiber || 0,
                ing.sugar || 0,
                ing.aiConfidence,
                ing.boundingBox?.x,
                ing.boundingBox?.y,
                ing.boundingBox?.width,
                ing.boundingBox?.height,
                ing.databaseSource,
                ing.databaseId,
                ing.quantity, // original_quantity = quantity initially
            ]);
            ingredients.push(ingResult.rows[0]);
        }
        return {
            id: entry.id,
            userId: entry.user_id,
            mealType: entry.meal_type,
            entryDate: entry.entry_date,
            totalCalories: parseFloat(entry.total_calories),
            totalProtein: parseFloat(entry.total_protein),
            totalCarbs: parseFloat(entry.total_carbs),
            totalFat: parseFloat(entry.total_fat),
            notes: entry.notes,
            createdAt: entry.created_at,
            updatedAt: entry.updated_at,
            photos,
            ingredients,
        };
    });
}
/**
 * Get food entries for a user
 */
export async function getFoodEntries(userId, options) {
    let query = `
    SELECT * FROM food_entries
    WHERE user_id = $1
  `;
    const params = [userId];
    if (options?.startDate) {
        params.push(options.startDate);
        query += ` AND entry_date >= $${params.length}`;
    }
    if (options?.endDate) {
        params.push(options.endDate);
        query += ` AND entry_date <= $${params.length}`;
    }
    query += ` ORDER BY entry_date DESC, created_at DESC`;
    if (options?.limit) {
        params.push(options.limit);
        query += ` LIMIT $${params.length}`;
    }
    const result = await pool.query(query, params);
    // Load photos and ingredients for each entry
    const entries = await Promise.all(result.rows.map(async (entry) => {
        const [photosResult, ingredientsResult] = await Promise.all([
            pool.query('SELECT * FROM photos WHERE entry_id = $1', [entry.id]),
            pool.query('SELECT * FROM ingredients WHERE entry_id = $1 ORDER BY created_at', [entry.id]),
        ]);
        return {
            id: entry.id,
            userId: entry.user_id,
            mealType: entry.meal_type,
            entryDate: entry.entry_date,
            totalCalories: parseFloat(entry.total_calories),
            totalProtein: parseFloat(entry.total_protein),
            totalCarbs: parseFloat(entry.total_carbs),
            totalFat: parseFloat(entry.total_fat),
            notes: entry.notes,
            createdAt: entry.created_at,
            updatedAt: entry.updated_at,
            photos: photosResult.rows,
            ingredients: ingredientsResult.rows,
        };
    }));
    return entries;
}
/**
 * Get a single food entry by ID
 */
export async function getFoodEntry(entryId) {
    const result = await pool.query('SELECT * FROM food_entries WHERE id = $1', [entryId]);
    if (result.rows.length === 0) {
        return null;
    }
    const entry = result.rows[0];
    const [photosResult, ingredientsResult] = await Promise.all([
        pool.query('SELECT * FROM photos WHERE entry_id = $1', [entryId]),
        pool.query('SELECT * FROM ingredients WHERE entry_id = $1 ORDER BY created_at', [entryId]),
    ]);
    return {
        id: entry.id,
        userId: entry.user_id,
        mealType: entry.meal_type,
        entryDate: entry.entry_date,
        totalCalories: parseFloat(entry.total_calories),
        totalProtein: parseFloat(entry.total_protein),
        totalCarbs: parseFloat(entry.total_carbs),
        totalFat: parseFloat(entry.total_fat),
        notes: entry.notes,
        createdAt: entry.created_at,
        updatedAt: entry.updated_at,
        photos: photosResult.rows,
        ingredients: ingredientsResult.rows,
    };
}
/**
 * Update an ingredient (for retrospective editing)
 */
export async function updateIngredient(ingredientId, updates) {
    return transaction(async (client) => {
        // Get current ingredient
        const currentResult = await client.query('SELECT * FROM ingredients WHERE id = $1', [ingredientId]);
        if (currentResult.rows.length === 0) {
            throw new Error('Ingredient not found');
        }
        const current = currentResult.rows[0];
        // Track modifications
        for (const [field, newValue] of Object.entries(updates)) {
            const currentValue = current[field.toLowerCase()] || current[field];
            if (currentValue != newValue) { // Use != to allow type coercion
                await client.query(`INSERT INTO modification_history (ingredient_id, field_name, old_value, new_value)
           VALUES ($1, $2, $3, $4)`, [ingredientId, field, String(currentValue), String(newValue)]);
            }
        }
        // If quantity is being updated, scale the nutrition values proportionally
        if (updates.quantity && current.quantity) {
            const scaleFactor = updates.quantity / parseFloat(current.quantity);
            if (!updates.calories)
                updates.calories = parseFloat(current.calories) * scaleFactor;
            if (!updates.protein)
                updates.protein = parseFloat(current.protein) * scaleFactor;
            if (!updates.carbs)
                updates.carbs = parseFloat(current.carbs) * scaleFactor;
            if (!updates.fat)
                updates.fat = parseFloat(current.fat) * scaleFactor;
        }
        // Update ingredient
        const updateFields = [];
        const updateValues = [];
        let paramIndex = 1;
        for (const [field, value] of Object.entries(updates)) {
            updateFields.push(`${field} = $${paramIndex}`);
            updateValues.push(value);
            paramIndex++;
        }
        updateFields.push('user_modified = TRUE');
        updateValues.push(ingredientId);
        const updateQuery = `
      UPDATE ingredients
      SET ${updateFields.join(', ')}
      WHERE id = $${paramIndex}
      RETURNING *
    `;
        const updateResult = await client.query(updateQuery, updateValues);
        const updated = updateResult.rows[0];
        // Recalculate entry totals
        const entryId = updated.entry_id;
        const ingredientsResult = await client.query('SELECT * FROM ingredients WHERE entry_id = $1', [entryId]);
        const totals = ingredientsResult.rows.reduce((acc, ing) => ({
            calories: acc.calories + parseFloat(ing.calories),
            protein: acc.protein + parseFloat(ing.protein || 0),
            carbs: acc.carbs + parseFloat(ing.carbs || 0),
            fat: acc.fat + parseFloat(ing.fat || 0),
        }), { calories: 0, protein: 0, carbs: 0, fat: 0 });
        await client.query(`UPDATE food_entries
       SET total_calories = $1, total_protein = $2, total_carbs = $3, total_fat = $4
       WHERE id = $5`, [totals.calories, totals.protein, totals.carbs, totals.fat, entryId]);
        return {
            id: updated.id,
            entryId: updated.entry_id,
            name: updated.name,
            quantity: parseFloat(updated.quantity),
            unit: updated.unit,
            calories: parseFloat(updated.calories),
            protein: parseFloat(updated.protein),
            carbs: parseFloat(updated.carbs),
            fat: parseFloat(updated.fat),
            fiber: updated.fiber ? parseFloat(updated.fiber) : undefined,
            sugar: updated.sugar ? parseFloat(updated.sugar) : undefined,
            userModified: updated.user_modified,
            originalQuantity: updated.original_quantity ? parseFloat(updated.original_quantity) : undefined,
            databaseSource: updated.database_source,
            createdAt: updated.created_at,
            updatedAt: updated.updated_at,
        };
    });
}
/**
 * Get modification history for an ingredient
 */
export async function getModificationHistory(ingredientId) {
    const result = await pool.query('SELECT * FROM modification_history WHERE ingredient_id = $1 ORDER BY modified_at DESC', [ingredientId]);
    return result.rows.map(row => ({
        id: row.id,
        ingredientId: row.ingredient_id,
        fieldName: row.field_name,
        oldValue: row.old_value,
        newValue: row.new_value,
        modifiedAt: row.modified_at,
        modifiedBy: row.modified_by,
    }));
}
/**
 * Delete a food entry
 */
export async function deleteFoodEntry(entryId) {
    await pool.query('DELETE FROM food_entries WHERE id = $1', [entryId]);
}
