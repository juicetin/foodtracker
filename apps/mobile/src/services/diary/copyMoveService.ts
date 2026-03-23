/**
 * Copy and move operations for diary entries.
 *
 * Supports copying an entry to a different date (with all dishes and ingredients),
 * moving an entry to a different meal type, and bulk-copying all entries from one
 * date to another.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// ---------------------------------------------------------------------------
// Copy entry to date
// ---------------------------------------------------------------------------

/**
 * Copy a single food entry (with its dishes and ingredients) to a target date.
 * Photos are NOT copied (they reference original files on disk).
 *
 * @param sourceEntryId - ID of the entry to copy
 * @param targetDate - YYYY-MM-DD date string for the new entry
 * @param targetMealType - Optional meal type override; uses source meal_type if omitted
 * @returns The new entry ID
 */
export function copyEntryToDate(
  sourceEntryId: string,
  targetDate: string,
  targetMealType?: string,
): string {
  // Read source entry
  const sourceRows = opsqlite.executeSync(
    `SELECT meal_type, total_calories, total_protein, total_carbs, total_fat, notes
     FROM food_entries WHERE id = ? AND is_deleted = 0`,
    [sourceEntryId],
  ).rows as Array<Record<string, unknown>>;

  if (sourceRows.length === 0) {
    throw new Error(`Source entry not found: ${sourceEntryId}`);
  }

  const source = sourceRows[0];
  const newEntryId = generateId();
  const mealType = targetMealType ?? (source.meal_type as string);

  // Insert new food entry
  opsqlite.executeSync(
    `INSERT INTO food_entries (id, meal_type, entry_date, total_calories, total_protein, total_carbs, total_fat, notes, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))`,
    [
      newEntryId,
      mealType,
      targetDate,
      source.total_calories as number,
      source.total_protein as number,
      source.total_carbs as number,
      source.total_fat as number,
      source.notes as string | null,
    ],
  );

  // Copy scanned_dishes
  const dishRows = opsqlite.executeSync(
    'SELECT id, name, cuisine, portion_scale FROM scanned_dishes WHERE entry_id = ?',
    [sourceEntryId],
  ).rows as Array<Record<string, unknown>>;

  const dishIdMap = new Map<string, string>(); // old dish id -> new dish id
  for (const dish of dishRows) {
    const newDishId = generateId();
    dishIdMap.set(dish.id as string, newDishId);
    opsqlite.executeSync(
      `INSERT INTO scanned_dishes (id, entry_id, name, cuisine, portion_scale, created_at)
       VALUES (?, ?, ?, ?, ?, datetime('now'))`,
      [newDishId, newEntryId, dish.name as string, dish.cuisine as string | null, dish.portion_scale as number],
    );
  }

  // Copy ingredients
  const ingredientRows = opsqlite.executeSync(
    `SELECT dish_id, name, quantity, unit, amount_g, original_amount_g,
            calories, protein, carbs, fat, fiber, sodium,
            ai_confidence, database_source, user_modified
     FROM ingredients WHERE entry_id = ?`,
    [sourceEntryId],
  ).rows as Array<Record<string, unknown>>;

  for (const ing of ingredientRows) {
    const newIngId = generateId();
    const oldDishId = ing.dish_id as string | null;
    const newDishId = oldDishId ? (dishIdMap.get(oldDishId) ?? null) : null;

    opsqlite.executeSync(
      `INSERT INTO ingredients (id, entry_id, dish_id, name, quantity, unit, amount_g, original_amount_g,
                                calories, protein, carbs, fat, fiber, sodium,
                                ai_confidence, database_source, user_modified, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))`,
      [
        newIngId,
        newEntryId,
        newDishId,
        ing.name as string,
        ing.quantity as number,
        ing.unit as string,
        ing.amount_g as number | null,
        ing.original_amount_g as number | null,
        ing.calories as number,
        ing.protein as number,
        ing.carbs as number,
        ing.fat as number,
        ing.fiber as number | null,
        ing.sodium as number | null,
        ing.ai_confidence as number | null,
        ing.database_source as string | null,
        ing.user_modified as number,
      ],
    );
  }

  return newEntryId;
}

// ---------------------------------------------------------------------------
// Move entry to meal
// ---------------------------------------------------------------------------

/**
 * Move an entry to a different meal type by updating the meal_type column.
 */
export function moveEntryToMeal(entryId: string, newMealType: string): void {
  opsqlite.executeSync(
    `UPDATE food_entries SET meal_type = ?, updated_at = datetime('now') WHERE id = ?`,
    [newMealType, entryId],
  );
}

// ---------------------------------------------------------------------------
// Copy all entries from date
// ---------------------------------------------------------------------------

/**
 * Copy all entries from one date to another, optionally filtered by meal type.
 *
 * @param sourceDate - Source date (YYYY-MM-DD)
 * @param targetDate - Target date (YYYY-MM-DD)
 * @param filterMealType - If provided, only copy entries with this meal_type
 * @returns Number of entries copied
 */
export function copyAllEntriesFromDate(
  sourceDate: string,
  targetDate: string,
  filterMealType?: string,
): number {
  let query = `SELECT id FROM food_entries WHERE entry_date = ? AND is_deleted = 0`;
  const params: (string | number)[] = [sourceDate];

  if (filterMealType) {
    query += ' AND meal_type = ?';
    params.push(filterMealType);
  }

  const rows = opsqlite.executeSync(query, params).rows as Array<Record<string, unknown>>;

  for (const row of rows) {
    copyEntryToDate(row.id as string, targetDate);
  }

  return rows.length;
}
