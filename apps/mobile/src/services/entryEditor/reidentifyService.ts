/**
 * ReidentifyService -- Gemini Nano re-scan + KG enrichment + merge logic.
 *
 * Provides:
 * - reidentifyEntry(): re-scans a photo via Gemini Nano, enriches with KG nutrition
 * - applyMergeResult(): persists the user's keep-column selection to SQLite
 *
 * Gemini Nano ONLY -- no VLM fallback, no cloud inference.
 * If geminiNanoService.identify() throws, the error propagates to the UI.
 */

import { geminiNanoService } from '../vlm/geminiNanoService';
import { getKnowledgeGraphService } from '../knowledge-graph';
import {
  addIngredient,
  removeIngredient,
  recalculateEntryTotals,
} from './entryEditorService';
import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single item (ingredient) in the merge view. */
export interface MergeItem {
  id: string;
  name: string;
  amountG: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  dishName: string;
  source: 'existing' | 'new';
}

/** A dish with enriched ingredients from the Gemini Nano re-scan. */
export interface MergeCandidate {
  dishName: string;
  cuisine: string | null;
  ingredients: MergeItem[];
}

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
// Re-identification
// ---------------------------------------------------------------------------

/**
 * Re-scan a photo with Gemini Nano and enrich results via KG.
 *
 * 1. Calls geminiNanoService.identify(photoUri) to get VlmFoodResult
 * 2. For each dish/ingredient, looks up nutrition via KG
 * 3. Returns enriched MergeCandidate[] for the merge UI
 *
 * Throws if Gemini Nano fails -- the UI shows an Alert.
 */
export async function reidentifyEntry(
  photoUri: string,
): Promise<MergeCandidate[]> {
  // 1. Re-scan with Gemini Nano
  const vlmResult = await geminiNanoService.identify(photoUri);

  if (!vlmResult.dishes || vlmResult.dishes.length === 0) {
    return [];
  }

  // 2. Enrich via KG
  const kg = await getKnowledgeGraphService();
  const candidates: MergeCandidate[] = [];

  for (const dish of vlmResult.dishes) {
    const enrichedIngredients: MergeItem[] = [];

    for (const ing of dish.ingredients) {
      let calories = 0;
      let protein = 0;
      let carbs = 0;
      let fat = 0;
      let fiber = 0;

      // Try KG nutrition lookup
      if (kg) {
        const nutrition = await kg.calculateDishNutrition(
          ing.name,
          ing.amount_g,
        );
        if (nutrition) {
          calories = nutrition.calories;
          protein = nutrition.protein;
          carbs = nutrition.carbs;
          fat = nutrition.fat;
        }
      }

      enrichedIngredients.push({
        id: generateId(),
        name: ing.name,
        amountG: ing.amount_g,
        calories: Math.round(calories),
        protein: Math.round(protein * 10) / 10,
        carbs: Math.round(carbs * 10) / 10,
        fat: Math.round(fat * 10) / 10,
        fiber: Math.round(fiber * 10) / 10,
        dishName: dish.name,
        source: 'new' as const,
      });
    }

    candidates.push({
      dishName: dish.name,
      cuisine: dish.cuisine ?? null,
      ingredients: enrichedIngredients,
    });
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// Apply merge result
// ---------------------------------------------------------------------------

/**
 * Persist the user's keep-column selection to SQLite.
 *
 * 1. Deletes ALL existing ingredients for the entry
 * 2. Inserts each keepItem via addIngredient()
 * 3. Recalculates entry totals
 *
 * This is the "Save+Confirm" action from the merge UI.
 */
export function applyMergeResult(
  entryId: string,
  keepItems: MergeItem[],
): void {
  // 1. Delete all existing ingredients for this entry
  const existingIngredients = opsqlite.executeSync(
    'SELECT id FROM ingredients WHERE entry_id = ?',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  for (const row of existingIngredients) {
    removeIngredient(row.id as string);
  }

  // 2. Get first dish ID for the entry (or null)
  const dishRows = opsqlite.executeSync(
    'SELECT id FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at LIMIT 1',
    [entryId],
  ).rows as Array<Record<string, unknown>>;
  const defaultDishId = dishRows.length > 0 ? (dishRows[0].id as string) : null;

  // 3. Insert keepItems
  for (const item of keepItems) {
    addIngredient({
      entryId,
      dishId: defaultDishId,
      name: item.name,
      amountG: item.amountG,
      calories: item.calories,
      protein: item.protein,
      carbs: item.carbs,
      fat: item.fat,
      fiber: item.fiber,
    });
  }

  // 4. Recalculate totals
  recalculateEntryTotals(entryId);
}
