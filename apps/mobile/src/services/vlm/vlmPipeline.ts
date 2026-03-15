/**
 * VLM identification pipeline.
 *
 * Primary food identification engine: VLM is the sole source of food names
 * (YOLO only provides bounding boxes labeled 'Food Region'). Includes retry
 * logic for resilience and positional assignment of dishes to bounding boxes.
 *
 * VLM is required for usable detection — throws if VLM is not ready.
 * KG is optional — items get vlmLabel even if KG lookup fails.
 */

import { vlmService } from './vlmService';
import type { VlmDish, VlmFoodResult } from './vlmTypes';
import type { DetectedItem } from '../detection/types';
import { getKnowledgeGraphService } from '../knowledge-graph';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Identify food items with one silent retry on failure.
 *
 * @param photoUri - File URI of the photo.
 * @param userText - Optional user-provided meal description for VLM disambiguation.
 * @returns VLM identification result, or { dishes: [] } if both attempts fail.
 */
export async function identifyWithRetry(
  photoUri: string,
  userText?: string,
): Promise<VlmFoodResult> {
  try {
    return await vlmService.identify(photoUri, userText);
  } catch {
    // Silent retry — one more attempt
    try {
      return await vlmService.identify(photoUri, userText);
    } catch {
      // Both attempts failed — return empty result (no throw)
      return { dishes: [] };
    }
  }
}

/**
 * Assign dish names to items sorted by bounding box area descending.
 *
 * Largest bbox gets the first dish name, second largest gets the second, etc.
 * Skips removed items. Only assigns min(items, dishNames) pairs.
 *
 * @param items - Detected items with bounding boxes.
 * @param dishNames - Dish names to assign.
 * @returns Map from item ID to dish name.
 */
export function assignDishesToBoxes(
  items: DetectedItem[],
  dishNames: string[],
): Map<string, string> {
  const result = new Map<string, string>();

  // Filter out removed items and sort by bbox area descending
  const activeItems = items
    .filter((i) => !i.isRemoved)
    .sort((a, b) => (b.bbox.w * b.bbox.h) - (a.bbox.w * a.bbox.h));

  const count = Math.min(activeItems.length, dishNames.length);
  for (let i = 0; i < count; i++) {
    result.set(activeItems[i].id, dishNames[i]);
  }

  return result;
}

/**
 * Run VLM identification on detected items (primary identification, not refinement).
 *
 * @param photoUri - File URI of the photo.
 * @param items - YOLO-detected items (all labeled 'Food Region').
 * @param userText - Optional user-provided meal description for VLM disambiguation.
 * @returns Items with vlmLabel/vlmCuisine/vlmIngredients populated for matches.
 */
export async function runVlmIdentification(
  photoUri: string,
  items: DetectedItem[],
  userText?: string,
): Promise<DetectedItem[]> {
  // VLM is required — YOLO labels alone are not usable
  if (!vlmService.isReady) {
    throw new Error('VLM model is not loaded. Download the VLM pack first.');
  }

  // Run VLM inference with retry
  const vlmResult = await identifyWithRetry(photoUri, userText);

  // If VLM returned no dishes, return items unchanged
  if (vlmResult.dishes.length === 0) {
    return items;
  }

  // Positional matching: assign dishes to items by bbox area descending
  const matchMap = matchVlmToItems(items, vlmResult);

  // Look up KG nutrition for matched VLM dishes
  const kgService = await getKnowledgeGraphService();

  // Build identified items
  const identified = await Promise.all(
    items.map(async (item) => {
      const vlmDish = matchMap.get(item.id);
      if (!vlmDish) return item;

      // Try KG lookup for the VLM dish name
      if (kgService) {
        try {
          await kgService.searchDish(vlmDish.name);
        } catch {
          // KG lookup failed -- still apply VLM label
        }
      }

      return {
        ...item,
        vlmLabel: vlmDish.name,
        vlmCuisine: vlmDish.cuisine,
        vlmIngredients: vlmDish.ingredients,
        vlmConfidence: 0.8, // Default VLM confidence (model doesn't output it)
      };
    }),
  );

  return identified;
}

// ---------------------------------------------------------------------------
// Internal: Positional VLM-to-item matching
// ---------------------------------------------------------------------------

/**
 * Match VLM dishes to detected items by positional assignment.
 *
 * Since all items have className 'Food Region' (no meaningful YOLO labels),
 * matching is purely positional: sort non-removed items by bbox area descending,
 * assign first VLM dish to largest bbox, second to next largest, etc.
 *
 * @returns Map from item ID to matched VLM dish.
 */
function matchVlmToItems(
  items: DetectedItem[],
  vlmResult: VlmFoodResult,
): Map<string, VlmDish> {
  const result = new Map<string, VlmDish>();

  // Sort non-removed items by bbox area descending (largest first)
  const activeItems = items
    .filter((i) => !i.isRemoved)
    .sort((a, b) => (b.bbox.w * b.bbox.h) - (a.bbox.w * a.bbox.h));

  const count = Math.min(activeItems.length, vlmResult.dishes.length);
  for (let i = 0; i < count; i++) {
    result.set(activeItems[i].id, vlmResult.dishes[i]);
  }

  // Log unmatched dishes in dev mode
  if (__DEV__ && vlmResult.dishes.length > activeItems.length) {
    const unmatchedDishes = vlmResult.dishes.slice(activeItems.length);
    console.log(
      '[VLM Pipeline] Unmatched VLM dishes (no bounding boxes available):',
      unmatchedDishes.map((d) => d.name),
    );
  }

  return result;
}
