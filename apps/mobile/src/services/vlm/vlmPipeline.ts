/**
 * VLM refinement pipeline.
 *
 * Orchestrates progressive refinement: YOLO results -> VLM identification ->
 * KG nutrition lookup. Matches VLM dishes to YOLO bounding boxes via
 * substring + word overlap similarity.
 *
 * Designed for graceful fallback:
 * - VLM not ready -> items unchanged
 * - VLM inference fails -> items unchanged (logged in __DEV__)
 * - KG not available -> items get vlmLabel but no KG nutrition override
 * - No match for a VLM dish -> dish ignored (no phantom items)
 */

import { vlmService } from './vlmService';
import type { VlmDish, VlmFoodResult } from './vlmTypes';
import type { DetectedItem } from '../detection/types';
import { getKnowledgeGraphService } from '../knowledge-graph';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run VLM refinement on detected items.
 *
 * @param photoUri - File URI of the photo.
 * @param items - YOLO-detected items to refine.
 * @param userText - Optional user-provided meal description for VLM disambiguation.
 * @returns Refined items with vlmLabel/vlmCuisine/vlmIngredients populated for matches.
 */
export async function runVlmRefinement(
  photoUri: string,
  items: DetectedItem[],
  userText?: string,
): Promise<DetectedItem[]> {
  // Early exit: VLM not initialized or not available
  if (!vlmService.isReady) {
    return items;
  }

  // Run VLM inference (wrapped in try/catch for graceful fallback)
  let vlmResult: VlmFoodResult;
  try {
    vlmResult = await vlmService.identify(photoUri, userText);
  } catch (err) {
    if (__DEV__) {
      console.warn(
        '[VLM Pipeline] Inference failed, falling back to YOLO labels:',
        err instanceof Error ? err.message : err,
      );
    }
    return items;
  }

  // Match VLM dishes to YOLO items
  const matchMap = matchVlmToYolo(items, vlmResult);

  // Look up KG nutrition for matched VLM dishes
  const kgService = await getKnowledgeGraphService();

  // Build refined items
  const refined = await Promise.all(
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

  return refined;
}

// ---------------------------------------------------------------------------
// Internal: VLM-to-YOLO matching
// ---------------------------------------------------------------------------

/**
 * Match VLM dishes to YOLO-detected items.
 *
 * Strategy:
 * 1. Substring match (case-insensitive): VLM dish name contains YOLO class or vice versa
 * 2. Word overlap ratio (>= 0.3 threshold)
 * 3. Positional fallback: first VLM dish -> largest YOLO bbox
 *
 * @returns Map from item ID to matched VLM dish.
 */
function matchVlmToYolo(
  items: DetectedItem[],
  vlmResult: VlmFoodResult,
): Map<string, VlmDish> {
  const result = new Map<string, VlmDish>();
  const usedItemIds = new Set<string>();
  const unmatchedDishes: VlmDish[] = [];

  for (const dish of vlmResult.dishes) {
    const dishNameLower = dish.name.toLowerCase();
    let matched = false;

    // Strategy 1: Substring match
    for (const item of items) {
      if (usedItemIds.has(item.id)) continue;
      const classLower = item.className.toLowerCase();

      if (dishNameLower.includes(classLower) || classLower.includes(dishNameLower)) {
        result.set(item.id, dish);
        usedItemIds.add(item.id);
        matched = true;
        break;
      }
    }

    if (matched) continue;

    // Strategy 2: Word overlap
    let bestOverlap = 0;
    let bestItem: DetectedItem | null = null;

    for (const item of items) {
      if (usedItemIds.has(item.id)) continue;
      const overlap = computeWordOverlap(dishNameLower, item.className.toLowerCase());
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestItem = item;
      }
    }

    if (bestItem && bestOverlap >= 0.3) {
      result.set(bestItem.id, dish);
      usedItemIds.add(bestItem.id);
      matched = true;
    }

    if (matched) continue;

    // Strategy 3: Positional fallback -- match to largest unmatched bbox
    const unmatched = items
      .filter((i) => !usedItemIds.has(i.id))
      .sort((a, b) => (b.bbox.w * b.bbox.h) - (a.bbox.w * a.bbox.h));

    if (unmatched.length > 0) {
      result.set(unmatched[0].id, dish);
      usedItemIds.add(unmatched[0].id);
    } else {
      unmatchedDishes.push(dish);
    }
  }

  if (__DEV__ && unmatchedDishes.length > 0) {
    console.log(
      '[VLM Pipeline] Unmatched VLM dishes (no YOLO boxes available):',
      unmatchedDishes.map((d) => d.name),
    );
  }

  return result;
}

/**
 * Compute word overlap ratio between two strings.
 *
 * Splits both strings into words, counts common words,
 * and returns overlap / max(wordsA.length, wordsB.length).
 */
function computeWordOverlap(a: string, b: string): number {
  const wordsA = a.split(/\s+/).filter(Boolean);
  const wordsB = b.split(/\s+/).filter(Boolean);

  if (wordsA.length === 0 || wordsB.length === 0) return 0;

  const setB = new Set(wordsB);
  let common = 0;
  for (const word of wordsA) {
    if (setB.has(word)) common++;
  }

  return common / Math.max(wordsA.length, wordsB.length);
}
