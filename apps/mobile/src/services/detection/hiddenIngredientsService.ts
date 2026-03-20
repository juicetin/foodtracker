/**
 * Hidden Ingredients Service (DET-04)
 *
 * Enriches detected dishes with KG-inferred ingredient lists when the VLM
 * did not provide ingredients. VLM-provided ingredients are always preserved.
 *
 * Flow per dish:
 *   1. If dish.ingredients.length > 0 => skip (VLM already provided)
 *   2. searchDish(dish.name) => find dish in KG
 *   3. getCanonicalRecipe(dishId) => get representative recipe
 *   4. getRecipeIngredients(recipeId) => get ingredient list with USDA nutrition
 *   5. Map to ScannedIngredient[] with kgInferred=true flag
 */

import {
  getKnowledgeGraphService,
} from '../knowledge-graph';
import type { ScannedDish, ScannedIngredient } from '../../types';

/** Extended ScannedIngredient with KG inference flag. */
export interface KgInferredIngredient extends ScannedIngredient {
  /** True when this ingredient was inferred from KG, not provided by VLM. */
  kgInferred: boolean;
}

/** Generate a simple UUID. */
function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/**
 * Enrich dishes that have no VLM-provided ingredients with KG ingredient data.
 *
 * - Dishes with existing ingredients are returned unchanged.
 * - Dishes without ingredients get KG recipe ingredients if a match is found.
 * - KG ingredients include kgInferred=true for UI differentiation.
 *
 * @param dishes - Array of ScannedDish from VLM pipeline
 * @returns Enriched dishes (same array order, same references for unchanged dishes)
 */
export async function enrichDishesWithKgIngredients(
  dishes: ScannedDish[],
): Promise<ScannedDish[]> {
  const kg = await getKnowledgeGraphService();
  if (!kg) return dishes;

  return Promise.all(
    dishes.map(async (dish) => {
      // VLM provided ingredients — keep as-is
      if (dish.ingredients.length > 0) {
        return dish;
      }

      try {
        // 1. Search KG for the dish
        const kgDish = await kg.searchDish(dish.name);
        if (!kgDish) return dish;

        // 2. Get canonical recipe
        const recipe = await kg.getCanonicalRecipe(kgDish.id);
        if (!recipe) return dish;

        // 3. Get recipe ingredients
        const kgIngredients = await kg.getRecipeIngredients(recipe.id);
        if (kgIngredients.length === 0) return dish;

        // 4. Map to ScannedIngredient with kgInferred flag
        const ingredients: KgInferredIngredient[] = kgIngredients.map((ing) => {
          const scale = ing.quantityGrams / 100;
          return {
            id: generateId(),
            name: ing.ingredientName,
            amount_g: ing.quantityGrams,
            originalAmount_g: ing.quantityGrams,
            calories: ing.caloriesPer100g * scale,
            protein: ing.proteinPer100g * scale,
            carbs: ing.carbsPer100g * scale,
            fat: ing.fatPer100g * scale,
            fiber: 0,
            sodium: 0,
            nutritionSource: 'kg' as const,
            userModified: false,
            kgInferred: true,
          };
        });

        return {
          ...dish,
          ingredients,
          kgInferredIngredients: kgIngredients.map((i) => i.ingredientName),
        };
      } catch {
        // KG lookup failed — return dish unchanged (graceful degradation)
        return dish;
      }
    }),
  );
}
