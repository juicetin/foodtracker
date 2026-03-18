/**
 * Gemini Nano scan pipeline.
 *
 * Sole inference path: Gemini Nano identifies dishes + ingredient weights.
 * KG provides per-ingredient nutrition. Mock data on unsupported devices.
 *
 * SmolVLM, YOLO, and EfficientNet are removed — Gemini Nano only.
 */

import { geminiNanoModule } from 'gemini-nano';
import { geminiNanoService } from './geminiNanoService';
import { getMockScanResult } from './geminiNanoMock';
import { getKnowledgeGraphService } from '../knowledge-graph';
import type { ScannedDish, ScannedIngredient, ScanResult } from '../../types';
import type { VlmIngredient } from './vlmTypes';

// ---------------------------------------------------------------------------
// Proxy constants (when KG has no data for an ingredient)
// ---------------------------------------------------------------------------

const PROXY_KCAL_PER_G   = 1.5;
const PROXY_PROTEIN_PER_G = 0.08;
const PROXY_CARBS_PER_G  = 0.20;
const PROXY_FAT_PER_G    = 0.06;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

type NutritionFields = Pick<
  ScannedIngredient,
  'calories' | 'protein' | 'carbs' | 'fat' | 'fiber' | 'sodium' | 'nutritionSource'
>;

async function lookupNutrition(name: string, amount_g: number): Promise<NutritionFields> {
  try {
    const kg = await getKnowledgeGraphService();
    if (kg) {
      const result = await kg.calculateDishNutrition(name, amount_g);
      if (result) {
        return {
          calories: result.calories,
          protein: result.protein,
          carbs:   result.carbs,
          fat:     result.fat,
          fiber:   0,
          sodium:  0,
          nutritionSource: 'kg',
        };
      }
    }
  } catch {
    // KG unavailable or ingredient not found — fall through to proxy
  }

  return {
    calories: amount_g * PROXY_KCAL_PER_G,
    protein:  amount_g * PROXY_PROTEIN_PER_G,
    carbs:    amount_g * PROXY_CARBS_PER_G,
    fat:      amount_g * PROXY_FAT_PER_G,
    fiber:    0,
    sodium:   0,
    nutritionSource: 'proxy',
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run a full food scan on a photo.
 *
 * 1. Check Gemini Nano availability.
 * 2. If available: run Gemini Nano identification.
 * 3. If unavailable / empty result: use mock data (flagged with isMock=true).
 * 4. Look up KG nutrition per ingredient.
 * 5. Return ScannedDish[] with full nutrition.
 */
export async function scanFood(photoUri: string): Promise<ScanResult> {
  let isMock = false;
  let vlmResult;

  try {
    const status = await geminiNanoModule.checkAvailability();
    if (status === 'available') {
      vlmResult = await geminiNanoService.identify(photoUri);
      if (!vlmResult || vlmResult.dishes.length === 0) {
        isMock = true;
        vlmResult = getMockScanResult();
      }
    } else {
      isMock = true;
      vlmResult = getMockScanResult();
    }
  } catch {
    isMock = true;
    vlmResult = getMockScanResult();
  }

  const dishes: ScannedDish[] = await Promise.all(
    vlmResult.dishes.map(async (dish) => {
      const ingredients: ScannedIngredient[] = await Promise.all(
        (dish.ingredients as VlmIngredient[]).map(async (ing) => {
          const nutrition = await lookupNutrition(ing.name, ing.amount_g);
          return {
            id: generateId(),
            name: ing.name,
            amount_g: ing.amount_g,
            originalAmount_g: ing.amount_g,
            userModified: false,
            ...nutrition,
          };
        }),
      );

      return {
        id: generateId(),
        name: dish.name,
        cuisine: dish.cuisine ?? null,
        photoUri,
        ingredients,
        portionScale: 1.0,
      };
    }),
  );

  return { photoUri, dishes, isMock };
}
