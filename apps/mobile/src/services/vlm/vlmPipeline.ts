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
import { enrichDishesWithKgIngredients } from '../detection/hiddenIngredientsService';
import type { ScannedDish, ScannedIngredient, ScanResult } from '../../types';
import type { VlmIngredient } from './vlmTypes';

// ---------------------------------------------------------------------------
// Model source tracking -- consumed by DetectionScreen to show which model ran
// ---------------------------------------------------------------------------

/** Which VLM ran for the last identification. Read via getLastVlmSource(). */
let _lastVlmSource: 'gemini-nano' | 'mock' | null = null;

/**
 * Returns which VLM ran the last scanFood() call in this session.
 * Used by DetectionScreen to show a model indicator badge.
 * Returns null if scanFood() hasn't been called yet.
 */
export function getLastVlmSource(): 'gemini-nano' | 'mock' | null {
  return _lastVlmSource;
}

/** Reset source tracking (for testing only). */
export function _resetVlmSource(): void {
  _lastVlmSource = null;
}

// ---------------------------------------------------------------------------
// Nutrition lookup constants
// ---------------------------------------------------------------------------

/** Last-resort flat-rate proxy when KG and USDA both return nothing. */
const PROXY_KCAL_PER_G    = 1.5;
const PROXY_PROTEIN_PER_G = 0.08;
const PROXY_CARBS_PER_G   = 0.20;
const PROXY_FAT_PER_G     = 0.06;

/**
 * Max fractional deviation allowed before a KG recipe/dish_average result is
 * rejected in favour of the USDA value. E.g., 0.40 = 40% tolerance.
 *
 * RecipeNLG dish data can be noisy. If KG says 800 kcal and USDA says 35 kcal
 * for the same 100g ingredient, we trust USDA. If they're within ±40% we use
 * the KG result (it may have better regional/preparation context).
 */
const KG_USDA_MARGIN = 0.40;

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

/**
 * Resolve nutrition for a detected ingredient/dish name at a given portion.
 *
 * Priority:
 *   1. USDA direct lookup — authoritative per-100g values for raw ingredients.
 *   2. KG recipe decomposition — USDA-backed per-ingredient sums, scaled to
 *      portion. Only accepted if within ±KG_USDA_MARGIN of USDA (when USDA
 *      also matched), or unconditionally when USDA had no match (composite dish).
 *   3. KG dish averages — same margin guard as recipe path.
 *   4. Flat-rate proxy — last resort, flags nutritionSource='proxy'.
 */
async function lookupNutrition(name: string, amount_g: number): Promise<NutritionFields> {
  let usdaResult: { calories: number; protein: number; carbs: number; fat: number } | null = null;
  let kgResult:   { calories: number; protein: number; carbs: number; fat: number } | null = null;

  try {
    const kg = await getKnowledgeGraphService();
    if (kg) {
      // 1. USDA direct — raw ingredient lookup (e.g., "broccoli" → usda_food)
      const usda = await kg.lookupUsdaIngredient(name, amount_g);
      if (usda) {
        usdaResult = { calories: usda.calories, protein: usda.protein, carbs: usda.carbs, fat: usda.fat };
      }

      // 2. KG recipe/dish_average (composite dishes, or USDA validation)
      const kg_r = await kg.calculateDishNutrition(name, amount_g);
      if (kg_r) {
        kgResult = { calories: kg_r.calories, protein: kg_r.protein, carbs: kg_r.carbs, fat: kg_r.fat };
      }
    }
  } catch {
    // KG unavailable — fall through
  }

  // If both found: use USDA unless KG is within margin (KG may have better
  // preparation-specific data, e.g., "sautéed" vs raw)
  if (usdaResult && kgResult) {
    const usda_cal = usdaResult.calories;
    const kg_cal   = kgResult.calories;
    const withinMargin =
      usda_cal > 0 &&
      Math.abs(kg_cal - usda_cal) / usda_cal <= KG_USDA_MARGIN;

    const chosen = withinMargin ? kgResult : usdaResult;
    return { ...chosen, fiber: 0, sodium: 0, nutritionSource: 'kg' };
  }

  // Only USDA matched (raw ingredient, not a composite dish)
  if (usdaResult) {
    return { ...usdaResult, fiber: 0, sodium: 0, nutritionSource: 'kg' };
  }

  // Only KG matched (composite dish not in USDA — accept without margin check)
  if (kgResult) {
    return { ...kgResult, fiber: 0, sodium: 0, nutritionSource: 'kg' };
  }

  // Last resort proxy
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
      } else {
        _lastVlmSource = 'gemini-nano';
      }
    } else {
      isMock = true;
      vlmResult = getMockScanResult();
    }
  } catch {
    isMock = true;
    vlmResult = getMockScanResult();
  }

  // Track mock fallback as source
  if (isMock) {
    _lastVlmSource = 'mock';
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

  // Enrich dishes that have no VLM-provided ingredients with KG data
  const enrichedDishes = await enrichDishesWithKgIngredients(dishes);

  return { photoUri, dishes: enrichedDishes, isMock };
}
