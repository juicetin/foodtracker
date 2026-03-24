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
import { EmbeddingService } from '../embedding/embeddingService';
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
 *   1. USDA prefix lookup — exact/prefix match (fast, works for simple names).
 *   2. BM25 keyword search — tokenised term overlap (handles compound names).
 *   3. Vec semantic search — cosine similarity on MiniLM embeddings (broadest recall).
 *   4. KG recipe decomposition — USDA-backed per-ingredient sums.
 *   5. KG dish averages — fallback aggregate.
 *   6. Flat-rate proxy — last resort, flags nutritionSource='proxy'.
 */
async function lookupNutrition(name: string, amount_g: number): Promise<NutritionFields> {
  // Lazy warmup: load embedding TFLite model on first nutrition lookup (idempotent)
  EmbeddingService.getInstance().warmup().catch(() => {});

  let usdaResult: { calories: number; protein: number; carbs: number; fat: number } | null = null;
  let kgResult:   { calories: number; protein: number; carbs: number; fat: number } | null = null;

  try {
    const kg = await getKnowledgeGraphService();
    if (kg) {
      // 1. USDA prefix lookup (single-word raw ingredients, e.g., "broccoli")
      const prefix = await kg.lookupUsdaIngredient(name, amount_g);
      if (prefix) {
        usdaResult = { calories: prefix.calories, protein: prefix.protein, carbs: prefix.carbs, fat: prefix.fat };
      }

      // 2. BM25 keyword search (compound names, e.g., "pork cutlet")
      if (!usdaResult) {
        const bm25 = await kg.searchUsdaByBm25(name, amount_g);
        if (bm25) {
          usdaResult = { calories: bm25.calories, protein: bm25.protein, carbs: bm25.carbs, fat: bm25.fat };
        }
      }

      // 3. Vec semantic search (semantically similar names, e.g., "tonkatsu" → pork loin)
      if (!usdaResult) {
        const embSvc = EmbeddingService.getInstance();
        if (embSvc.ready) {
          const vec = await embSvc.embed(name);
          if (vec) {
            const vecResult = await kg.searchUsdaByVector(vec, amount_g);
            if (vecResult) {
              usdaResult = { calories: vecResult.calories, protein: vecResult.protein, carbs: vecResult.carbs, fat: vecResult.fat };
            }
          }
        }
      }

      // 4. KG recipe/dish_average (composite dishes, or USDA validation)
      const kg_r = await kg.calculateDishNutrition(name, amount_g);
      if (kg_r) {
        kgResult = { calories: kg_r.calories, protein: kg_r.protein, carbs: kg_r.carbs, fat: kg_r.fat };
      }
    }
  } catch {
    // KG unavailable — fall through
  }

  // If both USDA and KG found: use USDA unless KG is within margin
  if (usdaResult && kgResult) {
    const usda_cal = usdaResult.calories;
    const kg_cal   = kgResult.calories;
    const withinMargin =
      usda_cal > 0 &&
      Math.abs(kg_cal - usda_cal) / usda_cal <= KG_USDA_MARGIN;

    const chosen = withinMargin ? kgResult : usdaResult;
    return { ...chosen, fiber: 0, sodium: 0, nutritionSource: 'kg' };
  }

  if (usdaResult) {
    return { ...usdaResult, fiber: 0, sodium: 0, nutritionSource: 'kg' };
  }

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
// Benchmark utility (dev/testing)
// ---------------------------------------------------------------------------

/** Result of a single USDA search benchmark run. */
export type UsdaSearchBenchmarkResult = {
  name: string;
  portionGrams: number;
  prefix: { match: string | null; calories: number | null; latencyMs: number };
  bm25:   { match: string | null; calories: number | null; latencyMs: number };
  vec:    { match: string | null; calories: number | null; latencyMs: number; modelReady: boolean };
};

/**
 * Run all three USDA lookup strategies for a list of food names and return
 * timing + match results. Useful for comparing accuracy and latency on device.
 *
 * Import and call from a debug screen or console via __DEV__ guard.
 */
export async function benchmarkUsdaSearch(
  names: string[],
  portionGrams = 100,
): Promise<UsdaSearchBenchmarkResult[]> {
  const kg = await getKnowledgeGraphService();
  if (!kg) throw new Error('KG service unavailable');

  const embSvc = EmbeddingService.getInstance();
  const results: UsdaSearchBenchmarkResult[] = [];

  for (const name of names) {
    // 1. Prefix
    const t1 = performance.now();
    const prefix = await kg.lookupUsdaIngredient(name, portionGrams).catch(() => null);
    const prefixMs = performance.now() - t1;

    // 2. BM25
    const t2 = performance.now();
    const bm25 = await kg.searchUsdaByBm25(name, portionGrams).catch(() => null);
    const bm25Ms = performance.now() - t2;

    // 3. Vec (embed + search)
    const t3 = performance.now();
    let vecMatch: typeof prefix = null;
    if (embSvc.ready) {
      const vec = await embSvc.embed(name);
      if (vec) vecMatch = await kg.searchUsdaByVector(vec, portionGrams).catch(() => null);
    }
    const vecMs = performance.now() - t3;

    results.push({
      name,
      portionGrams,
      prefix: { match: prefix ? 'hit' : null, calories: prefix?.calories ?? null, latencyMs: prefixMs },
      bm25:   { match: bm25   ? 'hit' : null, calories: bm25?.calories   ?? null, latencyMs: bm25Ms  },
      vec:    { match: vecMatch ? 'hit' : null, calories: vecMatch?.calories ?? null, latencyMs: vecMs, modelReady: embSvc.ready },
    });

    // Log inline for easy logcat reading
    console.log(
      `[usda-bench] ${name.padEnd(25)} ` +
      `prefix=${prefix ? `${prefix.calories.toFixed(0)}kcal` : 'miss'} (${prefixMs.toFixed(1)}ms) | ` +
      `bm25=${bm25 ? `${bm25.calories.toFixed(0)}kcal` : 'miss'} (${bm25Ms.toFixed(1)}ms) | ` +
      `vec=${vecMatch ? `${vecMatch.calories.toFixed(0)}kcal` : embSvc.ready ? 'miss' : 'no-model'} (${vecMs.toFixed(1)}ms)`,
    );
  }

  return results;
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
  let mockReason: 'unavailable' | 'error' | 'empty' | undefined;
  let vlmResult;

  try {
    const status = await geminiNanoModule.checkAvailability();
    if (__DEV__) console.log('[scanFood] availability:', status);
    if (status === 'available') {
      vlmResult = await geminiNanoService.identify(photoUri);
      if (!vlmResult || vlmResult.dishes.length === 0) {
        if (__DEV__) console.log('[scanFood] Gemini Nano returned empty dishes, using mock');
        isMock = true;
        mockReason = 'empty';
        vlmResult = getMockScanResult();
      } else {
        _lastVlmSource = 'gemini-nano';
      }
    } else {
      if (__DEV__) console.log('[scanFood] Gemini Nano unavailable, status:', status);
      isMock = true;
      mockReason = 'unavailable';
      vlmResult = getMockScanResult();
    }
  } catch (err) {
    console.error('[scanFood] Error during identification:', err);
    isMock = true;
    mockReason = 'error';
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

  return { photoUri, dishes: enrichedDishes, isMock, mockReason };
}
