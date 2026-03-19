/**
 * Open Food Facts API service -- barcode lookup + text search.
 *
 * Uses OFF API v2 (https://wiki.openfoodfacts.org/API).
 * All nutrition values are per 100g.
 *
 * Implements stale-while-revalidate via offCacheService:
 *  - Cache hit (fresh): return immediately, skip network
 *  - Cache hit (stale): return immediately, refresh in background
 *  - Cache miss: fetch from network, cache result
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OFFNutriments {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sodium: number;
  sugar: number;
  saturatedFat: number;
}

export interface OFFProduct {
  barcode: string;
  name: string;
  brand: string | null;
  quantity: string | null;
  imageUrl: string | null;
  servingSize: string | null;
  servingQuantityG: number | null;
  nutritionGrade: string | null;
  nutrimentsPer100g: OFFNutriments;
  categories: string[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BASE_URL = 'https://world.openfoodfacts.org';
const PRODUCT_FIELDS = [
  'code',
  'product_name',
  'brands',
  'quantity',
  'image_front_url',
  'nutriments',
  'serving_size',
  'serving_quantity',
  'nutrition_grades',
  'categories_tags',
].join(',');

const HEADERS = {
  'User-Agent': 'Tastimate/1.0 (tastimate-app; contact@tastimate.app)',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function num(val: unknown): number {
  return typeof val === 'number' ? val : 0;
}

function parseProduct(raw: Record<string, unknown>): OFFProduct {
  const nutriments = (raw.nutriments ?? {}) as Record<string, unknown>;
  const categories = raw.categories_tags;

  return {
    barcode: raw.code as string,
    name: (raw.product_name as string) ?? '',
    brand: (raw.brands as string) ?? null,
    quantity: (raw.quantity as string) ?? null,
    imageUrl: (raw.image_front_url as string) ?? null,
    servingSize: (raw.serving_size as string) ?? null,
    servingQuantityG: typeof raw.serving_quantity === 'number' ? raw.serving_quantity : null,
    nutritionGrade: (raw.nutrition_grades as string) ?? null,
    nutrimentsPer100g: {
      calories: num(nutriments['energy-kcal_100g']),
      protein: num(nutriments['proteins_100g']),
      carbs: num(nutriments['carbohydrates_100g']),
      fat: num(nutriments['fat_100g']),
      fiber: num(nutriments['fiber_100g']),
      sodium: num(nutriments['sodium_100g']),
      sugar: num(nutriments['sugars_100g']),
      saturatedFat: num(nutriments['saturated-fat_100g']),
    },
    categories: Array.isArray(categories) ? (categories as string[]) : [],
  };
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

import { offRateLimiter } from './rateLimiter';
import {
  getCachedProduct,
  cacheProduct,
  getCachedSearch,
  cacheSearch,
} from './offCacheService';

// ---------------------------------------------------------------------------
// Network helpers (private -- not exported)
// ---------------------------------------------------------------------------

/**
 * Fetch a product from the OFF API by barcode.
 * Returns null on not found, rate-limited, or error.
 */
async function fetchBarcodeFromNetwork(barcode: string): Promise<OFFProduct | null> {
  try {
    const delay = offRateLimiter.getDelay('product');
    if (delay > 0) await new Promise((r) => setTimeout(r, delay));
    if (!offRateLimiter.canRequest('product')) return null;

    offRateLimiter.recordRequest('product');
    const url = `${BASE_URL}/api/v2/product/${barcode}?fields=${PRODUCT_FIELDS}`;
    const response = await fetch(url, { headers: HEADERS });

    if (!response.ok) return null;

    const data = await response.json();
    if (data.status !== 1 || !data.product) return null;

    return parseProduct(data.product);
  } catch {
    return null;
  }
}

/**
 * Search products from the OFF API by text query.
 * Returns empty array on error.
 */
async function fetchSearchFromNetwork(
  query: string,
  pageSize: number,
): Promise<OFFProduct[]> {
  try {
    const delay = offRateLimiter.getDelay('search');
    if (delay > 0) await new Promise((r) => setTimeout(r, delay));
    if (!offRateLimiter.canRequest('search')) return [];

    offRateLimiter.recordRequest('search');
    const params = new URLSearchParams({
      search_terms: query,
      search_simple: '1',
      json: '1',
      page_size: String(pageSize),
      fields: PRODUCT_FIELDS,
    });

    const url = `${BASE_URL}/cgi/search.pl?${params.toString()}`;
    const response = await fetch(url, { headers: HEADERS });

    if (!response.ok) return [];

    const data = await response.json();
    const products = data.products;
    if (!Array.isArray(products)) return [];

    return products.map(parseProduct);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Public API (cache-wrapped)
// ---------------------------------------------------------------------------

/**
 * Look up a product by barcode (EAN/UPC).
 *
 * Cache-first with stale-while-revalidate:
 *  - Fresh cache hit: return immediately
 *  - Stale cache hit: return immediately, refresh in background
 *  - Cache miss: fetch from network, cache result
 */
export async function lookupBarcode(barcode: string): Promise<OFFProduct | null> {
  // Check cache first
  const cached = getCachedProduct(barcode);

  if (cached && !cached.stale) {
    return cached.data;
  }

  if (cached && cached.stale) {
    // Return stale data immediately, refresh in background
    setTimeout(async () => {
      const fresh = await fetchBarcodeFromNetwork(barcode);
      if (fresh) cacheProduct(barcode, fresh);
    }, 0);
    return cached.data;
  }

  // Cache miss -- fetch from network
  const result = await fetchBarcodeFromNetwork(barcode);
  if (result) {
    cacheProduct(barcode, result);
  }
  return result;
}

/**
 * Search products by text query.
 *
 * Cache-first with stale-while-revalidate:
 *  - Fresh cache hit: return immediately
 *  - Stale cache hit: return immediately, refresh in background
 *  - Cache miss: fetch from network, cache results
 */
export async function searchProducts(
  query: string,
  pageSize: number = 20,
): Promise<OFFProduct[]> {
  // Check cache first
  const cached = getCachedSearch(query, pageSize);

  if (cached && !cached.stale) {
    return cached.data;
  }

  if (cached && cached.stale) {
    // Return stale data immediately, refresh in background
    setTimeout(async () => {
      const fresh = await fetchSearchFromNetwork(query, pageSize);
      if (fresh.length > 0) cacheSearch(query, pageSize, fresh);
    }, 0);
    return cached.data;
  }

  // Cache miss -- fetch from network
  const results = await fetchSearchFromNetwork(query, pageSize);
  if (results.length > 0) {
    cacheSearch(query, pageSize, results);
  }
  return results;
}
