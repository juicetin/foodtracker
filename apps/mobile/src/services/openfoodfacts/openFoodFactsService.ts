/**
 * Open Food Facts API service — barcode lookup + text search.
 *
 * Uses OFF API v2 (https://wiki.openfoodfacts.org/API).
 * All nutrition values are per 100g.
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
// Public API
// ---------------------------------------------------------------------------

/**
 * Look up a product by barcode (EAN/UPC).
 * Returns null if the product is not found or on error.
 */
export async function lookupBarcode(barcode: string): Promise<OFFProduct | null> {
  try {
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
 * Search products by text query.
 * Returns an array of matching products (empty on error).
 */
export async function searchProducts(
  query: string,
  pageSize: number = 20,
): Promise<OFFProduct[]> {
  try {
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
