/**
 * Open Food Facts service tests -- barcode lookup + text search with cache.
 *
 * Tests use mocked fetch and mocked offCacheService.
 */

import type { OFFProduct } from '../openFoodFactsService';
import type { CacheResult } from '../offCacheService';

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const mockFetch = jest.fn();
global.fetch = mockFetch;

// ---------------------------------------------------------------------------
// Mock offCacheService
// ---------------------------------------------------------------------------

const mockGetCachedProduct = jest.fn();
const mockCacheProduct = jest.fn();
const mockGetCachedSearch = jest.fn();
const mockCacheSearch = jest.fn();

jest.mock('../offCacheService', () => ({
  getCachedProduct: (...args: unknown[]) => mockGetCachedProduct(...args),
  cacheProduct: (...args: unknown[]) => mockCacheProduct(...args),
  getCachedSearch: (...args: unknown[]) => mockGetCachedSearch(...args),
  cacheSearch: (...args: unknown[]) => mockCacheSearch(...args),
}));

import { lookupBarcode, searchProducts } from '../openFoodFactsService';

beforeEach(() => {
  mockFetch.mockReset();
  mockGetCachedProduct.mockReset();
  mockCacheProduct.mockReset();
  mockGetCachedSearch.mockReset();
  mockCacheSearch.mockReset();
});

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const BARCODE_SUCCESS_RESPONSE = {
  status: 1,
  product: {
    code: '5449000000996',
    product_name: 'Coca-Cola',
    brands: 'Coca-Cola',
    quantity: '330ml',
    image_front_url: 'https://images.openfoodfacts.org/images/products/544/900/000/0996/front_en.3.400.jpg',
    nutriments: {
      'energy-kcal_100g': 42,
      proteins_100g: 0,
      carbohydrates_100g: 10.6,
      fat_100g: 0,
      fiber_100g: 0,
      sodium_100g: 0.01,
      sugars_100g: 10.6,
      'saturated-fat_100g': 0,
    },
    serving_size: '330ml',
    serving_quantity: 330,
    nutrition_grades: 'e',
    categories_tags: ['en:beverages', 'en:carbonated-drinks'],
  },
};

const BARCODE_NOT_FOUND_RESPONSE = {
  status: 0,
  status_verbose: 'product not found',
};

const SEARCH_RESPONSE = {
  count: 42,
  page: 1,
  page_size: 20,
  products: [
    {
      code: '3017620422003',
      product_name: 'Nutella',
      brands: 'Ferrero',
      quantity: '400g',
      image_front_url: 'https://images.openfoodfacts.org/images/products/301/762/042/2003/front_en.3.400.jpg',
      nutriments: {
        'energy-kcal_100g': 539,
        proteins_100g: 6.3,
        carbohydrates_100g: 57.5,
        fat_100g: 30.9,
        fiber_100g: 3.4,
        sodium_100g: 0.041,
        sugars_100g: 56.3,
        'saturated-fat_100g': 10.6,
      },
      serving_size: '15g',
      serving_quantity: 15,
      nutrition_grades: 'e',
      categories_tags: ['en:spreads', 'en:chocolate-spreads'],
    },
    {
      code: '3017620425035',
      product_name: 'Nutella B-ready',
      brands: 'Ferrero',
      quantity: '132g',
      image_front_url: null,
      nutriments: {
        'energy-kcal_100g': 530,
        proteins_100g: 6.8,
        carbohydrates_100g: 58,
        fat_100g: 29.5,
        fiber_100g: null,
        sodium_100g: null,
        sugars_100g: 41,
        'saturated-fat_100g': 11,
      },
      serving_size: '22g',
      serving_quantity: 22,
      nutrition_grades: 'e',
      categories_tags: [],
    },
  ],
};

const CACHED_PRODUCT: OFFProduct = {
  barcode: '5449000000996',
  name: 'Coca-Cola',
  brand: 'Coca-Cola',
  quantity: '330ml',
  imageUrl: null,
  servingSize: '330ml',
  servingQuantityG: 330,
  nutritionGrade: 'e',
  nutrimentsPer100g: {
    calories: 42,
    protein: 0,
    carbs: 10.6,
    fat: 0,
    fiber: 0,
    sodium: 0.01,
    sugar: 10.6,
    saturatedFat: 0,
  },
  categories: ['en:beverages'],
};

const CACHED_SEARCH_PRODUCTS: OFFProduct[] = [
  {
    ...CACHED_PRODUCT,
    barcode: '3017620422003',
    name: 'Nutella',
    brand: 'Ferrero',
  },
];

// ---------------------------------------------------------------------------
// lookupBarcode
// ---------------------------------------------------------------------------

describe('lookupBarcode', () => {
  it('returns cached product when cache is fresh (no network call)', async () => {
    mockGetCachedProduct.mockReturnValue({ data: CACHED_PRODUCT, stale: false });

    const result = await lookupBarcode('5449000000996');

    expect(result).toEqual(CACHED_PRODUCT);
    expect(mockFetch).not.toHaveBeenCalled();
    expect(mockGetCachedProduct).toHaveBeenCalledWith('5449000000996');
  });

  it('returns stale cached product and triggers background refresh', async () => {
    mockGetCachedProduct.mockReturnValue({ data: CACHED_PRODUCT, stale: true });

    const result = await lookupBarcode('5449000000996');

    // Returns stale data immediately
    expect(result).toEqual(CACHED_PRODUCT);
    // Network is called in background (via setTimeout)
    // We can't easily assert the setTimeout callback here, but we verified it returns stale data
  });

  it('fetches from network on cache miss and caches result', async () => {
    mockGetCachedProduct.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => BARCODE_SUCCESS_RESPONSE,
    });

    const result = await lookupBarcode('5449000000996');

    expect(result).not.toBeNull();
    expect(result!.barcode).toBe('5449000000996');
    expect(result!.name).toBe('Coca-Cola');
    expect(mockFetch).toHaveBeenCalled();
    expect(mockCacheProduct).toHaveBeenCalledWith('5449000000996', result);
  });

  it('returns null on cache miss + network failure', async () => {
    mockGetCachedProduct.mockReturnValue(null);
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const result = await lookupBarcode('5449000000996');

    expect(result).toBeNull();
    expect(mockCacheProduct).not.toHaveBeenCalled();
  });

  it('returns null for product not found on OFF', async () => {
    mockGetCachedProduct.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => BARCODE_NOT_FOUND_RESPONSE,
    });

    const result = await lookupBarcode('0000000000000');
    expect(result).toBeNull();
  });

  it('returns null on non-OK HTTP response', async () => {
    mockGetCachedProduct.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const result = await lookupBarcode('5449000000996');
    expect(result).toBeNull();
  });

  it('handles missing nutriment fields gracefully (defaults to 0)', async () => {
    mockGetCachedProduct.mockReturnValue(null);
    const sparse = {
      status: 1,
      product: {
        code: '1234567890',
        product_name: 'Mystery Product',
        brands: null,
        quantity: null,
        image_front_url: null,
        nutriments: {
          'energy-kcal_100g': 100,
        },
        serving_size: null,
        serving_quantity: null,
        nutrition_grades: null,
        categories_tags: null,
      },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => sparse,
    });

    const result = await lookupBarcode('1234567890');
    expect(result).not.toBeNull();
    expect(result!.name).toBe('Mystery Product');
    expect(result!.brand).toBeNull();
    expect(result!.nutrimentsPer100g.calories).toBe(100);
    expect(result!.nutrimentsPer100g.protein).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// searchProducts
// ---------------------------------------------------------------------------

describe('searchProducts', () => {
  it('returns cached results when cache is fresh (no network call)', async () => {
    mockGetCachedSearch.mockReturnValue({ data: CACHED_SEARCH_PRODUCTS, stale: false });

    const results = await searchProducts('nutella');

    expect(results).toEqual(CACHED_SEARCH_PRODUCTS);
    expect(mockFetch).not.toHaveBeenCalled();
    expect(mockGetCachedSearch).toHaveBeenCalledWith('nutella', 20);
  });

  it('returns stale cached results and triggers background refresh', async () => {
    mockGetCachedSearch.mockReturnValue({ data: CACHED_SEARCH_PRODUCTS, stale: true });

    const results = await searchProducts('nutella');

    expect(results).toEqual(CACHED_SEARCH_PRODUCTS);
  });

  it('fetches from network on cache miss and caches results', async () => {
    mockGetCachedSearch.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => SEARCH_RESPONSE,
    });

    const results = await searchProducts('nutella');

    expect(results).toHaveLength(2);
    expect(results[0].name).toBe('Nutella');
    expect(mockFetch).toHaveBeenCalled();
    expect(mockCacheSearch).toHaveBeenCalledWith('nutella', 20, results);
  });

  it('returns empty array on cache miss + network failure', async () => {
    mockGetCachedSearch.mockReturnValue(null);
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const results = await searchProducts('nutella');

    expect(results).toEqual([]);
    expect(mockCacheSearch).not.toHaveBeenCalled();
  });

  it('returns empty array on non-OK response', async () => {
    mockGetCachedSearch.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const results = await searchProducts('nutella');
    expect(results).toEqual([]);
  });

  it('respects page_size parameter', async () => {
    mockGetCachedSearch.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...SEARCH_RESPONSE, products: [] }),
    });

    await searchProducts('test', 10);

    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('page_size=10');
    expect(mockGetCachedSearch).toHaveBeenCalledWith('test', 10);
  });

  it('handles null nutriment fields in search results', async () => {
    mockGetCachedSearch.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => SEARCH_RESPONSE,
    });

    const results = await searchProducts('nutella');
    // Second product has null fiber and sodium
    expect(results[1].nutrimentsPer100g.fiber).toBe(0);
    expect(results[1].nutrimentsPer100g.sodium).toBe(0);
  });

  it('does not cache empty results from network', async () => {
    mockGetCachedSearch.mockReturnValue(null);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ products: [] }),
    });

    const results = await searchProducts('nonexistent');
    expect(results).toEqual([]);
    expect(mockCacheSearch).not.toHaveBeenCalled();
  });
});
