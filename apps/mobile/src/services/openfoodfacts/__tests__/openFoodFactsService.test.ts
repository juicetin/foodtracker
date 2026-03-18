/**
 * Open Food Facts service tests — barcode lookup + text search.
 *
 * Tests use mocked fetch to avoid hitting the real API.
 */

import {
  lookupBarcode,
  searchProducts,
  type OFFProduct,
} from '../openFoodFactsService';

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const mockFetch = jest.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
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

// ---------------------------------------------------------------------------
// lookupBarcode
// ---------------------------------------------------------------------------

describe('lookupBarcode', () => {
  it('returns product data for a valid barcode', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => BARCODE_SUCCESS_RESPONSE,
    });

    const result = await lookupBarcode('5449000000996');

    expect(mockFetch).toHaveBeenCalledWith(
      'https://world.openfoodfacts.org/api/v2/product/5449000000996?fields=code,product_name,brands,quantity,image_front_url,nutriments,serving_size,serving_quantity,nutrition_grades,categories_tags',
      expect.objectContaining({
        headers: expect.objectContaining({
          'User-Agent': expect.stringContaining('Tastimate'),
        }),
      }),
    );

    expect(result).not.toBeNull();
    const product = result!;
    expect(product.barcode).toBe('5449000000996');
    expect(product.name).toBe('Coca-Cola');
    expect(product.brand).toBe('Coca-Cola');
    expect(product.quantity).toBe('330ml');
    expect(product.imageUrl).toBe(BARCODE_SUCCESS_RESPONSE.product.image_front_url);
    expect(product.servingSize).toBe('330ml');
    expect(product.servingQuantityG).toBe(330);
    expect(product.nutritionGrade).toBe('e');

    // Per-100g nutrition
    expect(product.nutrimentsPer100g.calories).toBe(42);
    expect(product.nutrimentsPer100g.protein).toBe(0);
    expect(product.nutrimentsPer100g.carbs).toBe(10.6);
    expect(product.nutrimentsPer100g.fat).toBe(0);
    expect(product.nutrimentsPer100g.fiber).toBe(0);
    expect(product.nutrimentsPer100g.sodium).toBe(0.01);
    expect(product.nutrimentsPer100g.sugar).toBe(10.6);
    expect(product.nutrimentsPer100g.saturatedFat).toBe(0);
  });

  it('returns null for a barcode not in the database', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => BARCODE_NOT_FOUND_RESPONSE,
    });

    const result = await lookupBarcode('0000000000000');
    expect(result).toBeNull();
  });

  it('returns null on network error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const result = await lookupBarcode('5449000000996');
    expect(result).toBeNull();
  });

  it('returns null on non-OK HTTP response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const result = await lookupBarcode('5449000000996');
    expect(result).toBeNull();
  });

  it('handles missing nutriment fields gracefully (defaults to 0)', async () => {
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
          // everything else missing
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
    expect(result!.nutrimentsPer100g.carbs).toBe(0);
    expect(result!.nutrimentsPer100g.fat).toBe(0);
    expect(result!.nutrimentsPer100g.fiber).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// searchProducts
// ---------------------------------------------------------------------------

describe('searchProducts', () => {
  it('returns matching products for a text query', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => SEARCH_RESPONSE,
    });

    const results = await searchProducts('nutella');

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('https://world.openfoodfacts.org/cgi/search.pl?'),
      expect.objectContaining({
        headers: expect.objectContaining({
          'User-Agent': expect.stringContaining('Tastimate'),
        }),
      }),
    );

    // Verify query params
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('search_terms=nutella');
    expect(url).toContain('search_simple=1');
    expect(url).toContain('json=1');

    expect(results).toHaveLength(2);
    expect(results[0].name).toBe('Nutella');
    expect(results[0].brand).toBe('Ferrero');
    expect(results[0].nutrimentsPer100g.calories).toBe(539);
    expect(results[1].name).toBe('Nutella B-ready');
  });

  it('respects page_size parameter', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...SEARCH_RESPONSE, products: [] }),
    });

    await searchProducts('test', 10);

    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('page_size=10');
  });

  it('returns empty array on network error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const results = await searchProducts('nutella');
    expect(results).toEqual([]);
  });

  it('returns empty array on non-OK response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const results = await searchProducts('nutella');
    expect(results).toEqual([]);
  });

  it('handles null nutriment fields in search results', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => SEARCH_RESPONSE,
    });

    const results = await searchProducts('nutella');
    // Second product has null fiber and sodium
    expect(results[1].nutrimentsPer100g.fiber).toBe(0);
    expect(results[1].nutrimentsPer100g.sodium).toBe(0);
  });
});
