/**
 * OFF cache service tests -- barcode + search caching with stale-while-revalidate.
 *
 * Mocks opsqlite.executeSync to avoid real DB access.
 */

import type { OFFProduct } from '../openFoodFactsService';

// ---------------------------------------------------------------------------
// Mock DB
// ---------------------------------------------------------------------------

const mockExecuteSync = jest.fn();
jest.mock('../../../../db/client', () => ({
  opsqlite: { executeSync: (...args: unknown[]) => mockExecuteSync(...args) },
}));

import {
  getCachedProduct,
  cacheProduct,
  getCachedSearch,
  cacheSearch,
  normalizeQuery,
  isStale,
  PRODUCT_STALE_MS,
  SEARCH_STALE_MS,
} from '../offCacheService';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const PRODUCT: OFFProduct = {
  barcode: '0737628064502',
  name: 'Nature Valley Granola Bar',
  brand: 'General Mills',
  quantity: '42g',
  imageUrl: null,
  servingSize: '42g',
  servingQuantityG: 42,
  nutritionGrade: 'c',
  nutrimentsPer100g: {
    calories: 460,
    protein: 6,
    carbs: 64,
    fat: 20,
    fiber: 3,
    sodium: 0.3,
    sugar: 28,
    saturatedFat: 3,
  },
  categories: ['en:snacks'],
};

const PRODUCT_2: OFFProduct = {
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

beforeEach(() => {
  mockExecuteSync.mockReset();
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe('constants', () => {
  it('PRODUCT_STALE_MS is 7 days', () => {
    expect(PRODUCT_STALE_MS).toBe(7 * 24 * 60 * 60 * 1000);
  });

  it('SEARCH_STALE_MS is 24 hours', () => {
    expect(SEARCH_STALE_MS).toBe(24 * 60 * 60 * 1000);
  });
});

// ---------------------------------------------------------------------------
// isStale
// ---------------------------------------------------------------------------

describe('isStale', () => {
  it('returns false when cached_at is within threshold', () => {
    const recentDate = new Date(Date.now() - 1000).toISOString().replace('Z', '').replace('T', ' ');
    expect(isStale(recentDate, PRODUCT_STALE_MS)).toBe(false);
  });

  it('returns true when cached_at exceeds threshold', () => {
    const oldDate = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString().replace('Z', '').replace('T', ' ');
    expect(isStale(oldDate, PRODUCT_STALE_MS)).toBe(true);
  });

  it('returns true for product after 7 days', () => {
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000 - 1000).toISOString().replace('Z', '').replace('T', ' ');
    expect(isStale(sevenDaysAgo, PRODUCT_STALE_MS)).toBe(true);
  });

  it('returns true for search after 24 hours', () => {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000 - 1000).toISOString().replace('Z', '').replace('T', ' ');
    expect(isStale(oneDayAgo, SEARCH_STALE_MS)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// normalizeQuery
// ---------------------------------------------------------------------------

describe('normalizeQuery', () => {
  it('trims and lowercases query', () => {
    expect(normalizeQuery('  Ramen  ')).toBe('ramen');
  });

  it('handles already normalized query', () => {
    expect(normalizeQuery('ramen')).toBe('ramen');
  });

  it('handles mixed case with spaces', () => {
    expect(normalizeQuery('  Chicken NUGGETS  ')).toBe('chicken nuggets');
  });
});

// ---------------------------------------------------------------------------
// getCachedProduct / cacheProduct
// ---------------------------------------------------------------------------

describe('getCachedProduct', () => {
  it('returns null when no cache entry exists', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    const result = getCachedProduct('0737628064502');
    expect(result).toBeNull();
  });

  it('returns product with stale=false when cache is fresh', () => {
    const freshDate = new Date(Date.now() - 1000).toISOString().replace('Z', '').replace('T', ' ');
    mockExecuteSync.mockReturnValue({
      rows: [{ response_json: JSON.stringify(PRODUCT), cached_at: freshDate }],
    });

    const result = getCachedProduct('0737628064502');
    expect(result).not.toBeNull();
    expect(result!.stale).toBe(false);
    expect(result!.data.barcode).toBe('0737628064502');
    expect(result!.data.name).toBe('Nature Valley Granola Bar');
  });

  it('returns product with stale=true when cache exceeds 7 days', () => {
    const oldDate = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString().replace('Z', '').replace('T', ' ');
    mockExecuteSync.mockReturnValue({
      rows: [{ response_json: JSON.stringify(PRODUCT), cached_at: oldDate }],
    });

    const result = getCachedProduct('0737628064502');
    expect(result).not.toBeNull();
    expect(result!.stale).toBe(true);
    expect(result!.data.name).toBe('Nature Valley Granola Bar');
  });
});

describe('cacheProduct', () => {
  it('executes INSERT OR REPLACE with correct params', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    cacheProduct('0737628064502', PRODUCT);

    expect(mockExecuteSync).toHaveBeenCalledWith(
      expect.stringContaining('INSERT OR REPLACE INTO off_product_cache'),
      expect.arrayContaining(['0737628064502', 'Nature Valley Granola Bar', 'General Mills']),
    );
  });

  it('updates existing row when called with same barcode', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    cacheProduct('0737628064502', PRODUCT);
    cacheProduct('0737628064502', { ...PRODUCT, name: 'Updated Name' });

    // Both calls should use INSERT OR REPLACE
    expect(mockExecuteSync).toHaveBeenCalledTimes(2);
    expect(mockExecuteSync.mock.calls[1][1]).toContain('Updated Name');
  });
});

// ---------------------------------------------------------------------------
// getCachedSearch / cacheSearch
// ---------------------------------------------------------------------------

describe('getCachedSearch', () => {
  it('returns null when no cache entry exists', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    const result = getCachedSearch('ramen', 20);
    expect(result).toBeNull();
  });

  it('returns products with stale=false when cache is fresh', () => {
    const freshDate = new Date(Date.now() - 1000).toISOString().replace('Z', '').replace('T', ' ');
    mockExecuteSync.mockReturnValue({
      rows: [{ response_json: JSON.stringify([PRODUCT, PRODUCT_2]), cached_at: freshDate }],
    });

    const result = getCachedSearch('ramen', 20);
    expect(result).not.toBeNull();
    expect(result!.stale).toBe(false);
    expect(result!.data).toHaveLength(2);
    expect(result!.data[0].name).toBe('Nature Valley Granola Bar');
  });

  it('returns products with stale=true when cache exceeds 24 hours', () => {
    const oldDate = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString().replace('Z', '').replace('T', ' ');
    mockExecuteSync.mockReturnValue({
      rows: [{ response_json: JSON.stringify([PRODUCT]), cached_at: oldDate }],
    });

    const result = getCachedSearch('ramen', 20);
    expect(result).not.toBeNull();
    expect(result!.stale).toBe(true);
  });

  it('normalizes search cache key', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    getCachedSearch('  Ramen  ', 20);

    expect(mockExecuteSync).toHaveBeenCalledWith(
      expect.stringContaining('off_search_cache'),
      expect.arrayContaining(['ramen::20']),
    );
  });
});

describe('cacheSearch', () => {
  it('executes INSERT OR REPLACE with normalized key', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    cacheSearch('  Ramen  ', 20, [PRODUCT]);

    expect(mockExecuteSync).toHaveBeenCalledWith(
      expect.stringContaining('INSERT OR REPLACE INTO off_search_cache'),
      expect.arrayContaining(['ramen::20', 'ramen', 20]),
    );
  });

  it('updates existing row when called with same query + pageSize', () => {
    mockExecuteSync.mockReturnValue({ rows: [] });
    cacheSearch('ramen', 20, [PRODUCT]);
    cacheSearch('ramen', 20, [PRODUCT, PRODUCT_2]);

    expect(mockExecuteSync).toHaveBeenCalledTimes(2);
  });
});
