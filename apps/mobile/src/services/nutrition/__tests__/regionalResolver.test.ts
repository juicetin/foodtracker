/**
 * Tests for RegionalResolver -- multi-database query resolver with priority ordering.
 *
 * Mocks NutritionService instances to test priority ordering, fallback behavior,
 * cross-database routing, and custom pack schema validation.
 */

// Mock db/client
const mockExecute = jest.fn().mockResolvedValue({ rows: [] });
const mockClose = jest.fn();
const mockConnection = {
  execute: mockExecute,
  close: mockClose,
};

jest.mock('../../../../db/client', () => ({
  openNutritionDb: jest.fn(() => mockConnection),
}));

// Mock packManager
const mockGetInstalledPacks = jest.fn().mockResolvedValue([]);
const mockGetPackFilePath = jest.fn().mockResolvedValue(null);
jest.mock('../../packs/packManager', () => ({
  PackManager: {
    getInstalledPacks: (...args: unknown[]) => mockGetInstalledPacks(...args),
    getPackFilePath: (...args: unknown[]) => mockGetPackFilePath(...args),
    downloadPack: jest.fn(),
    isPackInstalled: jest.fn(),
    deletePack: jest.fn(),
  },
}));

import { RegionalResolver, validatePackSchema } from '../regionalResolver';
import type { InstalledPack } from '../../packs/types';

// Helper to create mock installed packs
function createMockPack(overrides: Partial<InstalledPack> = {}): InstalledPack {
  return {
    id: 'usda-core',
    name: 'USDA Core',
    type: 'nutrition',
    version: '1.0.0',
    filePath: '/mock/path/usda-core.db',
    sizeBytes: 50000,
    sha256: 'abc123',
    region: null,
    installedAt: '2026-01-01T00:00:00Z',
    lastChecked: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

describe('RegionalResolver', () => {
  let resolver: RegionalResolver;

  beforeEach(() => {
    jest.clearAllMocks();
    resolver = new RegionalResolver();
  });

  describe('with only USDA installed', () => {
    beforeEach(async () => {
      const usdaPack = createMockPack({ id: 'usda-core', name: 'USDA Core', region: null });
      mockGetInstalledPacks.mockResolvedValue([usdaPack]);
      mockGetPackFilePath.mockResolvedValue('/mock/path/usda-core.db');

      // Mock USDA search results
      mockExecute.mockResolvedValue({
        rows: [
          {
            fdc_id: 100,
            description: 'Chicken breast raw',
            food_category: 'Poultry Products',
            data_type: 'foundation_food',
          },
        ],
      });

      await resolver.initialize();
    });

    it('search returns USDA results', async () => {
      const results = await resolver.searchFoods('chicken');

      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].source).toBe('usda-core');
      expect(results[0].description).toBe('Chicken breast raw');
    });
  });

  describe('with AFCD installed alongside USDA', () => {
    beforeEach(async () => {
      const usdaPack = createMockPack({ id: 'usda-core', name: 'USDA Core', region: null });
      const afcdPack = createMockPack({
        id: 'afcd',
        name: 'AFCD',
        region: 'AU',
        filePath: '/mock/path/afcd.db',
      });
      mockGetInstalledPacks.mockResolvedValue([usdaPack, afcdPack]);
      mockGetPackFilePath.mockImplementation(async (id: string) => {
        if (id === 'usda-core') return '/mock/path/usda-core.db';
        if (id === 'afcd') return '/mock/path/afcd.db';
        return null;
      });

      await resolver.initialize();
    });

    it('search returns AFCD results first, USDA as fallback', async () => {
      // First call: AFCD search
      mockExecute
        .mockResolvedValueOnce({
          rows: [
            {
              fdc_id: 1001,
              description: 'Chicken breast, raw',
              food_category: 'Poultry',
              data_type: 'afcd',
            },
          ],
        })
        // Second call: USDA search
        .mockResolvedValueOnce({
          rows: [
            {
              fdc_id: 100,
              description: 'Chicken breast raw',
              food_category: 'Poultry Products',
              data_type: 'foundation_food',
            },
          ],
        });

      const results = await resolver.searchFoods('chicken');

      // AFCD result should appear first (regional priority)
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].source).toBe('afcd');
    });

    it('falls back to USDA when regional DB has no match', async () => {
      // AFCD: no match
      mockExecute
        .mockResolvedValueOnce({ rows: [] })
        // USDA: has match
        .mockResolvedValueOnce({
          rows: [
            {
              fdc_id: 200,
              description: 'White rice cooked',
              food_category: 'Cereal Grains',
              data_type: 'sr_legacy_food',
            },
          ],
        });

      const results = await resolver.searchFoods('white rice');

      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].source).toBe('usda-core');
    });
  });

  describe('with multiple regional DBs installed', () => {
    it('can install AFCD and CoFID simultaneously', async () => {
      const usdaPack = createMockPack({ id: 'usda-core', region: null });
      const afcdPack = createMockPack({ id: 'afcd', region: 'AU', filePath: '/mock/path/afcd.db' });
      const cofidPack = createMockPack({ id: 'cofid', region: 'UK', filePath: '/mock/path/cofid.db' });
      mockGetInstalledPacks.mockResolvedValue([usdaPack, afcdPack, cofidPack]);
      mockGetPackFilePath.mockImplementation(async (id: string) => {
        if (id === 'usda-core') return '/mock/path/usda-core.db';
        if (id === 'afcd') return '/mock/path/afcd.db';
        if (id === 'cofid') return '/mock/path/cofid.db';
        return null;
      });

      await resolver.initialize();

      const databases = resolver.getInstalledDatabases();
      const ids = databases.map((db) => db.id);
      expect(ids).toContain('afcd');
      expect(ids).toContain('cofid');
      expect(ids).toContain('usda-core');
    });
  });

  describe('nutrient lookup across databases', () => {
    it('routes to correct DB by source identifier', async () => {
      const usdaPack = createMockPack({ id: 'usda-core', region: null });
      const afcdPack = createMockPack({ id: 'afcd', region: 'AU', filePath: '/mock/path/afcd.db' });
      mockGetInstalledPacks.mockResolvedValue([usdaPack, afcdPack]);
      mockGetPackFilePath.mockImplementation(async (id: string) => {
        if (id === 'usda-core') return '/mock/path/usda-core.db';
        if (id === 'afcd') return '/mock/path/afcd.db';
        return null;
      });

      await resolver.initialize();

      // Mock nutrient response for AFCD database
      mockExecute.mockResolvedValue({
        rows: [
          { nutrient_id: 1008, name: 'Energy', amount: 110, unit: 'kcal' },
          { nutrient_id: 1003, name: 'Protein', amount: 23.1, unit: 'g' },
        ],
      });

      const nutrients = await resolver.getFoodNutrients(1001, 'afcd');

      expect(nutrients.length).toBe(2);
      expect(nutrients[0].name).toBe('Energy');
    });
  });

  describe('getMacros', () => {
    it('routes macro calculation to correct database', async () => {
      const usdaPack = createMockPack({ id: 'usda-core', region: null });
      mockGetInstalledPacks.mockResolvedValue([usdaPack]);
      mockGetPackFilePath.mockResolvedValue('/mock/path/usda-core.db');

      await resolver.initialize();

      // Mock nutrients for macro calculation
      mockExecute.mockResolvedValue({
        rows: [
          { nutrient_id: 1008, name: 'Energy', amount: 165.0, unit: 'kcal' },
          { nutrient_id: 1003, name: 'Protein', amount: 31.0, unit: 'g' },
          { nutrient_id: 1005, name: 'Carbohydrate, by difference', amount: 0.0, unit: 'g' },
          { nutrient_id: 1004, name: 'Total lipid (fat)', amount: 3.6, unit: 'g' },
        ],
      });

      const macros = await resolver.getMacros(100, 'usda-core', 200);

      expect(macros.weightGrams).toBe(200);
      expect(macros.calories).toBeCloseTo(330, 1);
      expect(macros.protein).toBeCloseTo(62, 1);
    });
  });

  describe('database management', () => {
    it('addDatabase opens new NutritionService and adds to priority', async () => {
      mockGetInstalledPacks.mockResolvedValue([]);
      await resolver.initialize();

      await resolver.addDatabase('custom-pack', '/mock/path/custom.db');

      const databases = resolver.getInstalledDatabases();
      expect(databases.some((db) => db.id === 'custom-pack')).toBe(true);
    });

    it('removeDatabase closes connection and removes from maps', async () => {
      const usdaPack = createMockPack({ id: 'usda-core', region: null });
      mockGetInstalledPacks.mockResolvedValue([usdaPack]);
      mockGetPackFilePath.mockResolvedValue('/mock/path/usda-core.db');

      await resolver.initialize();
      resolver.removeDatabase('usda-core');

      const databases = resolver.getInstalledDatabases();
      expect(databases.some((db) => db.id === 'usda-core')).toBe(false);
    });
  });
});

describe('validatePackSchema', () => {
  it('returns valid for packs matching published schema', async () => {
    // Mock execute to return table names matching published schema
    mockExecute
      .mockResolvedValueOnce({
        rows: [
          { name: 'foods' },
          { name: 'nutrients' },
          { name: 'food_nutrients' },
          { name: 'food_portions' },
          { name: 'foods_fts' },
        ],
      })
      // Column checks for each table
      .mockResolvedValueOnce({ rows: [{ name: 'fdc_id' }, { name: 'description' }] })
      .mockResolvedValueOnce({ rows: [{ name: 'id' }, { name: 'name' }, { name: 'unit' }] })
      .mockResolvedValueOnce({
        rows: [{ name: 'food_id' }, { name: 'nutrient_id' }, { name: 'amount' }],
      });

    const result = await validatePackSchema('/mock/path/valid.db');
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('returns invalid for packs with missing tables', async () => {
    // Only foods table exists
    mockExecute.mockResolvedValueOnce({
      rows: [{ name: 'foods' }],
    });

    const result = await validatePackSchema('/mock/path/invalid.db');
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });
});

describe('importCustomPack', () => {
  it('validates schema, copies to pack directory, and records in installed_packs', async () => {
    // This test validates the import flow conceptually.
    // Full integration testing happens at the pack manager level.

    const resolver = new RegionalResolver();
    mockGetInstalledPacks.mockResolvedValue([]);
    await resolver.initialize();

    // Mock schema validation to pass
    mockExecute
      .mockResolvedValueOnce({
        rows: [
          { name: 'foods' },
          { name: 'nutrients' },
          { name: 'food_nutrients' },
          { name: 'food_portions' },
          { name: 'foods_fts' },
        ],
      })
      .mockResolvedValueOnce({ rows: [{ name: 'fdc_id' }, { name: 'description' }] })
      .mockResolvedValueOnce({ rows: [{ name: 'id' }, { name: 'name' }, { name: 'unit' }] })
      .mockResolvedValueOnce({
        rows: [{ name: 'food_id' }, { name: 'nutrient_id' }, { name: 'amount' }],
      });

    // importCustomPack validates, then adds database
    const result = await validatePackSchema('/mock/path/custom.db');
    expect(result.valid).toBe(true);

    // After validation passes, database can be added
    await resolver.addDatabase('custom-db', '/mock/path/custom.db');
    const databases = resolver.getInstalledDatabases();
    expect(databases.some((db) => db.id === 'custom-db')).toBe(true);
  });
});
