/**
 * Tests for RegionalResolver -- multi-database query resolver with priority ordering.
 *
 * Mocks NutritionService instances to test priority ordering, fallback behavior,
 * cross-database routing, custom pack schema validation, and importCustomPack.
 */

// Mock db/client
const mockExecute = jest.fn().mockResolvedValue({ rows: [] });
const mockClose = jest.fn();
const mockConnection = {
  execute: mockExecute,
  close: mockClose,
};

const mockInsertValues = jest.fn().mockResolvedValue(undefined);
const mockInsert = jest.fn().mockReturnValue({ values: mockInsertValues });

jest.mock('../../../../db/client', () => ({
  openNutritionDb: jest.fn(() => mockConnection),
  userDb: {
    insert: (...args: unknown[]) => mockInsert(...args),
  },
}));

// Mock db/schema
jest.mock('../../../../db/schema', () => ({
  installedPacks: {
    id: 'id',
    name: 'name',
    type: 'type',
    version: 'version',
    filePath: 'file_path',
    sizeBytes: 'size_bytes',
    sha256: 'sha256',
    region: 'region',
    installedAt: 'installed_at',
    lastChecked: 'last_checked',
  },
}));

// Mock expo-file-system (v19 class-based API)
const mockFileWrite = jest.fn();
const mockFileBytes = jest.fn().mockReturnValue(new Uint8Array([1, 2, 3, 4]));
const mockDirCreate = jest.fn();

jest.mock('expo-file-system', () => ({
  Paths: {
    document: { uri: '/mock/documents/' },
  },
  File: jest.fn().mockImplementation(() => ({
    exists: true,
    bytes: mockFileBytes,
    write: mockFileWrite,
  })),
  Directory: jest.fn().mockImplementation(() => ({
    exists: false,
    create: mockDirCreate,
  })),
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
  let resolver: RegionalResolver;

  beforeEach(async () => {
    jest.clearAllMocks();
    resolver = new RegionalResolver();
    mockGetInstalledPacks.mockResolvedValue([]);
    await resolver.initialize();
  });

  it('validates schema, copies file, registers in DB, and adds to resolver', async () => {
    // Mock schema validation to pass (validatePackSchema calls)
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

    const result = await resolver.importCustomPack(
      '/mock/path/custom.db',
      'my-custom-pack',
      'My Custom Pack'
    );

    // Returns a well-formed InstalledPack
    expect(result.id).toBe('my-custom-pack');
    expect(result.name).toBe('My Custom Pack');
    expect(result.type).toBe('nutrition');

    // Database is now visible in installed databases
    const databases = resolver.getInstalledDatabases();
    expect(databases.some((db) => db.id === 'my-custom-pack')).toBe(true);

    // File was copied (Directory created, File.write called)
    const { Directory, File } = require('expo-file-system');
    expect(Directory).toHaveBeenCalled();
    expect(mockDirCreate).toHaveBeenCalled();
    expect(mockFileWrite).toHaveBeenCalled();

    // Record inserted into installed_packs table
    expect(mockInsert).toHaveBeenCalled();
    expect(mockInsertValues).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'my-custom-pack',
        name: 'My Custom Pack',
        type: 'nutrition',
      })
    );
  });

  it('rejects packs that fail schema validation without copying or registering', async () => {
    // Mock schema validation to fail (only foods table exists)
    mockExecute.mockResolvedValueOnce({
      rows: [{ name: 'foods' }],
    });

    await expect(
      resolver.importCustomPack('/mock/path/bad.db', 'bad-pack', 'Bad Pack')
    ).rejects.toThrow('schema validation failed');

    // No file copy should have occurred
    expect(mockFileWrite).not.toHaveBeenCalled();

    // No DB insert should have occurred
    expect(mockInsert).not.toHaveBeenCalled();

    // Database should NOT be in the resolver
    const databases = resolver.getInstalledDatabases();
    expect(databases.some((db) => db.id === 'bad-pack')).toBe(false);
  });
});
