/**
 * Tests for PackManager and PackManifest services.
 *
 * Mocks: expo-file-system (v19 File/Directory/Paths), expo-crypto,
 *        db/client, db/schema, drizzle-orm, global fetch.
 */

import type {
  PackEntry,
  PackManifest,
  InstalledPack,
} from '../types';

// ── Mock expo-file-system (v19 class-based API) ──
const mockFileInstance = {
  exists: true,
  base64: jest.fn().mockReturnValue('base64content'),
  text: jest.fn().mockReturnValue('text'),
  write: jest.fn(),
  delete: jest.fn(),
};

const mockDirInstance = {
  exists: false,
  create: jest.fn(),
};

jest.mock('expo-file-system', () => ({
  Paths: {
    document: { uri: '/mock/documents/' },
    cache: { uri: '/mock/cache/' },
  },
  File: jest.fn().mockImplementation(() => ({ ...mockFileInstance })),
  Directory: jest.fn().mockImplementation(() => ({ ...mockDirInstance })),
}));

// ── Mock expo-crypto ──
jest.mock('expo-crypto', () => ({
  digestStringAsync: jest.fn().mockResolvedValue('abc123hash'),
  CryptoDigestAlgorithm: { SHA256: 'SHA-256' },
}));

// ── Mock db/client with proper drizzle-like chaining ──
const mockInsertValues = jest.fn().mockResolvedValue(undefined);
const mockInsert = jest.fn().mockReturnValue({ values: mockInsertValues });

const mockDeleteWhere = jest.fn().mockResolvedValue(undefined);
const mockDelete = jest.fn().mockReturnValue({ where: mockDeleteWhere });

let selectFromResult: unknown[] = [];
const mockSelectWhere = jest.fn().mockImplementation(() => Promise.resolve(selectFromResult));
const mockSelectFrom = jest.fn().mockImplementation(() => {
  const result = Promise.resolve(selectFromResult);
  (result as unknown as Record<string, unknown>).where = mockSelectWhere;
  return result;
});
const mockSelect = jest.fn().mockReturnValue({ from: mockSelectFrom });

jest.mock('../../../../db/client', () => ({
  userDb: {
    insert: (...args: unknown[]) => mockInsert(...args),
    select: (...args: unknown[]) => mockSelect(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
  },
}));

// ── Mock drizzle-orm ──
jest.mock('drizzle-orm', () => ({
  eq: jest.fn((col: unknown, val: unknown) => ({ col, val, type: 'eq' })),
  and: jest.fn((...args: unknown[]) => ({ args, type: 'and' })),
  sql: jest.fn(),
}));

// ── Mock db/schema ──
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

// Need to import after mocks are set up
import { PackManager } from '../packManager';
import { fetchManifest, getAvailableUpdates, getPacksByType, getPacksByRegion } from '../packManifest';

describe('PackManager', () => {
  const testPack: PackEntry = {
    id: 'usda-core',
    name: 'USDA Core Nutrition',
    type: 'nutrition',
    version: '1.0.0',
    sizeBytes: 50_000_000,
    sha256: 'abc123hash',
    url: 'https://r2.example.com/packs/nutrition/usda-core-1.0.0.db',
    description: 'Core USDA nutrition database',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    selectFromResult = [];
    mockInsertValues.mockResolvedValue(undefined);
    mockInsert.mockReturnValue({ values: mockInsertValues });
    mockDeleteWhere.mockResolvedValue(undefined);
    mockDelete.mockReturnValue({ where: mockDeleteWhere });

    // Mock global fetch for download
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: jest.fn().mockReturnValue('50000000'),
      },
      arrayBuffer: jest.fn().mockResolvedValue(new ArrayBuffer(1024)),
    });
  });

  describe('downloadPack', () => {
    it('downloads file and records in installed_packs table', async () => {
      const onProgress = jest.fn();

      const result = await PackManager.downloadPack(testPack, onProgress);

      expect(result).toBeDefined();
      expect(result.id).toBe('usda-core');
      expect(result.version).toBe('1.0.0');
      expect(global.fetch).toHaveBeenCalledWith(
        testPack.url,
        expect.objectContaining({ headers: expect.any(Object) })
      );
      expect(mockInsert).toHaveBeenCalled();
      expect(onProgress).toHaveBeenCalled();
    });
  });

  describe('getInstalledPacks', () => {
    it('returns list from installed_packs table', async () => {
      selectFromResult = [
        {
          id: 'usda-core',
          name: 'USDA Core',
          type: 'nutrition',
          version: '1.0.0',
          filePath: '/mock/path',
          sizeBytes: 50000000,
          sha256: 'abc123',
          region: null,
          installedAt: '2026-01-01T00:00:00Z',
          lastChecked: null,
        },
      ];

      const result = await PackManager.getInstalledPacks();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(1);
      expect(result[0].id).toBe('usda-core');
      expect(mockSelect).toHaveBeenCalled();
    });
  });

  describe('isPackInstalled', () => {
    it('returns true when pack is installed', async () => {
      selectFromResult = [
        {
          id: 'usda-core',
          name: 'USDA Core',
          type: 'nutrition',
          version: '1.0.0',
          filePath: '/mock/path',
          sizeBytes: 50000000,
          sha256: 'abc123',
          region: null,
          installedAt: '2026-01-01T00:00:00Z',
          lastChecked: null,
        },
      ];

      const result = await PackManager.isPackInstalled('usda-core');
      expect(result).toBe(true);
    });

    it('returns false when pack is not installed', async () => {
      selectFromResult = [];

      const result = await PackManager.isPackInstalled('nonexistent');
      expect(result).toBe(false);
    });
  });

  describe('deletePack', () => {
    it('removes file and installed_packs record', async () => {
      selectFromResult = [
        {
          id: 'usda-core',
          name: 'USDA Core',
          type: 'nutrition',
          version: '1.0.0',
          filePath: '/mock/documents/packs/nutrition/usda-core/usda-core.db',
          sizeBytes: 50000000,
          sha256: 'abc123',
          region: null,
          installedAt: '2026-01-01T00:00:00Z',
          lastChecked: null,
        },
      ];

      await PackManager.deletePack('usda-core');

      const { File: FileMock } = require('expo-file-system');
      expect(FileMock).toHaveBeenCalled();
      expect(mockDelete).toHaveBeenCalled();
    });
  });
});

describe('PackManifest', () => {
  const testManifest: PackManifest = {
    version: 1,
    lastUpdated: '2026-03-01T00:00:00Z',
    packs: [
      {
        id: 'usda-core',
        name: 'USDA Core',
        type: 'nutrition',
        version: '2.0.0',
        sizeBytes: 50_000_000,
        sha256: 'newhash',
        url: '/packs/nutrition/usda-core-2.0.0.db',
        description: 'Updated USDA core',
      },
      {
        id: 'afcd',
        name: 'Australian Food Composition',
        type: 'nutrition',
        version: '1.0.0',
        sizeBytes: 20_000_000,
        sha256: 'afcdhash',
        url: '/packs/nutrition/afcd-1.0.0.db',
        region: 'AU',
        locale: 'en-AU',
        description: 'Australian food data',
      },
      {
        id: 'yolo-food-v1',
        name: 'YOLO Food Detection',
        type: 'model',
        version: '1.0.0',
        sizeBytes: 30_000_000,
        sha256: 'modelhash',
        url: '/packs/model/yolo-food-v1.tflite',
        description: 'Food detection model',
      },
    ],
  };

  describe('fetchManifest', () => {
    it('parses JSON manifest into PackManifest type', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue(testManifest),
      });

      const result = await fetchManifest('https://example.com/manifest.json', 'test-key');

      expect(result.version).toBe(1);
      expect(result.packs).toHaveLength(3);
      expect(result.packs[0].id).toBe('usda-core');

      expect(global.fetch).toHaveBeenCalledWith(
        'https://example.com/manifest.json',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-API-Key': 'test-key',
          }),
        })
      );
    });
  });

  describe('getAvailableUpdates', () => {
    it('compares manifest versions against installed versions', () => {
      const installed: InstalledPack[] = [
        {
          id: 'usda-core',
          name: 'USDA Core',
          type: 'nutrition',
          version: '1.0.0',
          filePath: '/path',
          sizeBytes: 50000000,
          sha256: 'oldhash',
          region: null,
          installedAt: '2026-01-01T00:00:00Z',
          lastChecked: null,
        },
      ];

      const updates = getAvailableUpdates(testManifest, installed);
      expect(updates.some((p) => p.id === 'usda-core')).toBe(true);
      expect(updates.some((p) => p.id === 'afcd')).toBe(false);
    });
  });

  describe('getPacksByType', () => {
    it('filters packs by nutrition type', () => {
      const nutritionPacks = getPacksByType(testManifest, 'nutrition');
      expect(nutritionPacks).toHaveLength(2);
      expect(nutritionPacks.every((p) => p.type === 'nutrition')).toBe(true);
    });

    it('filters packs by model type', () => {
      const modelPacks = getPacksByType(testManifest, 'model');
      expect(modelPacks).toHaveLength(1);
      expect(modelPacks[0].id).toBe('yolo-food-v1');
    });
  });

  describe('getPacksByRegion', () => {
    it('filters nutrition packs by region', () => {
      const auPacks = getPacksByRegion(testManifest, 'AU');
      expect(auPacks).toHaveLength(1);
      expect(auPacks[0].id).toBe('afcd');
    });
  });
});
