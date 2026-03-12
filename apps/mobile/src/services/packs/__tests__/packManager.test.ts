/**
 * Tests for PackManager and PackManifest services.
 *
 * Mocks: expo-file-system, expo-crypto, op-sqlite (via __mocks__),
 *        and the userDb from db/client.
 */

import type {
  PackEntry,
  PackManifest,
  InstalledPack,
  DownloadProgress,
} from '../types';

// ── Mock expo-file-system ──
const mockDownloadCallback = jest.fn();
const mockDownloadResumable = {
  downloadAsync: jest.fn(),
};

jest.mock('expo-file-system', () => ({
  documentDirectory: '/mock/documents/',
  createDownloadResumable: jest.fn(
    (
      url: string,
      fileUri: string,
      _options: Record<string, unknown>,
      callback: (progress: { totalBytesWritten: number; totalBytesExpectedToWrite: number }) => void
    ) => {
      mockDownloadCallback.mockImplementation(callback);
      return mockDownloadResumable;
    }
  ),
  deleteAsync: jest.fn().mockResolvedValue(undefined),
  getInfoAsync: jest.fn().mockResolvedValue({ exists: true, size: 1024 }),
  makeDirectoryAsync: jest.fn().mockResolvedValue(undefined),
}));

// ── Mock expo-crypto ──
jest.mock('expo-crypto', () => ({
  digestStringAsync: jest.fn().mockResolvedValue('abc123hash'),
  CryptoDigestAlgorithm: { SHA256: 'SHA-256' },
}));

// ── Mock db/client ──
const mockExecute = jest.fn().mockReturnValue({ rows: [] });
const mockUserDbInsert = jest.fn().mockReturnValue({
  values: jest.fn().mockResolvedValue(undefined),
});
const mockUserDbSelect = jest.fn().mockReturnValue({
  from: jest.fn().mockReturnValue({
    where: jest.fn().mockResolvedValue([]),
    then: jest.fn().mockResolvedValue([]),
  }),
});
const mockUserDbDelete = jest.fn().mockReturnValue({
  where: jest.fn().mockResolvedValue(undefined),
});

jest.mock('../../../../db/client', () => ({
  userDb: {
    insert: mockUserDbInsert,
    select: mockUserDbSelect,
    delete: mockUserDbDelete,
  },
}));

// ── Mock drizzle-orm ──
jest.mock('drizzle-orm', () => ({
  eq: jest.fn((col, val) => ({ col, val, type: 'eq' })),
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
    url: '/packs/nutrition/usda-core-1.0.0.db',
    description: 'Core USDA nutrition database',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockDownloadResumable.downloadAsync.mockResolvedValue({
      uri: '/mock/documents/packs/nutrition/usda-core/usda-core-1.0.0.db',
    });
  });

  describe('downloadPack', () => {
    it('downloads file and records in installed_packs table', async () => {
      const onProgress = jest.fn();

      const result = await PackManager.downloadPack(testPack, onProgress);

      expect(result).toBeDefined();
      expect(result.id).toBe('usda-core');
      expect(result.version).toBe('1.0.0');
      expect(mockDownloadResumable.downloadAsync).toHaveBeenCalled();
      expect(mockUserDbInsert).toHaveBeenCalled();
    });
  });

  describe('getInstalledPacks', () => {
    it('returns list from installed_packs table', async () => {
      const mockPacks: InstalledPack[] = [
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

      // Set up the mock chain for select
      const mockWhere = jest.fn().mockResolvedValue(mockPacks);
      const mockFrom = jest.fn().mockReturnValue(mockPacks);
      mockUserDbSelect.mockReturnValue({ from: mockFrom });

      const result = await PackManager.getInstalledPacks();
      expect(Array.isArray(result)).toBe(true);
      expect(mockUserDbSelect).toHaveBeenCalled();
    });
  });

  describe('isPackInstalled', () => {
    it('returns true when pack is installed', async () => {
      const mockPack = {
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
      };

      const mockWhere = jest.fn().mockResolvedValue([mockPack]);
      const mockFrom = jest.fn().mockReturnValue({ where: mockWhere });
      mockUserDbSelect.mockReturnValue({ from: mockFrom });

      const result = await PackManager.isPackInstalled('usda-core');
      expect(result).toBe(true);
    });

    it('returns false when pack is not installed', async () => {
      const mockWhere = jest.fn().mockResolvedValue([]);
      const mockFrom = jest.fn().mockReturnValue({ where: mockWhere });
      mockUserDbSelect.mockReturnValue({ from: mockFrom });

      const result = await PackManager.isPackInstalled('nonexistent');
      expect(result).toBe(false);
    });
  });

  describe('deletePack', () => {
    it('removes file and installed_packs record', async () => {
      // Mock getInstalledPack to return a pack
      const mockPack = {
        id: 'usda-core',
        filePath: '/mock/documents/packs/nutrition/usda-core/usda-core.db',
      };
      const mockWhere = jest.fn().mockResolvedValue([mockPack]);
      const mockFrom = jest.fn().mockReturnValue({ where: mockWhere });
      mockUserDbSelect.mockReturnValue({ from: mockFrom });

      const FileSystem = require('expo-file-system');

      await PackManager.deletePack('usda-core');

      expect(FileSystem.deleteAsync).toHaveBeenCalled();
      expect(mockUserDbDelete).toHaveBeenCalled();
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
      // Mock global fetch
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue(testManifest),
      });

      const result = await fetchManifest('https://example.com/manifest.json', 'test-key');

      expect(result.version).toBe(1);
      expect(result.packs).toHaveLength(3);
      expect(result.packs[0].id).toBe('usda-core');

      // Check API key header was sent
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
          version: '1.0.0', // older than manifest 2.0.0
          filePath: '/path',
          sizeBytes: 50000000,
          sha256: 'oldhash',
          region: null,
          installedAt: '2026-01-01T00:00:00Z',
          lastChecked: null,
        },
      ];

      const updates = getAvailableUpdates(testManifest, installed);

      // usda-core has an update (1.0.0 -> 2.0.0)
      expect(updates.some((p) => p.id === 'usda-core')).toBe(true);
      // afcd is not installed so it should NOT be in updates
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
