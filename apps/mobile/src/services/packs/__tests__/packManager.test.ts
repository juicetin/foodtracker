/**
 * Tests for PackManager and PackManifest services.
 *
 * Mocks: expo-file-system (v19 File/Directory/Paths), expo-file-system/legacy,
 *        expo-crypto, db/client, db/schema, drizzle-orm, global fetch.
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

// ── Mock expo-file-system/legacy (streaming download API) ──
const mockDownloadAsync = jest.fn().mockResolvedValue({ uri: '/mock/downloaded', status: 200 });
const mockCreateDownloadResumable = jest.fn().mockReturnValue({
  downloadAsync: mockDownloadAsync,
});

jest.mock('expo-file-system/legacy', () => ({
  createDownloadResumable: (...args: unknown[]) => mockCreateDownloadResumable(...args),
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
    mmprojFilePath: 'mmproj_file_path',
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

    // Reset createDownloadResumable mock to default (streaming download)
    mockDownloadAsync.mockResolvedValue({ uri: '/mock/downloaded', status: 200 });
    mockCreateDownloadResumable.mockReturnValue({
      downloadAsync: mockDownloadAsync,
    });

    // Mock global fetch (no longer used for pack download, but needed for manifest tests)
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: jest.fn().mockReturnValue('50000000'),
      },
      arrayBuffer: jest.fn().mockResolvedValue(new ArrayBuffer(1024)),
    });
  });

  describe('downloadPack', () => {
    it('downloads file via streaming and records in installed_packs table', async () => {
      const onProgress = jest.fn();

      const result = await PackManager.downloadPack(testPack, onProgress);

      expect(result).toBeDefined();
      expect(result.id).toBe('usda-core');
      expect(result.version).toBe('1.0.0');
      // Uses streaming createDownloadResumable instead of fetch().arrayBuffer()
      expect(mockCreateDownloadResumable).toHaveBeenCalledWith(
        testPack.url,
        expect.stringContaining('usda-core-1.0.0.db'),
        expect.objectContaining({ headers: expect.any(Object) }),
        expect.any(Function)
      );
      expect(mockInsert).toHaveBeenCalled();
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

    it('removes mmproj file for VLM packs', async () => {
      selectFromResult = [
        {
          id: 'smolvlm-256m',
          name: 'SmolVLM 256M',
          type: 'vlm',
          version: '1.0.0',
          filePath: '/mock/documents/packs/vlm/smolvlm-256m/model.gguf',
          mmprojFilePath: '/mock/documents/packs/vlm/smolvlm-256m/mmproj.gguf',
          sizeBytes: 300000000,
          sha256: 'modelhash',
          region: null,
          installedAt: '2026-01-01T00:00:00Z',
          lastChecked: null,
        },
      ];

      await PackManager.deletePack('smolvlm-256m');

      const { File: FileMock } = require('expo-file-system');
      // File should be called for both model and mmproj files
      expect(FileMock).toHaveBeenCalledTimes(2);
      expect(mockDelete).toHaveBeenCalled();
    });
  });

  describe('streaming download', () => {
    it('uses createDownloadResumable instead of fetch().arrayBuffer()', async () => {
      const onProgress = jest.fn();

      await PackManager.downloadPack(testPack, onProgress);

      // Should NOT use fetch().arrayBuffer() (the OOM pattern)
      const fetchMock = global.fetch as jest.Mock;
      if (fetchMock.mock.calls.length > 0) {
        // If fetch was called at all, arrayBuffer should NOT have been called
        const response = await fetchMock.mock.results[0]?.value;
        if (response?.arrayBuffer) {
          expect(response.arrayBuffer).not.toHaveBeenCalled();
        }
      }

      // Should use createDownloadResumable for streaming
      expect(mockCreateDownloadResumable).toHaveBeenCalled();
      expect(mockDownloadAsync).toHaveBeenCalled();
    });

    it('progress callback called during download', async () => {
      // Set up createDownloadResumable to capture and invoke the progress callback
      mockCreateDownloadResumable.mockImplementation(
        (_url: string, _dest: string, _opts: unknown, progressCb: (data: { totalBytesWritten: number; totalBytesExpectedToWrite: number }) => void) => {
          // Simulate progress callback invocation during download
          if (progressCb) {
            progressCb({ totalBytesWritten: 25_000_000, totalBytesExpectedToWrite: 50_000_000 });
            progressCb({ totalBytesWritten: 50_000_000, totalBytesExpectedToWrite: 50_000_000 });
          }
          return {
            downloadAsync: jest.fn().mockResolvedValue({ uri: '/mock/downloaded', status: 200 }),
          };
        }
      );

      const onProgress = jest.fn();
      await PackManager.downloadPack(testPack, onProgress);

      expect(onProgress).toHaveBeenCalledWith(
        expect.objectContaining({
          totalBytesWritten: expect.any(Number),
          totalBytesExpected: expect.any(Number),
          fraction: expect.any(Number),
        })
      );
    });
  });

  describe('VLM paired file downloads', () => {
    const vlmPack: PackEntry = {
      id: 'smolvlm-256m',
      name: 'SmolVLM 256M',
      type: 'vlm',
      version: '1.0.0',
      sizeBytes: 256_000_000,
      sha256: 'abc123hash',
      url: 'https://r2.example.com/packs/vlm/smolvlm-256m/model.gguf',
      mmprojUrl: 'https://r2.example.com/packs/vlm/smolvlm-256m/mmproj.gguf',
      mmprojSha256: 'abc123hash',
      mmprojSizeBytes: 50_000_000,
      description: 'SmolVLM 256M VLM model',
    };

    it('downloads VLM pack with paired files', async () => {
      const onProgress = jest.fn();

      const result = await PackManager.downloadPack(vlmPack, onProgress);

      expect(result).toBeDefined();
      expect(result.id).toBe('smolvlm-256m');
      expect(result.type).toBe('vlm');
      expect(result.mmprojFilePath).toBeDefined();
      expect(result.mmprojFilePath).toContain('mmproj');

      // createDownloadResumable should be called twice: once for model, once for mmproj
      expect(mockCreateDownloadResumable).toHaveBeenCalledTimes(2);
    });

    it('cleans up model file if mmproj download fails', async () => {
      // Make the first download succeed, second fail
      let callCount = 0;
      mockCreateDownloadResumable.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          // Model download succeeds
          return {
            downloadAsync: jest.fn().mockResolvedValue({ uri: '/mock/model', status: 200 }),
          };
        }
        // mmproj download fails
        return {
          downloadAsync: jest.fn().mockResolvedValue(undefined),
        };
      });

      const onProgress = jest.fn();

      await expect(PackManager.downloadPack(vlmPack, onProgress)).rejects.toThrow();

      // Model file should be deleted (cleanup on mmproj failure)
      const { File: FileMock } = require('expo-file-system');
      const fileInstances = FileMock.mock.results.map((r: { value: unknown }) => r.value);
      const deleteCallCount = fileInstances.filter(
        (f: { delete: { mock: { calls: unknown[] } } }) => f.delete.mock.calls.length > 0
      ).length;
      expect(deleteCallCount).toBeGreaterThan(0);
    });

    it('non-VLM packs still download successfully', async () => {
      const onProgress = jest.fn();

      const result = await PackManager.downloadPack(testPack, onProgress);

      expect(result).toBeDefined();
      expect(result.id).toBe('usda-core');
      expect(result.type).toBe('nutrition');
      expect(result.mmprojFilePath).toBeUndefined();

      // Only one download (no mmproj)
      expect(mockCreateDownloadResumable).toHaveBeenCalledTimes(1);
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
