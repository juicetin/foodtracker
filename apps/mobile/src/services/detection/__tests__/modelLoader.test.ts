/**
 * Tests for model loader: loading detect-only model from
 * PackManager file paths via react-native-fast-tflite.
 *
 * Updated for single-model loading (detect only, no classify).
 */

// -- Mock react-native-fast-tflite --
const mockDetectModel = {
  run: jest.fn().mockResolvedValue([new Float32Array(0)]),
  runSync: jest.fn().mockReturnValue([new Float32Array(0)]),
  inputs: [],
  outputs: [],
  delegate: 'default' as const,
};

const mockLoadTensorflowModel = jest.fn();

jest.mock('react-native-fast-tflite', () => ({
  loadTensorflowModel: (...args: unknown[]) => mockLoadTensorflowModel(...args),
}));

// -- Mock db/client with drizzle-like chaining --
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
    select: (...args: unknown[]) => mockSelect(...args),
  },
}));

// -- Mock drizzle-orm --
jest.mock('drizzle-orm', () => ({
  eq: jest.fn((col: unknown, val: unknown) => ({ col, val, type: 'eq' })),
  and: jest.fn((...args: unknown[]) => ({ args, type: 'and' })),
  like: jest.fn((col: unknown, val: unknown) => ({ col, val, type: 'like' })),
  sql: jest.fn(),
}));

// -- Mock db/schema --
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

import { loadModelSet, getModelSet, releaseModels } from '../modelLoader';

describe('modelLoader', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    releaseModels();
    selectFromResult = [];
  });

  describe('loadModelSet', () => {
    it('loads 1 model from installed pack (detect only)', async () => {
      selectFromResult = [
        { id: 'yolo-detect-v1', name: 'Detector', type: 'model', version: '1.0.0', filePath: '/data/packs/model/yolo-detect-v1/detect.tflite', sizeBytes: 10000000, sha256: 'hash2', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

      const modelSet = await loadModelSet();

      expect(modelSet).toBeDefined();
      expect(modelSet.detect).toBe(mockDetectModel);

      // loadTensorflowModel should be called exactly once (detect only)
      expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(1);
      expect(mockLoadTensorflowModel).toHaveBeenCalledWith(
        expect.objectContaining({ url: expect.stringContaining('file://') }),
        expect.any(String),
      );
    });

    it('ModelSet has detect only (no classify property)', async () => {
      selectFromResult = [
        { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

      const modelSet = await loadModelSet();

      expect(modelSet.detect).toBeDefined();
      expect((modelSet as unknown as Record<string, unknown>).classify).toBeUndefined();
    });

    it('caches loaded models (second call returns same instances)', async () => {
      selectFromResult = [
        { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

      const firstCall = await loadModelSet();
      const secondCall = await loadModelSet();

      expect(firstCall).toBe(secondCall);
      // loadTensorflowModel should only be called 1 time total (not 2)
      expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(1);
    });
  });

  describe('getModelSet', () => {
    it('returns null before loading', () => {
      expect(getModelSet()).toBeNull();
    });

    it('returns ModelSet after loading (detect only)', async () => {
      selectFromResult = [
        { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

      await loadModelSet();

      const modelSet = getModelSet();
      expect(modelSet).not.toBeNull();
      expect(modelSet!.detect).toBe(mockDetectModel);
      expect((modelSet as unknown as Record<string, unknown>).classify).toBeUndefined();
    });
  });
});
