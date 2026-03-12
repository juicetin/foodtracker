/**
 * Tests for model loader: loading three-stage pipeline models from
 * PackManager file paths via react-native-fast-tflite.
 */

// ── Mock react-native-fast-tflite ──
const mockBinaryModel = {
  run: jest.fn().mockResolvedValue([new Float32Array(0)]),
  runSync: jest.fn().mockReturnValue([new Float32Array(0)]),
  inputs: [],
  outputs: [],
  delegate: 'default' as const,
};
const mockDetectModel = {
  run: jest.fn().mockResolvedValue([new Float32Array(0)]),
  runSync: jest.fn().mockReturnValue([new Float32Array(0)]),
  inputs: [],
  outputs: [],
  delegate: 'default' as const,
};
const mockClassifyModel = {
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

// ── Mock db/client with drizzle-like chaining ──
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

// ── Mock drizzle-orm ──
jest.mock('drizzle-orm', () => ({
  eq: jest.fn((col: unknown, val: unknown) => ({ col, val, type: 'eq' })),
  and: jest.fn((...args: unknown[]) => ({ args, type: 'and' })),
  like: jest.fn((col: unknown, val: unknown) => ({ col, val, type: 'like' })),
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

import { loadModelSet, getModelSet, releaseModels } from '../modelLoader';

describe('modelLoader', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    releaseModels();
    selectFromResult = [];
  });

  describe('loadModelSet', () => {
    it('loads all three models from installed pack file paths', async () => {
      selectFromResult = [
        { id: 'yolo-binary-v1', name: 'Binary Gate', type: 'model', version: '1.0.0', filePath: '/data/packs/model/yolo-binary-v1/binary.tflite', sizeBytes: 5000000, sha256: 'hash1', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-detect-v1', name: 'Detector', type: 'model', version: '1.0.0', filePath: '/data/packs/model/yolo-detect-v1/detect.tflite', sizeBytes: 10000000, sha256: 'hash2', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-classify-v1', name: 'Classifier', type: 'model', version: '1.0.0', filePath: '/data/packs/model/yolo-classify-v1/classify.tflite', sizeBytes: 8000000, sha256: 'hash3', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel
        .mockResolvedValueOnce(mockBinaryModel)
        .mockResolvedValueOnce(mockDetectModel)
        .mockResolvedValueOnce(mockClassifyModel);

      const modelSet = await loadModelSet();

      expect(modelSet).toBeDefined();
      expect(modelSet.binary).toBe(mockBinaryModel);
      expect(modelSet.detect).toBe(mockDetectModel);
      expect(modelSet.classify).toBe(mockClassifyModel);

      // Verify file:// prefix is used
      expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(3);
      expect(mockLoadTensorflowModel).toHaveBeenCalledWith(
        expect.objectContaining({ url: expect.stringContaining('file://') }),
        expect.any(String),
      );
    });

    it('throws if any required model pack is missing', async () => {
      // Only return binary and detect, missing classify
      selectFromResult = [
        { id: 'yolo-binary-v1', name: 'Binary', type: 'model', version: '1.0.0', filePath: '/data/binary.tflite', sizeBytes: 5000000, sha256: 'h1', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      await expect(loadModelSet()).rejects.toThrow(/model pack/i);
    });

    it('caches loaded models (second call returns same instances)', async () => {
      selectFromResult = [
        { id: 'yolo-binary-v1', name: 'Binary', type: 'model', version: '1.0.0', filePath: '/data/binary.tflite', sizeBytes: 5000000, sha256: 'h1', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-classify-v1', name: 'Classify', type: 'model', version: '1.0.0', filePath: '/data/classify.tflite', sizeBytes: 8000000, sha256: 'h3', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel
        .mockResolvedValueOnce(mockBinaryModel)
        .mockResolvedValueOnce(mockDetectModel)
        .mockResolvedValueOnce(mockClassifyModel);

      const firstCall = await loadModelSet();
      const secondCall = await loadModelSet();

      expect(firstCall).toBe(secondCall);
      // loadTensorflowModel should only be called 3 times total (not 6)
      expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(3);
    });
  });

  describe('getModelSet', () => {
    it('returns null before loading', () => {
      expect(getModelSet()).toBeNull();
    });

    it('returns ModelSet after loading', async () => {
      selectFromResult = [
        { id: 'yolo-binary-v1', name: 'Binary', type: 'model', version: '1.0.0', filePath: '/data/binary.tflite', sizeBytes: 5000000, sha256: 'h1', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
        { id: 'yolo-classify-v1', name: 'Classify', type: 'model', version: '1.0.0', filePath: '/data/classify.tflite', sizeBytes: 8000000, sha256: 'h3', region: null, installedAt: '2026-01-01', lastChecked: null },
      ];

      mockLoadTensorflowModel
        .mockResolvedValueOnce(mockBinaryModel)
        .mockResolvedValueOnce(mockDetectModel)
        .mockResolvedValueOnce(mockClassifyModel);

      await loadModelSet();

      const modelSet = getModelSet();
      expect(modelSet).not.toBeNull();
      expect(modelSet!.binary).toBe(mockBinaryModel);
      expect(modelSet!.detect).toBe(mockDetectModel);
      expect(modelSet!.classify).toBe(mockClassifyModel);
    });
  });
});
