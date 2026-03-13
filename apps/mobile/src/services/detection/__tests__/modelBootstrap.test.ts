/**
 * Tests for bundled model loading fallback.
 *
 * When installed_packs has no model entries, modelLoader should fall back
 * to loading models via require() bundled assets.
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
const mockFood101Model = {
  run: jest.fn().mockResolvedValue([new Float32Array(101)]),
  runSync: jest.fn().mockReturnValue([new Float32Array(101)]),
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

describe('modelLoader - bundled model fallback', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    releaseModels();
    selectFromResult = [];
  });

  it('falls back to bundled require() models when installed_packs is empty', async () => {
    // installed_packs returns no model-type rows
    selectFromResult = [];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockBinaryModel)
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel)
      .mockResolvedValueOnce(mockFood101Model);

    const modelSet = await loadModelSet();

    expect(modelSet).toBeDefined();
    expect(modelSet.binary).toBeDefined();
    expect(modelSet.detect).toBeDefined();
    expect(modelSet.classify).toBeDefined();

    // Should have called loadTensorflowModel 4 times for bundled models (3 main + 1 food101)
    expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(4);

    // All calls should use require() number values, not { url: string }
    for (const call of mockLoadTensorflowModel.mock.calls) {
      // The first argument should be a number (require() result) not an object
      expect(typeof call[0]).toBe('number');
    }
  });

  it('returns valid ModelSet with all three models from bundled fallback', async () => {
    selectFromResult = [];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockBinaryModel)
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel)
      .mockResolvedValueOnce(mockFood101Model);

    const modelSet = await loadModelSet();

    // Each model should have run/runSync methods
    expect(typeof modelSet.binary.run).toBe('function');
    expect(typeof modelSet.binary.runSync).toBe('function');
    expect(typeof modelSet.detect.run).toBe('function');
    expect(typeof modelSet.detect.runSync).toBe('function');
    expect(typeof modelSet.classify.run).toBe('function');
    expect(typeof modelSet.classify.runSync).toBe('function');
  });

  it('still uses installed_packs path when packs exist', async () => {
    // installed_packs has all three models
    selectFromResult = [
      { id: 'yolo-binary-v1', name: 'Binary', type: 'model', version: '1.0.0', filePath: '/data/binary.tflite', sizeBytes: 5000000, sha256: 'h1', region: null, installedAt: '2026-01-01', lastChecked: null },
      { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
      { id: 'yolo-classify-v1', name: 'Classify', type: 'model', version: '1.0.0', filePath: '/data/classify.tflite', sizeBytes: 8000000, sha256: 'h3', region: null, installedAt: '2026-01-01', lastChecked: null },
    ];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockBinaryModel)
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel)
      .mockResolvedValueOnce(mockFood101Model);

    const modelSet = await loadModelSet();

    expect(modelSet).toBeDefined();
    // When packs exist, should be called with { url: file://... } pattern
    expect(mockLoadTensorflowModel).toHaveBeenCalledWith(
      expect.objectContaining({ url: expect.stringContaining('file://') }),
      expect.any(String),
    );
  });

  it('caches bundled models (second call returns same instances)', async () => {
    selectFromResult = [];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockBinaryModel)
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel)
      .mockResolvedValueOnce(mockFood101Model);

    const first = await loadModelSet();
    const second = await loadModelSet();

    expect(first).toBe(second);
    expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(4);
  });
});
