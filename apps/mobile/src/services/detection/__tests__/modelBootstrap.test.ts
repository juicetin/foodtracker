/**
 * Tests for bundled model loading fallback.
 *
 * When installed_packs has no model entries, modelLoader should fall back
 * to loading models via require() bundled assets.
 *
 * Updated for two-model loading (detect + classify only, no binary or food101).
 */

// ── Mock react-native-fast-tflite ──
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

describe('modelLoader - bundled model fallback (2-model)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    releaseModels();
    selectFromResult = [];
  });

  it('falls back to bundled require() models when installed_packs is empty', async () => {
    // installed_packs returns no model-type rows
    selectFromResult = [];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel);

    const modelSet = await loadModelSet();

    expect(modelSet).toBeDefined();
    expect(modelSet.detect).toBeDefined();
    expect(modelSet.classify).toBeDefined();

    // Should have called loadTensorflowModel exactly 2 times (detect + classify)
    expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(2);

    // All calls should use require() number values, not { url: string }
    for (const call of mockLoadTensorflowModel.mock.calls) {
      expect(typeof call[0]).toBe('number');
    }
  });

  it('returns valid ModelSet with exactly 2 models (detect + classify)', async () => {
    selectFromResult = [];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel);

    const modelSet = await loadModelSet();

    // Each model should have run/runSync methods
    expect(typeof modelSet.detect.run).toBe('function');
    expect(typeof modelSet.detect.runSync).toBe('function');
    expect(typeof modelSet.classify.run).toBe('function');
    expect(typeof modelSet.classify.runSync).toBe('function');

    // ModelSet should NOT have binary or food101 properties
    expect((modelSet as Record<string, unknown>).binary).toBeUndefined();
    expect((modelSet as Record<string, unknown>).food101).toBeUndefined();
  });

  it('still uses installed_packs path when packs exist', async () => {
    // installed_packs has detect and classify models
    selectFromResult = [
      { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
      { id: 'efficientnet-classify-v1', name: 'Classify', type: 'model', version: '1.0.0', filePath: '/data/classify.tflite', sizeBytes: 4000000, sha256: 'h3', region: null, installedAt: '2026-01-01', lastChecked: null },
    ];

    mockLoadTensorflowModel
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel);

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
      .mockResolvedValueOnce(mockDetectModel)
      .mockResolvedValueOnce(mockClassifyModel);

    const first = await loadModelSet();
    const second = await loadModelSet();

    expect(first).toBe(second);
    expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(2);
  });
});
