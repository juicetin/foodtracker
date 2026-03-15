/**
 * Tests for bundled model loading fallback.
 *
 * When installed_packs has no model entries, modelLoader should fall back
 * to loading the detect model via require() bundled asset.
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

describe('modelLoader - bundled model fallback (detect-only)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    releaseModels();
    selectFromResult = [];
  });

  it('falls back to bundled require() model when installed_packs is empty', async () => {
    selectFromResult = [];

    mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

    const modelSet = await loadModelSet();

    expect(modelSet).toBeDefined();
    expect(modelSet.detect).toBeDefined();

    // Should have called loadTensorflowModel exactly 1 time (detect only)
    expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(1);

    // Should use require() number value, not { url: string }
    expect(typeof mockLoadTensorflowModel.mock.calls[0][0]).toBe('number');
  });

  it('returns valid ModelSet with detect only (no classify)', async () => {
    selectFromResult = [];

    mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

    const modelSet = await loadModelSet();

    // Detect model should have run/runSync methods
    expect(typeof modelSet.detect.run).toBe('function');
    expect(typeof modelSet.detect.runSync).toBe('function');

    // ModelSet should NOT have classify, binary, or food101 properties
    expect((modelSet as unknown as Record<string, unknown>).classify).toBeUndefined();
    expect((modelSet as unknown as Record<string, unknown>).binary).toBeUndefined();
    expect((modelSet as unknown as Record<string, unknown>).food101).toBeUndefined();
  });

  it('still uses installed_packs path when detect pack exists', async () => {
    selectFromResult = [
      { id: 'yolo-detect-v1', name: 'Detect', type: 'model', version: '1.0.0', filePath: '/data/detect.tflite', sizeBytes: 10000000, sha256: 'h2', region: null, installedAt: '2026-01-01', lastChecked: null },
    ];

    mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

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

    mockLoadTensorflowModel.mockResolvedValueOnce(mockDetectModel);

    const first = await loadModelSet();
    const second = await loadModelSet();

    expect(first).toBe(second);
    expect(mockLoadTensorflowModel).toHaveBeenCalledTimes(1);
  });
});
