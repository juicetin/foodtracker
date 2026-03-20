/**
 * Tests for foodClassifier: Gemini Nano food/not-food classification + drainScanQueue.
 */

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockIdentify = jest.fn();
const mockIsAvailable = jest.fn();

jest.mock('../../vlm/geminiNanoService', () => ({
  geminiNanoService: {
    identify: (...args: unknown[]) => mockIdentify(...args),
    isAvailable: (...args: unknown[]) => mockIsAvailable(...args),
  },
}));

const mockScanFood = jest.fn();
jest.mock('../../vlm/vlmPipeline', () => ({
  scanFood: (...args: unknown[]) => mockScanFood(...args),
}));

const mockLogScanResult = jest.fn();
jest.mock('../../../store/useFoodLogStore', () => ({
  useFoodLogStore: {
    getState: () => ({
      logScanResult: mockLogScanResult,
    }),
  },
}));

const mockGetPendingScanItems = jest.fn();
const mockMarkScanItemDone = jest.fn();
const mockSetLastScanTimestamp = jest.fn();

jest.mock('../galleryScanService', () => ({
  getPendingScanItems: (...args: unknown[]) => mockGetPendingScanItems(...args),
  markScanItemDone: (...args: unknown[]) => mockMarkScanItemDone(...args),
  setLastScanTimestamp: (...args: unknown[]) => mockSetLastScanTimestamp(...args),
}));

const mockGroupIntoMeals = jest.fn();
jest.mock('../mealGrouper', () => ({
  groupIntoMeals: (...args: unknown[]) => mockGroupIntoMeals(...args),
}));

const mockImportPhoto = jest.fn();
jest.mock('../photoImporter', () => ({
  importPhoto: (...args: unknown[]) => mockImportPhoto(...args),
}));

import { classifyPhoto, drainScanQueue } from '../foodClassifier';
import type { ScanQueueItem } from '../types';

// Helper to create a pending scan item
function makeScanItem(overrides: Partial<ScanQueueItem> = {}): ScanQueueItem {
  return {
    id: 1,
    assetId: 'asset-1',
    uri: 'file:///photo.jpg',
    status: 'pending' as const,
    creationTime: Date.now(),
    latitude: null,
    longitude: null,
    isFood: null,
    mealGroupId: null,
    createdAt: new Date().toISOString(),
    processedAt: null,
    ...overrides,
  };
}

describe('classifyPhoto', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns true when Gemini Nano identifies food (dishes.length > 0)', async () => {
    mockIdentify.mockResolvedValue({
      dishes: [{ name: 'Pad Thai', cuisine: 'Thai', ingredients: [] }],
    });

    const result = await classifyPhoto('file:///food.jpg');
    expect(result).toBe(true);
    expect(mockIdentify).toHaveBeenCalledWith('file:///food.jpg');
  });

  it('returns false when Gemini Nano returns empty dishes (not food)', async () => {
    mockIdentify.mockResolvedValue({ dishes: [] });

    const result = await classifyPhoto('file:///cat.jpg');
    expect(result).toBe(false);
  });

  it('returns false on error', async () => {
    mockIdentify.mockRejectedValue(new Error('AICore BUSY'));

    const result = await classifyPhoto('file:///broken.jpg');
    expect(result).toBe(false);
  });
});

describe('drainScanQueue', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers({ advanceTimers: true });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('processes pending items, classifies each, and reports counts', async () => {
    const items: ScanQueueItem[] = [
      makeScanItem({ id: 1, assetId: 'food-1', uri: 'file:///food1.jpg', creationTime: 1700001000000 }),
      makeScanItem({ id: 2, assetId: 'cat-1', uri: 'file:///cat.jpg', creationTime: 1700002000000 }),
    ];

    mockGetPendingScanItems.mockResolvedValue(items);
    mockIdentify
      .mockResolvedValueOnce({ dishes: [{ name: 'Rice', cuisine: 'Asian', ingredients: [] }] })
      .mockResolvedValueOnce({ dishes: [] }); // not food

    mockGroupIntoMeals.mockReturnValue([
      { id: 'meal-1', photos: [{ id: 1, assetId: 'food-1', uri: 'file:///food1.jpg', creationTime: 1700001000000, isFood: true }], firstTimestamp: 1700001000000, lastTimestamp: 1700001000000 },
    ]);

    mockImportPhoto.mockResolvedValue('file:///imported.jpg');
    mockMarkScanItemDone.mockResolvedValue(undefined);

    const result = await drainScanQueue();

    expect(result.classified).toBe(2);
    expect(result.foodPhotos).toBe(1);
    expect(result.mealGroups).toBe(1);
    // 2 initial classify calls + 1 meal group update for the food photo
    expect(mockMarkScanItemDone).toHaveBeenCalledTimes(3);
  });

  it('calls onProgress callback after each item', async () => {
    const items = [
      makeScanItem({ id: 1, uri: 'file:///p1.jpg', creationTime: 1700001000000 }),
    ];

    mockGetPendingScanItems.mockResolvedValue(items);
    mockIdentify.mockResolvedValue({ dishes: [] });
    mockGroupIntoMeals.mockReturnValue([]);
    mockMarkScanItemDone.mockResolvedValue(undefined);

    const onProgress = jest.fn();
    await drainScanQueue({ onProgress });

    expect(onProgress).toHaveBeenCalledWith(1, 1);
  });

  it('returns empty results when no pending items', async () => {
    mockGetPendingScanItems.mockResolvedValue([]);

    const result = await drainScanQueue();
    expect(result).toEqual({ classified: 0, foodPhotos: 0, mealGroups: 0, entriesCreated: 0 });
  });

  it('creates diary entries via logScanResult for each meal group', async () => {
    const items: ScanQueueItem[] = [
      makeScanItem({ id: 1, assetId: 'food-1', uri: 'file:///food1.jpg', creationTime: 1700001000000 }),
      makeScanItem({ id: 2, assetId: 'food-2', uri: 'file:///food2.jpg', creationTime: 1700005000000 }),
    ];

    mockGetPendingScanItems.mockResolvedValue(items);
    mockIdentify
      .mockResolvedValueOnce({ dishes: [{ name: 'Rice', cuisine: 'Asian', ingredients: [] }] })
      .mockResolvedValueOnce({ dishes: [{ name: 'Soup', cuisine: null, ingredients: [] }] });

    const mealGroup1 = {
      id: 'meal-1',
      photos: [{ id: 1, assetId: 'food-1', uri: 'file:///food1.jpg', creationTime: 1700001000000, isFood: true }],
      firstTimestamp: 1700001000000,
      lastTimestamp: 1700001000000,
    };
    const mealGroup2 = {
      id: 'meal-2',
      photos: [{ id: 2, assetId: 'food-2', uri: 'file:///food2.jpg', creationTime: 1700005000000, isFood: true }],
      firstTimestamp: 1700005000000,
      lastTimestamp: 1700005000000,
    };
    mockGroupIntoMeals.mockReturnValue([mealGroup1, mealGroup2]);

    mockImportPhoto.mockResolvedValue('file:///imported.jpg');
    mockMarkScanItemDone.mockResolvedValue(undefined);

    const mockScanResult = {
      photoUri: 'file:///food1.jpg',
      dishes: [{ id: 'd1', name: 'Rice', cuisine: 'Asian', photoUri: 'file:///food1.jpg', ingredients: [], portionScale: 1.0 }],
      isMock: false,
    };
    mockScanFood.mockResolvedValue(mockScanResult);
    mockLogScanResult.mockResolvedValue(undefined);

    const result = await drainScanQueue();

    expect(result.entriesCreated).toBe(2);
    expect(mockScanFood).toHaveBeenCalledTimes(2);
    expect(mockLogScanResult).toHaveBeenCalledTimes(2);
  });
});
