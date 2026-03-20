/**
 * Tests for galleryScanScheduler: background task registration,
 * foreground drain, and task lifecycle.
 */

// ---------------------------------------------------------------------------
// Mocks — inline jest.fn() inside factory for hoisting
// ---------------------------------------------------------------------------

jest.mock('expo-task-manager', () => ({
  defineTask: jest.fn(),
}));

jest.mock('expo-background-task', () => ({
  BackgroundTaskResult: { Success: 1, Failed: 2 },
  registerTaskAsync: jest.fn(),
  unregisterTaskAsync: jest.fn(),
}));

jest.mock('expo-media-library', () => ({
  requestPermissionsAsync: jest.fn(),
}));

jest.mock('../galleryScanService', () => ({
  discoverNewPhotos: jest.fn(),
}));

jest.mock('../foodClassifier', () => ({
  drainScanQueue: jest.fn(),
}));

// ---------------------------------------------------------------------------
// Imports (after mocks)
// ---------------------------------------------------------------------------

import {
  GALLERY_SCAN_TASK,
  registerGalleryScan,
  unregisterGalleryScan,
  triggerForegroundDrain,
} from '../galleryScanScheduler';

import * as TaskManager from 'expo-task-manager';
import * as BackgroundTask from 'expo-background-task';
import * as MediaLibrary from 'expo-media-library';
import { discoverNewPhotos } from '../galleryScanService';
import { drainScanQueue } from '../foodClassifier';

const mockDefineTask = TaskManager.defineTask as jest.Mock;
const mockRegisterTaskAsync = BackgroundTask.registerTaskAsync as jest.Mock;
const mockUnregisterTaskAsync = BackgroundTask.unregisterTaskAsync as jest.Mock;
const mockRequestPermissionsAsync = MediaLibrary.requestPermissionsAsync as jest.Mock;
const mockDiscoverNewPhotos = discoverNewPhotos as jest.Mock;
const mockDrainScanQueue = drainScanQueue as jest.Mock;

// Capture the task callback registered at module load (before any clearAllMocks)
let backgroundTaskCallback: () => Promise<number>;

beforeAll(() => {
  // defineTask was called at module scope; capture the callback
  expect(mockDefineTask).toHaveBeenCalledTimes(1);
  backgroundTaskCallback = mockDefineTask.mock.calls[0][1];
});

beforeEach(() => {
  // Clear call history but don't reset implementations
  mockRegisterTaskAsync.mockClear();
  mockUnregisterTaskAsync.mockClear();
  mockRequestPermissionsAsync.mockClear();
  mockDiscoverNewPhotos.mockClear();
  mockDrainScanQueue.mockClear();

  mockRequestPermissionsAsync.mockResolvedValue({ status: 'granted' });
  mockDiscoverNewPhotos.mockResolvedValue(5);
  mockDrainScanQueue.mockResolvedValue({ classified: 5, foodPhotos: 2, mealGroups: 1 });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('galleryScanScheduler', () => {
  test('defineTask is called at module load with correct task name', () => {
    expect(mockDefineTask).toHaveBeenCalledWith(
      'TASTIMATE_GALLERY_SCAN',
      expect.any(Function),
    );
  });

  test('GALLERY_SCAN_TASK has correct value', () => {
    expect(GALLERY_SCAN_TASK).toBe('TASTIMATE_GALLERY_SCAN');
  });

  test('background task calls discoverNewPhotos (not drainScanQueue)', async () => {
    const result = await backgroundTaskCallback();

    expect(mockDiscoverNewPhotos).toHaveBeenCalled();
    expect(mockDrainScanQueue).not.toHaveBeenCalled();
    expect(result).toBe(1); // BackgroundTaskResult.Success
  });

  test('background task returns Failed on error', async () => {
    mockDiscoverNewPhotos.mockRejectedValueOnce(new Error('DB error'));
    const result = await backgroundTaskCallback();

    expect(result).toBe(2); // BackgroundTaskResult.Failed
  });

  test('registerGalleryScan registers with 4-hour interval', async () => {
    await registerGalleryScan();

    expect(mockRegisterTaskAsync).toHaveBeenCalledWith(
      'TASTIMATE_GALLERY_SCAN',
      { minimumInterval: 4 * 60 * 60 },
    );
  });

  test('unregisterGalleryScan unregisters task', async () => {
    await unregisterGalleryScan();

    expect(mockUnregisterTaskAsync).toHaveBeenCalledWith('TASTIMATE_GALLERY_SCAN');
  });

  test('triggerForegroundDrain requests permission then discovers and drains', async () => {
    const onProgress = jest.fn();
    const result = await triggerForegroundDrain(onProgress);

    expect(mockRequestPermissionsAsync).toHaveBeenCalled();
    expect(mockDiscoverNewPhotos).toHaveBeenCalled();
    expect(mockDrainScanQueue).toHaveBeenCalledWith({ onProgress });
    expect(result).toEqual({ classified: 5, foodPhotos: 2, mealGroups: 1 });
  });

  test('triggerForegroundDrain throws if permission denied', async () => {
    mockRequestPermissionsAsync.mockResolvedValueOnce({ status: 'denied' });

    await expect(triggerForegroundDrain()).rejects.toThrow(
      'Media library permission not granted',
    );
    expect(mockDiscoverNewPhotos).not.toHaveBeenCalled();
    expect(mockDrainScanQueue).not.toHaveBeenCalled();
  });
});
