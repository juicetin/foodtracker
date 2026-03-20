/**
 * Tests for galleryScanService: gallery enumeration, EXIF extraction, dedup.
 */

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockGetAssetsAsync = jest.fn();
const mockGetAssetInfoAsync = jest.fn();
const mockRequestPermissionsAsync = jest.fn();

jest.mock('expo-media-library', () => ({
  getAssetsAsync: (...args: unknown[]) => mockGetAssetsAsync(...args),
  getAssetInfoAsync: (...args: unknown[]) => mockGetAssetInfoAsync(...args),
  requestPermissionsAsync: (...args: unknown[]) => mockRequestPermissionsAsync(...args),
  MediaType: { photo: 'photo' },
  SortBy: { creationTime: 'creationTime' },
}));

// Mock opsqlite
const mockExecute = jest.fn(() => ({ rows: [], rowsAffected: 0 }));
jest.mock('../../../../db/client', () => ({
  opsqlite: {
    execute: (...args: unknown[]) => mockExecute(...args),
  },
}));

import {
  discoverNewPhotos,
  getLastScanTimestamp,
  setLastScanTimestamp,
  getPendingScanItems,
  markScanItemDone,
} from '../galleryScanService';

describe('galleryScanService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockExecute.mockReturnValue({ rows: [], rowsAffected: 0 });
  });

  describe('discoverNewPhotos', () => {
    it('calls getAssetsAsync with createdAfter filter and chunk size', async () => {
      // Mock last scan timestamp
      mockExecute.mockReturnValueOnce({
        rows: [{ value: '1700000000000' }],
        rowsAffected: 0,
      });

      mockGetAssetsAsync.mockResolvedValue({
        assets: [
          { id: 'asset-1', uri: 'file:///photo1.jpg', creationTime: 1700001000000, width: 4032, height: 3024 },
          { id: 'asset-2', uri: 'file:///photo2.jpg', creationTime: 1700002000000, width: 3024, height: 4032 },
        ],
        hasNextPage: false,
      });

      // Mock getAssetInfoAsync for EXIF
      mockGetAssetInfoAsync.mockResolvedValueOnce({
        location: { latitude: -33.85, longitude: 151.21 },
      });
      mockGetAssetInfoAsync.mockResolvedValueOnce({
        location: null,
      });

      const count = await discoverNewPhotos(50);

      expect(mockGetAssetsAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          first: 50,
          mediaType: 'photo',
          createdAfter: 1700000000000,
        }),
      );
      expect(count).toBe(2);
    });

    it('extracts EXIF location via getAssetInfoAsync for each asset', async () => {
      mockExecute.mockReturnValueOnce({ rows: [], rowsAffected: 0 });

      mockGetAssetsAsync.mockResolvedValue({
        assets: [
          { id: 'asset-1', uri: 'file:///photo1.jpg', creationTime: 1700001000000, width: 4032, height: 3024 },
        ],
        hasNextPage: false,
      });

      mockGetAssetInfoAsync.mockResolvedValue({
        location: { latitude: 40.7128, longitude: -74.006 },
      });

      await discoverNewPhotos();

      expect(mockGetAssetInfoAsync).toHaveBeenCalledWith('asset-1');
    });

    it('uses INSERT OR IGNORE for dedup on assetId', async () => {
      mockExecute.mockReturnValueOnce({ rows: [], rowsAffected: 0 });

      mockGetAssetsAsync.mockResolvedValue({
        assets: [
          { id: 'asset-1', uri: 'file:///photo1.jpg', creationTime: 1700001000000, width: 4032, height: 3024 },
        ],
        hasNextPage: false,
      });

      mockGetAssetInfoAsync.mockResolvedValue({ location: null });

      await discoverNewPhotos();

      // Find the INSERT OR IGNORE call
      const insertCall = mockExecute.mock.calls.find(
        (call) => typeof call[0] === 'string' && call[0].includes('INSERT OR IGNORE'),
      );
      expect(insertCall).toBeTruthy();
    });

    it('handles EXIF extraction errors gracefully', async () => {
      mockExecute.mockReturnValueOnce({ rows: [], rowsAffected: 0 });

      mockGetAssetsAsync.mockResolvedValue({
        assets: [
          { id: 'asset-1', uri: 'file:///photo1.jpg', creationTime: 1700001000000, width: 4032, height: 3024 },
        ],
        hasNextPage: false,
      });

      mockGetAssetInfoAsync.mockRejectedValue(new Error('EXIF read failed'));

      // Should not throw -- graceful fallback
      const count = await discoverNewPhotos();
      expect(count).toBe(1);
    });
  });

  describe('getLastScanTimestamp', () => {
    it('returns stored timestamp from DB', async () => {
      mockExecute.mockReturnValueOnce({
        rows: [{ value: '1700000000000' }],
        rowsAffected: 0,
      });

      const ts = await getLastScanTimestamp();
      expect(ts).toBe(1700000000000);
    });

    it('returns default (30 days back) when no stored timestamp', async () => {
      mockExecute.mockReturnValueOnce({ rows: [], rowsAffected: 0 });

      const ts = await getLastScanTimestamp();
      const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
      // Within 5 seconds of expected
      expect(Math.abs(ts - thirtyDaysAgo)).toBeLessThan(5000);
    });
  });

  describe('setLastScanTimestamp', () => {
    it('persists timestamp via INSERT OR REPLACE', async () => {
      await setLastScanTimestamp(1700000000000);

      const call = mockExecute.mock.calls.find(
        (c) => typeof c[0] === 'string' && c[0].includes('INSERT OR REPLACE'),
      );
      expect(call).toBeTruthy();
    });
  });

  describe('getPendingScanItems', () => {
    it('queries scan_queue for pending items with limit', async () => {
      mockExecute.mockReturnValueOnce({
        rows: [
          { id: 1, asset_id: 'a1', uri: 'file:///p1.jpg', status: 'pending', creation_time: 1700001000000 },
        ],
        rowsAffected: 0,
      });

      const items = await getPendingScanItems(10);
      expect(items).toHaveLength(1);
      expect(items[0].assetId).toBe('a1');
    });
  });

  describe('markScanItemDone', () => {
    it('updates status, isFood, mealGroupId and processedAt', async () => {
      await markScanItemDone(1, true, 'meal-abc');

      const updateCall = mockExecute.mock.calls.find(
        (c) => typeof c[0] === 'string' && c[0].includes('UPDATE scan_queue'),
      );
      expect(updateCall).toBeTruthy();
    });
  });
});
