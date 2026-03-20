/**
 * Gallery scan service: discovers new photos from the device gallery,
 * extracts EXIF metadata, and manages the scan_queue table.
 *
 * Uses cursor-based pagination via expo-media-library.getAssetsAsync()
 * and INSERT OR IGNORE for assetId-based deduplication.
 */

import * as MediaLibrary from 'expo-media-library';
import { opsqlite } from '../../../db/client';
import type { ScanQueueItem } from './types';
import { DEFAULT_SCAN_PREFS } from './types';

// ---------------------------------------------------------------------------
// Key-value store for scan metadata (reuses user_settings table)
// ---------------------------------------------------------------------------

const LAST_SCAN_KEY = 'gallery_last_scan_timestamp';

/**
 * Get the timestamp of the last successful gallery scan.
 * Returns a default of 30 days back if never scanned.
 */
export async function getLastScanTimestamp(): Promise<number> {
  const result = opsqlite.execute(
    `SELECT value FROM user_settings WHERE id = ?`,
    [LAST_SCAN_KEY],
  );
  if (result.rows.length > 0 && result.rows[0].value) {
    return Number(result.rows[0].value);
  }
  return Date.now() - DEFAULT_SCAN_PREFS.firstScanDaysBack * 24 * 60 * 60 * 1000;
}

/**
 * Persist the last scan timestamp.
 */
export async function setLastScanTimestamp(ts: number): Promise<void> {
  opsqlite.execute(
    `INSERT OR REPLACE INTO user_settings (id, value) VALUES (?, ?)`,
    [LAST_SCAN_KEY, String(ts)],
  );
}

// ---------------------------------------------------------------------------
// Photo discovery
// ---------------------------------------------------------------------------

/**
 * Discover new photos from the device gallery since last scan.
 *
 * - Uses cursor-based pagination (getAssetsAsync)
 * - Extracts EXIF GPS via getAssetInfoAsync (wrapped in try/catch)
 * - INSERT OR IGNORE into scan_queue for deduplication on asset_id
 * - Updates lastScanTimestamp after processing chunk
 *
 * @returns Number of new photos queued
 */
export async function discoverNewPhotos(
  chunkSize: number = DEFAULT_SCAN_PREFS.scanChunkSize,
): Promise<number> {
  const lastTs = await getLastScanTimestamp();

  const result = await MediaLibrary.getAssetsAsync({
    first: chunkSize,
    mediaType: 'photo' as MediaLibrary.MediaType,
    sortBy: [MediaLibrary.SortBy.creationTime],
    createdAfter: lastTs,
  });

  let count = 0;
  let maxCreationTime = lastTs;

  for (const asset of result.assets) {
    // Extract EXIF location (may fail for some photos)
    let latitude: number | null = null;
    let longitude: number | null = null;

    try {
      const info = await MediaLibrary.getAssetInfoAsync(asset.id);
      if (info?.location) {
        latitude = info.location.latitude;
        longitude = info.location.longitude;
      }
    } catch {
      // EXIF extraction failure is non-fatal -- photo queued without GPS
    }

    opsqlite.execute(
      `INSERT OR IGNORE INTO scan_queue (asset_id, uri, status, creation_time, latitude, longitude)
       VALUES (?, ?, 'pending', ?, ?, ?)`,
      [asset.id, asset.uri, asset.creationTime, latitude, longitude],
    );

    count++;
    if (asset.creationTime > maxCreationTime) {
      maxCreationTime = asset.creationTime;
    }
  }

  // Update last scan timestamp
  if (count > 0) {
    await setLastScanTimestamp(maxCreationTime);
  }

  return count;
}

// ---------------------------------------------------------------------------
// Queue management
// ---------------------------------------------------------------------------

/**
 * Get pending scan items from the queue.
 */
export async function getPendingScanItems(
  limit: number = 50,
): Promise<ScanQueueItem[]> {
  const result = opsqlite.execute(
    `SELECT id, asset_id, uri, status, creation_time, latitude, longitude,
            is_food, meal_group_id, created_at, processed_at
     FROM scan_queue WHERE status = 'pending' ORDER BY creation_time ASC LIMIT ?`,
    [limit],
  );

  return result.rows.map((row: Record<string, unknown>) => ({
    id: row.id as number,
    assetId: row.asset_id as string,
    uri: row.uri as string,
    status: row.status as ScanQueueItem['status'],
    creationTime: row.creation_time as number,
    latitude: row.latitude as number | null,
    longitude: row.longitude as number | null,
    isFood: row.is_food != null ? Boolean(row.is_food) : null,
    mealGroupId: row.meal_group_id as string | null,
    createdAt: row.created_at as string,
    processedAt: row.processed_at as string | null,
  }));
}

/**
 * Mark a scan queue item as done with classification result.
 */
export async function markScanItemDone(
  id: number,
  isFood: boolean,
  mealGroupId?: string,
): Promise<void> {
  opsqlite.execute(
    `UPDATE scan_queue SET status = 'done', is_food = ?, meal_group_id = ?,
     processed_at = datetime('now') WHERE id = ?`,
    [isFood ? 1 : 0, mealGroupId ?? null, id],
  );
}

/**
 * Insert a scan queue item (used internally by discoverNewPhotos).
 * Exported for testing.
 */
export function insertScanQueueItem(
  assetId: string,
  uri: string,
  creationTime: number,
  latitude: number | null,
  longitude: number | null,
): void {
  opsqlite.execute(
    `INSERT OR IGNORE INTO scan_queue (asset_id, uri, status, creation_time, latitude, longitude)
     VALUES (?, ?, 'pending', ?, ?, ?)`,
    [assetId, uri, creationTime, latitude, longitude],
  );
}
