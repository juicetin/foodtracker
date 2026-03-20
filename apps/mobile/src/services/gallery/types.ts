/**
 * Type contracts for gallery scanning pipeline.
 *
 * ScanQueueItem maps 1:1 to scan_queue DB rows.
 * ClassifiedPhoto is the subset needed for meal grouping.
 * MealGroup clusters food photos by time+GPS proximity.
 */

// ---------------------------------------------------------------------------
// Scan queue item (DB row shape)
// ---------------------------------------------------------------------------

export interface ScanQueueItem {
  id: number;
  assetId: string;
  uri: string;
  status: 'pending' | 'processing' | 'done' | 'error';
  creationTime: number; // EXIF epoch ms
  latitude?: number | null;
  longitude?: number | null;
  isFood?: boolean | null;
  mealGroupId?: string | null;
  createdAt: string;
  processedAt?: string | null;
}

// ---------------------------------------------------------------------------
// Classified photo (after Gemini Nano food/not-food check)
// ---------------------------------------------------------------------------

export interface ClassifiedPhoto {
  id: number;
  assetId: string;
  uri: string;
  creationTime: number; // epoch ms
  latitude?: number | null;
  longitude?: number | null;
  isFood: boolean;
}

// ---------------------------------------------------------------------------
// Meal group (clustered food photos)
// ---------------------------------------------------------------------------

export interface MealGroup {
  id: string;
  photos: ClassifiedPhoto[];
  firstTimestamp: number;
  lastTimestamp: number;
  location?: {
    latitude: number;
    longitude: number;
  };
}

// ---------------------------------------------------------------------------
// Preferences with defaults
// ---------------------------------------------------------------------------

export interface GalleryScanPreferences {
  /** Max time gap between photos in same meal (ms). Default: 1 hour. */
  mealWindowMs: number;
  /** Max GPS distance between photos in same meal (meters). Default: 150m. */
  gpsProximityM: number;
  /** Number of gallery assets to fetch per pagination chunk. */
  scanChunkSize: number;
  /** How far back to scan on first use (days). */
  firstScanDaysBack: number;
}

export const DEFAULT_SCAN_PREFS: GalleryScanPreferences = {
  mealWindowMs: 3_600_000, // 1 hour
  gpsProximityM: 150,
  scanChunkSize: 50,
  firstScanDaysBack: 30,
};
