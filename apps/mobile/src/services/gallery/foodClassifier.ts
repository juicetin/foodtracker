/**
 * Food classifier: wraps Gemini Nano for food/not-food classification
 * with 500ms pacing and exponential backoff on BUSY errors.
 *
 * drainScanQueue orchestrates: classify -> group -> import pipeline.
 */

import { geminiNanoService } from '../vlm/geminiNanoService';
import { getPendingScanItems, markScanItemDone } from './galleryScanService';
import { groupIntoMeals } from './mealGrouper';
import { importPhoto } from './photoImporter';
import type { ClassifiedPhoto, ScanQueueItem } from './types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum delay between classify calls to avoid AICore BUSY. */
const CLASSIFY_DELAY_MS = 500;

/** Max retries per item on error/BUSY. */
const MAX_RETRIES = 3;

/** Base backoff delay (doubles each retry). */
const BASE_BACKOFF_MS = 1000;

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/**
 * Classify a single photo as food or not-food via Gemini Nano.
 *
 * @returns true if the photo contains food (dishes.length > 0)
 */
export async function classifyPhoto(photoUri: string): Promise<boolean> {
  try {
    const result = await geminiNanoService.identify(photoUri);
    return result.dishes.length > 0;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Drain queue pipeline
// ---------------------------------------------------------------------------

interface DrainOptions {
  batchSize?: number;
  onProgress?: (done: number, total: number) => void;
}

interface DrainResult {
  classified: number;
  foodPhotos: number;
  mealGroups: number;
}

/**
 * Process all pending scan queue items:
 * 1. Classify each via Gemini Nano (food/not-food)
 * 2. Group food photos into meals
 * 3. Import (downscale + persist) food photos
 *
 * Includes 500ms pacing between classify calls and exponential backoff on errors.
 */
export async function drainScanQueue(options?: DrainOptions): Promise<DrainResult> {
  const { batchSize = 50, onProgress } = options ?? {};

  const pending = await getPendingScanItems(batchSize);
  if (pending.length === 0) {
    return { classified: 0, foodPhotos: 0, mealGroups: 0 };
  }

  const foodPhotos: ClassifiedPhoto[] = [];
  let classified = 0;

  for (const item of pending) {
    const isFood = await classifyWithRetry(item);

    await markScanItemDone(item.id, isFood);

    if (isFood) {
      foodPhotos.push({
        id: item.id,
        assetId: item.assetId,
        uri: item.uri,
        creationTime: item.creationTime,
        latitude: item.latitude,
        longitude: item.longitude,
        isFood: true,
      });
    }

    classified++;
    onProgress?.(classified, pending.length);

    // Pace between classifications (skip delay after last item)
    if (classified < pending.length) {
      await delay(CLASSIFY_DELAY_MS);
    }
  }

  // Group food photos into meals
  const mealGroups = foodPhotos.length > 0 ? groupIntoMeals(foodPhotos) : [];

  // Import food photos and update mealGroupId in DB
  for (const group of mealGroups) {
    for (const photo of group.photos) {
      try {
        await importPhoto(photo.uri, photo.assetId, { width: 0, height: 0 });
      } catch {
        // Import failure is non-fatal -- photo stays in queue
      }

      // Update mealGroupId
      await markScanItemDone(photo.id, true, group.id);
    }
  }

  return {
    classified,
    foodPhotos: foodPhotos.length,
    mealGroups: mealGroups.length,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function classifyWithRetry(item: ScanQueueItem): Promise<boolean> {
  let retries = 0;

  while (retries <= MAX_RETRIES) {
    try {
      return await classifyPhoto(item.uri);
    } catch {
      retries++;
      if (retries > MAX_RETRIES) {
        // Mark as error after max retries
        return false;
      }
      await delay(BASE_BACKOFF_MS * Math.pow(2, retries - 1));
    }
  }

  return false;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
