/**
 * Gallery scanning barrel export.
 *
 * Re-exports all public APIs from the gallery scanning service layer.
 */

// Type contracts
export type {
  ScanQueueItem,
  ClassifiedPhoto,
  MealGroup,
  GalleryScanPreferences,
} from './types';
export { DEFAULT_SCAN_PREFS } from './types';

// Gallery scan service (discovery + queue management)
export {
  discoverNewPhotos,
  getLastScanTimestamp,
  setLastScanTimestamp,
  getPendingScanItems,
  markScanItemDone,
  insertScanQueueItem,
} from './galleryScanService';

// Food classifier (Gemini Nano food/not-food)
export { classifyPhoto, drainScanQueue } from './foodClassifier';

// Meal grouper (temporal + GPS clustering)
export { groupIntoMeals, haversineDistance } from './mealGrouper';

// Photo importer (downscale + persist)
export { importPhoto } from './photoImporter';
