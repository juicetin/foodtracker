/**
 * Background gallery scan scheduler using expo-task-manager + expo-background-task.
 *
 * IMPORTANT: This module is imported as a side-effect in App.tsx so that
 * TaskManager.defineTask() runs at module load time -- before React renders.
 *
 * Background task: discovers new photos only (MediaLibrary + SQLite).
 * Foreground drain: classifies via Gemini Nano (foreground-only requirement).
 */

import * as TaskManager from 'expo-task-manager';
import * as BackgroundTask from 'expo-background-task';
import * as MediaLibrary from 'expo-media-library';

export const GALLERY_SCAN_TASK = 'TASTIMATE_GALLERY_SCAN';

// Define task at module scope (must be called before React renders)
TaskManager.defineTask(GALLERY_SCAN_TASK, async () => {
  try {
    // Dynamic require to avoid circular deps at module load
    const { discoverNewPhotos } = require('./galleryScanService');
    await discoverNewPhotos();
    // NOTE: Do NOT call drainScanQueue here -- Gemini Nano is foreground-only
    return BackgroundTask.BackgroundTaskResult.Success;
  } catch {
    return BackgroundTask.BackgroundTaskResult.Failed;
  }
});

/**
 * Register periodic background gallery discovery (every 4 hours).
 */
export async function registerGalleryScan(): Promise<void> {
  try {
    await BackgroundTask.registerTaskAsync(GALLERY_SCAN_TASK, {
      minimumInterval: 4 * 60 * 60, // 4 hours
    });
  } catch (err) {
    if (__DEV__) console.warn('Failed to register gallery scan task:', err);
  }
}

/**
 * Unregister the periodic gallery scan task.
 */
export async function unregisterGalleryScan(): Promise<void> {
  try {
    await BackgroundTask.unregisterTaskAsync(GALLERY_SCAN_TASK);
  } catch {
    // Task may not be registered -- ignore
  }
}

/**
 * Trigger a foreground drain: discover new photos then classify + group + import.
 *
 * This must run in the foreground because Gemini Nano requires foreground context.
 */
export async function triggerForegroundDrain(
  onProgress?: (done: number, total: number) => void,
): Promise<{ classified: number; foodPhotos: number; mealGroups: number; entriesCreated: number }> {
  // Request media library permission if not yet granted
  const { status } = await MediaLibrary.requestPermissionsAsync();
  if (status !== 'granted') {
    throw new Error('Media library permission not granted');
  }

  // Catch up on any missed photos
  const { discoverNewPhotos } = require('./galleryScanService');
  await discoverNewPhotos();

  // Classify + group + import
  const { drainScanQueue } = require('./foodClassifier');
  return drainScanQueue({ onProgress });
}
