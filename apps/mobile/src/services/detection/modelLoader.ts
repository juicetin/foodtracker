/**
 * Model loader: loads detect-only model from PackManager file paths.
 *
 * Queries the installed_packs table for model-type packs matching the
 * yolo-detect-* naming convention.
 * Model is loaded via react-native-fast-tflite and cached for reuse.
 *
 * When no packs are installed, falls back to the bundled model loaded via
 * require() -- the pre-trained model bundled with the APK.
 *
 * File paths from PackManager use the file:// prefix required by
 * loadTensorflowModel. If the stored path doesn't have the prefix,
 * it is prepended automatically.
 */

import { loadTensorflowModel } from 'react-native-fast-tflite';
import { eq } from 'drizzle-orm';
import { userDb } from '../../../db/client';
import { installedPacks } from '../../../db/schema';
import type { ModelSet } from './types';

// Bundled detect model -- resolved at build time by Metro bundler.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const BUNDLED_DETECT = require('../../../assets/models/detect.tflite');

/** Module-level cache for loaded models. */
let cachedModelSet: ModelSet | null = null;

/**
 * Ensure a file path has the file:// prefix.
 */
function ensureFilePrefix(path: string): string {
  if (path.startsWith('file://')) return path;
  return `file://${path}`;
}

/**
 * Find an installed pack whose ID starts with the given prefix.
 * Returns the filePath or null if not found.
 * Returns null if the installed_packs table doesn't exist yet (no migration run).
 */
async function findModelPackPath(idPrefix: string): Promise<string | null> {
  try {
    const rows = await userDb
      .select()
      .from(installedPacks)
      .where(eq(installedPacks.type, 'model'));

    const match = rows.find((row) => row.id.startsWith(idPrefix));
    return match?.filePath ?? null;
  } catch {
    // Table may not exist yet if migrations haven't run -- fall through to bundled models
    return null;
  }
}

/**
 * Load the detect model from an installed pack file path.
 *
 * Queries installedPacks for model-type packs with IDs matching:
 * - yolo-detect-*  (detection: where is the food?)
 *
 * Model is loaded with the default delegate. On iOS the Expo config
 * plugin enables the CoreML delegate automatically.
 *
 * @returns Cached ModelSet with detect model
 */
export async function loadModelSet(): Promise<ModelSet> {
  // Return cached models if already loaded
  if (cachedModelSet !== null) return cachedModelSet;

  const detectPath = await findModelPackPath('yolo-detect-');

  // If no packs are installed, fall back to bundled model
  if (!detectPath) {
    const detect = await loadTensorflowModel(BUNDLED_DETECT, 'default');
    cachedModelSet = {
      detect: detect as unknown as ModelSet['detect'],
    };
    return cachedModelSet;
  }

  const detect = await loadTensorflowModel(
    { url: ensureFilePrefix(detectPath) },
    'default',
  );

  cachedModelSet = {
    detect: detect as unknown as ModelSet['detect'],
  };

  return cachedModelSet;
}

/**
 * Get the currently cached model set, or null if not yet loaded.
 */
export function getModelSet(): ModelSet | null {
  return cachedModelSet;
}

/**
 * Release cached models. Used for cleanup and testing.
 */
export function releaseModels(): void {
  cachedModelSet = null;
}
