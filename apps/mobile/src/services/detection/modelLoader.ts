/**
 * Model loader: loads three-stage pipeline models from PackManager file paths.
 *
 * Queries the installed_packs table for model-type packs matching the
 * yolo-binary-*, yolo-detect-*, yolo-classify-* naming convention.
 * Models are loaded via react-native-fast-tflite and cached for reuse.
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
 */
async function findModelPackPath(idPrefix: string): Promise<string | null> {
  const rows = await userDb
    .select()
    .from(installedPacks)
    .where(eq(installedPacks.type, 'model'));

  const match = rows.find((row) => row.id.startsWith(idPrefix));
  return match?.filePath ?? null;
}

/**
 * Load all three pipeline models from installed pack file paths.
 *
 * Queries installedPacks for model-type packs with IDs matching:
 * - yolo-binary-*  (binary gate: is this food?)
 * - yolo-detect-*  (detection: where is the food?)
 * - yolo-classify-* (classification: what food is it?)
 *
 * Models are loaded with the default delegate. On iOS the Expo config
 * plugin enables the CoreML delegate automatically.
 *
 * @throws If any required model pack is not installed
 * @returns Cached ModelSet with binary, detect, and classify models
 */
export async function loadModelSet(): Promise<ModelSet> {
  // Return cached models if already loaded
  if (cachedModelSet !== null) return cachedModelSet;

  const binaryPath = await findModelPackPath('yolo-binary-');
  const detectPath = await findModelPackPath('yolo-detect-');
  const classifyPath = await findModelPackPath('yolo-classify-');

  if (!binaryPath || !detectPath || !classifyPath) {
    const missing: string[] = [];
    if (!binaryPath) missing.push('yolo-binary-*');
    if (!detectPath) missing.push('yolo-detect-*');
    if (!classifyPath) missing.push('yolo-classify-*');
    throw new Error(
      `Required model pack(s) not installed: ${missing.join(', ')}. Download required.`,
    );
  }

  const [binary, detect, classify] = await Promise.all([
    loadTensorflowModel({ url: ensureFilePrefix(binaryPath) }, 'default'),
    loadTensorflowModel({ url: ensureFilePrefix(detectPath) }, 'default'),
    loadTensorflowModel({ url: ensureFilePrefix(classifyPath) }, 'default'),
  ]);

  cachedModelSet = {
    binary: binary as unknown as ModelSet['binary'],
    detect: detect as unknown as ModelSet['detect'],
    classify: classify as unknown as ModelSet['classify'],
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
