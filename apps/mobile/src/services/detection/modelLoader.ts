/**
 * Model loader: loads three-stage pipeline models from PackManager file paths.
 *
 * Queries the installed_packs table for model-type packs matching the
 * yolo-binary-*, yolo-detect-*, yolo-classify-* naming convention.
 * Models are loaded via react-native-fast-tflite and cached for reuse.
 *
 * When no packs are installed, falls back to bundled models loaded via
 * require() -- these are the pre-trained models bundled with the APK.
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

// Bundled models -- resolved at build time by Metro bundler.
// These are the pre-trained baseline models (AIY Food V1 + YOLO11n COCO).
// eslint-disable-next-line @typescript-eslint/no-var-requires
const BUNDLED_BINARY = require('../../../assets/models/binary.tflite');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const BUNDLED_DETECT = require('../../../assets/models/detect.tflite');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const BUNDLED_CLASSIFY = require('../../../assets/models/classify.tflite');
// Food-101 fallback classifier (MobileNetV1 0.5x int8, 224x224, 101 classes).
// eslint-disable-next-line @typescript-eslint/no-var-requires
const BUNDLED_FOOD101 = require('../../../assets/models/food101.tflite');

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
    // Table may not exist yet if migrations haven't run — fall through to bundled models
    return null;
  }
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

  // If no packs are installed at all, fall back to bundled models
  if (!binaryPath && !detectPath && !classifyPath) {
    const [binary, detect, classify, food101] = await Promise.all([
      loadTensorflowModel(BUNDLED_BINARY, 'default'),
      loadTensorflowModel(BUNDLED_DETECT, 'default'),
      loadTensorflowModel(BUNDLED_CLASSIFY, 'default'),
      loadTensorflowModel(BUNDLED_FOOD101, 'default').catch(() => null),
    ]);
    cachedModelSet = {
      binary: binary as unknown as ModelSet['binary'],
      detect: detect as unknown as ModelSet['detect'],
      classify: classify as unknown as ModelSet['classify'],
      food101: food101
        ? (food101 as unknown as ModelSet['food101'])
        : undefined,
    };
    return cachedModelSet;
  }

  // Partial install: some but not all packs found -- error
  if (!binaryPath || !detectPath || !classifyPath) {
    const missing: string[] = [];
    if (!binaryPath) missing.push('yolo-binary-*');
    if (!detectPath) missing.push('yolo-detect-*');
    if (!classifyPath) missing.push('yolo-classify-*');
    throw new Error(
      `Required model pack(s) not installed: ${missing.join(', ')}. Download required.`,
    );
  }

  const [binary, detect, classify, food101] = await Promise.all([
    loadTensorflowModel({ url: ensureFilePrefix(binaryPath) }, 'default'),
    loadTensorflowModel({ url: ensureFilePrefix(detectPath) }, 'default'),
    loadTensorflowModel({ url: ensureFilePrefix(classifyPath) }, 'default'),
    loadTensorflowModel(BUNDLED_FOOD101, 'default').catch(() => null),
  ]);

  cachedModelSet = {
    binary: binary as unknown as ModelSet['binary'],
    detect: detect as unknown as ModelSet['detect'],
    classify: classify as unknown as ModelSet['classify'],
    food101: food101
      ? (food101 as unknown as ModelSet['food101'])
      : undefined,
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
