/**
 * Photo importer: downscales gallery photos to 1024px longest edge,
 * JPEG-compresses at 0.8 quality, and copies to persistent app storage.
 *
 * Uses manipulateAsync (legacy API) per project convention (ADR from Phase 02.6).
 */

import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import { randomUUID } from 'expo-crypto';

const MAX_EDGE = 1024;
const JPEG_QUALITY = 0.8;
const IMPORT_DIR = 'gallery-imports';

/**
 * Import a gallery photo into persistent app storage.
 *
 * - Downscales to 1024px longest edge if needed
 * - JPEG-compresses at 0.8 quality
 * - Moves result to ${documentDirectory}/gallery-imports/${uuid}.jpg
 *
 * @returns Persistent file URI (file://...)
 */
export async function importPhoto(
  galleryUri: string,
  _assetId: string,
  dimensions: { width: number; height: number },
): Promise<string> {
  const { width, height } = dimensions;
  const longestEdge = Math.max(width, height);

  // Build manipulation actions
  const actions: Array<{ resize: { width: number; height: number } }> = [];

  if (longestEdge > MAX_EDGE) {
    const scale = MAX_EDGE / longestEdge;
    actions.push({
      resize: {
        width: Math.round(width * scale),
        height: Math.round(height * scale),
      },
    });
  }

  const result = await manipulateAsync(galleryUri, actions, {
    compress: JPEG_QUALITY,
    format: SaveFormat.JPEG,
  });

  // Ensure gallery-imports directory exists
  const importDir = `${FileSystem.documentDirectory}${IMPORT_DIR}`;
  const dirInfo = await FileSystem.getInfoAsync(importDir);
  if (!dirInfo.exists) {
    await FileSystem.makeDirectoryAsync(importDir, { intermediates: true });
  }

  // Move from cache to persistent storage
  const filename = `${randomUUID()}.jpg`;
  const destUri = `${importDir}/${filename}`;
  await FileSystem.moveAsync({ from: result.uri, to: destUri });

  return destUri;
}
