/**
 * Generic versioned pack download, cache, and lifecycle management.
 *
 * Handles downloading packs from R2, verifying SHA-256 integrity,
 * storing them locally, and tracking installed packs in the user database.
 *
 * This is the GENERIC pack manager -- handles nutrition DBs, ML model packs,
 * and VLM packs (paired model + mmproj files) using the same logic.
 *
 * Downloads use expo-file-system/legacy createDownloadResumable for streaming
 * to disk, avoiding OOM on large files (300MB+ VLM models).
 *
 * Phase 1: Both platforms download from R2 (no platform-native delivery).
 * Phase 6: AI pack resolution added -- checks Play for On-Device AI before R2 fallback.
 */

import { Platform } from 'react-native';
import { Paths, File, Directory } from 'expo-file-system';
import { createDownloadResumable } from 'expo-file-system/legacy';
import * as Crypto from 'expo-crypto';
import { eq } from 'drizzle-orm';
import { userDb } from '../../../db/client';
import { installedPacks } from '../../../db/schema';
import type {
  PackEntry,
  InstalledPack,
  DownloadProgress,
} from './types';

/** API key header name for R2 access (Phase 1 interim auth). */
const API_KEY_HEADER = 'X-API-Key';

/**
 * Get the packs base directory.
 */
function getPacksDir(): string {
  return `${Paths.document.uri}packs/`;
}

/**
 * Get the local storage directory for a pack.
 */
function getPackDir(pack: PackEntry): string {
  return `${getPacksDir()}${pack.type}/${pack.id}/`;
}

/**
 * Get the filename from a URL.
 */
function getFilenameFromUrl(url: string): string {
  const urlParts = url.split('/');
  return urlParts[urlParts.length - 1];
}

/**
 * Ensure a directory exists, creating it and parents if needed.
 */
function ensureDirectoryExists(dirUri: string): void {
  const dir = new Directory(dirUri);
  if (!dir.exists) {
    dir.create();
  }
}

/**
 * Compute SHA-256 hash of a file's content.
 *
 * TODO: For very large files (300MB+ VLM models), this reads the entire file as
 * base64 then hashes the base64 string. A streaming hash solution should be
 * considered to reduce peak memory for integrity verification of large files.
 */
async function hashFile(fileUri: string): Promise<string> {
  const file = new File(fileUri);
  const base64Content = await file.base64();
  return Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    base64Content
  );
}

/**
 * Stream-download a file to disk using createDownloadResumable.
 *
 * Unlike fetch().arrayBuffer() which buffers the entire response in RAM,
 * this streams directly to disk -- critical for 300MB+ VLM downloads.
 *
 * @param url - Remote URL to download
 * @param destPath - Local file URI to write to
 * @param headers - HTTP headers (e.g. API key)
 * @param onProgress - Progress callback with DownloadProgress shape
 * @throws If download fails or returns null/undefined result
 */
async function streamDownloadFile(
  url: string,
  destPath: string,
  headers: Record<string, string>,
  onProgress?: (progress: DownloadProgress) => void
): Promise<void> {
  const resumable = createDownloadResumable(
    url,
    destPath,
    { headers },
    onProgress
      ? (data: { totalBytesWritten: number; totalBytesExpectedToWrite: number }) => {
          const expected = data.totalBytesExpectedToWrite > 0 ? data.totalBytesExpectedToWrite : 1;
          onProgress({
            totalBytesWritten: data.totalBytesWritten,
            totalBytesExpected: data.totalBytesExpectedToWrite,
            fraction: data.totalBytesWritten / expected,
          });
        }
      : undefined
  );

  const result = await resumable.downloadAsync();
  if (!result) {
    throw new Error(`Streaming download failed for ${url}: result is null/undefined`);
  }
}

/**
 * Check if a model file is available via Play for On-Device AI pack.
 *
 * Android-only. Uses require() (not static import) for the ai-pack-delivery
 * module to avoid breaking iOS builds where the native module is a no-op stub.
 * Returns null on iOS, on any error, or if the AI pack is not yet completed.
 *
 * @param packId - Not used directly; AI pack is always 'ml-models'
 * @param filename - The expected model filename within the AI pack assets
 * @returns Full path to the model file if available, null otherwise
 */
async function resolveAiPackPath(_packId: string, filename: string): Promise<string | null> {
  if (Platform.OS !== 'android') return null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { aiPackDeliveryModule } = require('../../../modules/ai-pack-delivery/src/aiPackDeliveryModule');
    const status = await aiPackDeliveryModule.getPackStatus('ml-models');
    if (status === 'completed') {
      const basePath = await aiPackDeliveryModule.getPackLocation('ml-models');
      if (basePath) {
        if (__DEV__) console.log(`[PackManager] Using AI pack path: ${basePath}/${filename}`);
        return `${basePath}/${filename}`;
      }
    }
    if (__DEV__) console.log(`[PackManager] AI pack not available (status: ${status}), falling back to R2`);
    return null;
  } catch {
    if (__DEV__) console.log('[PackManager] AI pack resolution failed, falling back to R2');
    return null;
  }
}

export const PackManager = {
  /**
   * Download a pack from R2, verify its integrity, and record it.
   *
   * Uses streaming download via createDownloadResumable to avoid OOM
   * on large files (300MB+ VLM models). For VLM packs, downloads both
   * the model GGUF and companion mmproj GGUF as paired files.
   *
   * @param pack - Pack entry from the manifest
   * @param onProgress - Progress callback
   * @param apiKey - Optional API key for R2 access
   * @returns The installed pack record
   * @throws If download fails, SHA-256 hash mismatches, or VLM mmproj download fails
   */
  async downloadPack(
    pack: PackEntry,
    onProgress: (progress: DownloadProgress) => void,
    apiKey?: string
  ): Promise<InstalledPack> {
    const packDir = getPackDir(pack);
    const filename = getFilenameFromUrl(pack.url);
    const fileUri = `${packDir}${filename}`;

    // Check AI pack availability before R2 download (Android only).
    // If the model is already delivered via Play for On-Device AI, skip R2 entirely.
    const aiPackPath = await resolveAiPackPath(pack.id, filename);
    if (aiPackPath) {
      const now = new Date().toISOString();
      const record: InstalledPack = {
        id: pack.id,
        name: pack.name,
        type: pack.type,
        version: pack.version,
        filePath: aiPackPath,
        sizeBytes: pack.sizeBytes,
        sha256: pack.sha256,
        region: pack.region ?? null,
        installedAt: now,
        lastChecked: now,
      };
      await userDb.insert(installedPacks).values({
        id: record.id,
        name: record.name,
        type: record.type,
        version: record.version,
        filePath: record.filePath,
        mmprojFilePath: null,
        sizeBytes: record.sizeBytes,
        sha256: record.sha256,
        region: record.region,
        installedAt: record.installedAt,
        lastChecked: record.lastChecked,
      });
      return record;
    }

    // Ensure directory exists
    ensureDirectoryExists(getPacksDir());
    ensureDirectoryExists(`${getPacksDir()}${pack.type}/`);
    ensureDirectoryExists(packDir);

    // Build headers
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers[API_KEY_HEADER] = apiKey;
    }

    // For VLM packs with paired files, track combined progress across both downloads.
    // Use HTTP-reported sizes (not config estimates) so progress never exceeds 100%.
    const isVlmPairedPack = pack.type === 'vlm' && pack.mmprojUrl;
    let actualModelSize = 0; // Set from HTTP Content-Length during download

    // Stream download model file
    await streamDownloadFile(
      pack.url,
      fileUri,
      headers,
      isVlmPairedPack
        ? (progress) => {
            // During model download, report partial progress (model portion only).
            // Use HTTP-reported expected size as the model's true size.
            actualModelSize = progress.totalBytesExpected;
            onProgress({
              totalBytesWritten: progress.totalBytesWritten,
              totalBytesExpected: progress.totalBytesExpected, // model-only for now
              fraction: progress.fraction * 0.5, // estimate 50% until we know mmproj size
            });
          }
        : onProgress
    );

    // Verify model SHA-256 hash (skip when sha256 is empty -- e.g. HuggingFace direct downloads)
    if (pack.sha256) {
      const fileHash = await hashFile(fileUri);
      if (fileHash !== pack.sha256) {
        const modelFile = new File(fileUri);
        modelFile.delete();
        throw new Error(
          `SHA-256 hash mismatch for pack ${pack.id}: expected ${pack.sha256}, got ${fileHash}`
        );
      }
    }

    // Handle VLM paired mmproj download
    let mmprojFileUri: string | undefined;
    if (isVlmPairedPack && pack.mmprojUrl) {
      const mmprojFilename = getFilenameFromUrl(pack.mmprojUrl);
      mmprojFileUri = `${packDir}${mmprojFilename}`;

      try {
        await streamDownloadFile(
          pack.mmprojUrl,
          mmprojFileUri,
          headers,
          (progress) => {
            // Use HTTP-reported sizes for accurate combined progress
            const totalCombined = actualModelSize + progress.totalBytesExpected;
            onProgress({
              totalBytesWritten: actualModelSize + progress.totalBytesWritten,
              totalBytesExpected: totalCombined,
              fraction: (actualModelSize + progress.totalBytesWritten) / totalCombined,
            });
          }
        );

        // Verify mmproj SHA-256 if provided
        if (pack.mmprojSha256) {
          const mmprojHash = await hashFile(mmprojFileUri);
          if (mmprojHash !== pack.mmprojSha256) {
            // Clean up both files on integrity failure
            const mmprojFile = new File(mmprojFileUri);
            mmprojFile.delete();
            const modelFile = new File(fileUri);
            modelFile.delete();
            throw new Error(
              `SHA-256 hash mismatch for mmproj of pack ${pack.id}: expected ${pack.mmprojSha256}, got ${mmprojHash}`
            );
          }
        }
      } catch (error) {
        // Clean up model file if mmproj download/verification fails (atomic pair)
        const modelFile = new File(fileUri);
        if (modelFile.exists) {
          modelFile.delete();
        }
        // Also clean up partial mmproj file
        if (mmprojFileUri) {
          const mmprojFile = new File(mmprojFileUri);
          if (mmprojFile.exists) {
            mmprojFile.delete();
          }
        }
        throw error;
      }
    }

    // Record in installed_packs table
    const now = new Date().toISOString();
    const record: InstalledPack = {
      id: pack.id,
      name: pack.name,
      type: pack.type,
      version: pack.version,
      filePath: fileUri,
      ...(mmprojFileUri ? { mmprojFilePath: mmprojFileUri } : {}),
      sizeBytes: pack.sizeBytes,
      sha256: pack.sha256,
      region: pack.region ?? null,
      installedAt: now,
      lastChecked: now,
    };

    await userDb.insert(installedPacks).values({
      id: record.id,
      name: record.name,
      type: record.type,
      version: record.version,
      filePath: record.filePath,
      mmprojFilePath: record.mmprojFilePath ?? null,
      sizeBytes: record.sizeBytes,
      sha256: record.sha256,
      region: record.region,
      installedAt: record.installedAt,
      lastChecked: record.lastChecked,
    });

    return record;
  },

  /**
   * Get all installed packs from the database.
   */
  async getInstalledPacks(): Promise<InstalledPack[]> {
    const rows = await userDb.select().from(installedPacks);
    return rows.map(mapRowToInstalledPack);
  },

  /**
   * Get a single installed pack by ID.
   */
  async getInstalledPack(packId: string): Promise<InstalledPack | null> {
    const rows = await userDb
      .select()
      .from(installedPacks)
      .where(eq(installedPacks.id, packId));
    return rows.length > 0 ? mapRowToInstalledPack(rows[0]) : null;
  },

  /**
   * Check if a pack is installed.
   */
  async isPackInstalled(packId: string): Promise<boolean> {
    const rows = await userDb
      .select()
      .from(installedPacks)
      .where(eq(installedPacks.id, packId));
    return rows.length > 0;
  },

  /**
   * Delete a pack: remove the file(s) and the database record.
   * For VLM packs, also removes the companion mmproj file.
   */
  async deletePack(packId: string): Promise<void> {
    const pack = await PackManager.getInstalledPack(packId);
    if (pack) {
      // Delete main model/data file
      const file = new File(pack.filePath);
      if (file.exists) {
        file.delete();
      }
      // Delete companion mmproj file if present (VLM packs)
      if (pack.mmprojFilePath) {
        const mmprojFile = new File(pack.mmprojFilePath);
        if (mmprojFile.exists) {
          mmprojFile.delete();
        }
      }
    }
    // Remove from database
    await userDb.delete(installedPacks).where(eq(installedPacks.id, packId));
  },

  /**
   * Get the local file path for an installed pack.
   */
  async getPackFilePath(packId: string): Promise<string | null> {
    const pack = await PackManager.getInstalledPack(packId);
    return pack?.filePath ?? null;
  },
};

/**
 * Map a database row to an InstalledPack object.
 */
function mapRowToInstalledPack(row: typeof installedPacks.$inferSelect): InstalledPack {
  return {
    id: row.id,
    name: row.name,
    type: row.type as InstalledPack['type'],
    version: row.version,
    filePath: row.filePath,
    ...(row.mmprojFilePath ? { mmprojFilePath: row.mmprojFilePath } : {}),
    sizeBytes: row.sizeBytes ?? null,
    sha256: row.sha256 ?? null,
    region: row.region ?? null,
    installedAt: row.installedAt ?? new Date().toISOString(),
    lastChecked: row.lastChecked ?? null,
  };
}
