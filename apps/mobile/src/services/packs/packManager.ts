/**
 * Generic versioned pack download, cache, and lifecycle management.
 *
 * Handles downloading packs from R2, verifying SHA-256 integrity,
 * storing them locally, and tracking installed packs in the user database.
 *
 * This is the GENERIC pack manager -- handles nutrition DBs now and
 * ML model packs in later phases using the same logic.
 *
 * Phase 1: Both platforms download from R2 (no platform-native delivery).
 * Phase 6: Platform-native delivery (Play Asset Delivery / iOS ODR) as optimization.
 */

import { Paths, File, Directory } from 'expo-file-system';
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
 * Get the filename from a pack's URL.
 */
function getPackFilename(pack: PackEntry): string {
  const urlParts = pack.url.split('/');
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
 */
async function hashFile(fileUri: string): Promise<string> {
  const file = new File(fileUri);
  const base64Content = await file.base64();
  return Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    base64Content
  );
}

export const PackManager = {
  /**
   * Download a pack from R2, verify its integrity, and record it.
   *
   * Uses fetch API for download with manual progress tracking via
   * Content-Length header. Files are written using expo-file-system v19
   * File API.
   *
   * @param pack - Pack entry from the manifest
   * @param onProgress - Progress callback
   * @param apiKey - Optional API key for R2 access
   * @returns The installed pack record
   * @throws If download fails or SHA-256 hash mismatches
   */
  async downloadPack(
    pack: PackEntry,
    onProgress: (progress: DownloadProgress) => void,
    apiKey?: string
  ): Promise<InstalledPack> {
    const packDir = getPackDir(pack);
    const filename = getPackFilename(pack);
    const fileUri = `${packDir}${filename}`;

    // Ensure directory exists
    ensureDirectoryExists(getPacksDir());
    ensureDirectoryExists(`${getPacksDir()}${pack.type}/`);
    ensureDirectoryExists(packDir);

    // Build headers
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers[API_KEY_HEADER] = apiKey;
    }

    // Download file
    const response = await fetch(pack.url, { headers });
    if (!response.ok) {
      throw new Error(`Download failed for pack ${pack.id}: ${response.status}`);
    }

    const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
    const totalExpected = contentLength || pack.sizeBytes;

    // Read the response as ArrayBuffer and convert to Uint8Array for writing
    const arrayBuffer = await response.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);

    // Report progress (complete after data is read)
    onProgress({
      totalBytesWritten: bytes.byteLength,
      totalBytesExpected: totalExpected,
      fraction: 1,
    });

    // Write bytes to file
    const file = new File(fileUri);
    file.write(bytes);

    // Verify SHA-256 hash
    const fileHash = await hashFile(fileUri);
    if (fileHash !== pack.sha256) {
      // Delete the corrupt file
      file.delete();
      throw new Error(
        `SHA-256 hash mismatch for pack ${pack.id}: expected ${pack.sha256}, got ${fileHash}`
      );
    }

    // Record in installed_packs table
    const now = new Date().toISOString();
    const record: InstalledPack = {
      id: pack.id,
      name: pack.name,
      type: pack.type,
      version: pack.version,
      filePath: fileUri,
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
   * Delete a pack: remove the file and the database record.
   */
  async deletePack(packId: string): Promise<void> {
    const pack = await PackManager.getInstalledPack(packId);
    if (pack) {
      const file = new File(pack.filePath);
      if (file.exists) {
        file.delete();
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
    type: row.type as 'nutrition' | 'model',
    version: row.version,
    filePath: row.filePath,
    sizeBytes: row.sizeBytes ?? null,
    sha256: row.sha256 ?? null,
    region: row.region ?? null,
    installedAt: row.installedAt ?? new Date().toISOString(),
    lastChecked: row.lastChecked ?? null,
  };
}
