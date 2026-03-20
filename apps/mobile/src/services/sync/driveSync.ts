/**
 * Drive file operations via react-native-cloud-storage.
 *
 * All operations follow the token-first pattern: call ensureDriveAccess()
 * before any Drive API call to ensure a valid access token is set.
 *
 * Files are stored in Drive's appdata folder (hidden from user).
 */

import { CloudStorage, CloudStorageScope } from 'react-native-cloud-storage';
import { File } from 'expo-file-system';
import { ensureDriveAccess } from './driveAuth';
import type { SyncManifest, RemoteBackupEntry } from './types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BACKUP_PREFIX = '/backups/';
const MANIFEST_PATH = '/sync-manifest.json';
const SCOPE = CloudStorageScope.AppData;

// ---------------------------------------------------------------------------
// Upload operations
// ---------------------------------------------------------------------------

/**
 * Upload an incremental JSON backup to Drive appdata.
 * Reads file content locally, then writes as text to Drive.
 */
export async function uploadIncremental(
  filename: string,
  localJsonPath: string,
): Promise<void> {
  await ensureDriveAccess();
  const file = new File(localJsonPath);
  const content = await file.text();
  await CloudStorage.writeFile(
    `${BACKUP_PREFIX}${filename}`,
    content,
    SCOPE,
  );
}

/**
 * Upload a full .db backup to Drive appdata.
 * Uses uploadFile for binary content with sqlite3 mimeType.
 */
export async function uploadFullBackup(
  filename: string,
  localDbPath: string,
): Promise<void> {
  await ensureDriveAccess();
  await CloudStorage.uploadFile(
    `${BACKUP_PREFIX}${filename}`,
    localDbPath,
    SCOPE,
    { mimeType: 'application/x-sqlite3' },
  );
}

// ---------------------------------------------------------------------------
// Download operations
// ---------------------------------------------------------------------------

/**
 * Download a backup file from Drive appdata to a local path.
 */
export async function downloadBackup(
  remoteFilename: string,
  localDestPath: string,
): Promise<string> {
  await ensureDriveAccess();
  return await CloudStorage.downloadFile(
    `${BACKUP_PREFIX}${remoteFilename}`,
    localDestPath,
    SCOPE,
  );
}

// ---------------------------------------------------------------------------
// Manifest operations
// ---------------------------------------------------------------------------

/**
 * List remote backups derived from the sync manifest.
 * Returns an array of RemoteBackupEntry (full + incremental).
 */
export async function listRemoteBackups(): Promise<RemoteBackupEntry[]> {
  await ensureDriveAccess();
  try {
    const json = await CloudStorage.readFile(MANIFEST_PATH, SCOPE);
    const manifest = JSON.parse(json) as SyncManifest;
    const entries: RemoteBackupEntry[] = [];
    if (manifest.lastFullBackupId) {
      entries.push({
        id: manifest.lastFullBackupId,
        type: 'full',
        filename: manifest.lastFullBackupId,
        uploadedAt: manifest.lastSyncedAt,
        sizeBytes: null,
      });
    }
    for (const incId of manifest.incrementalIds) {
      entries.push({
        id: incId,
        type: 'incremental',
        filename: incId,
        uploadedAt: manifest.lastSyncedAt,
        sizeBytes: null,
      });
    }
    return entries;
  } catch {
    return [];
  }
}

/**
 * Upload the sync manifest to Drive appdata.
 */
export async function uploadSyncManifest(
  manifest: SyncManifest,
): Promise<void> {
  await ensureDriveAccess();
  await CloudStorage.writeFile(
    MANIFEST_PATH,
    JSON.stringify(manifest, null, 2),
    SCOPE,
  );
}

/**
 * Download and parse the sync manifest from Drive appdata.
 */
export async function downloadSyncManifest(): Promise<SyncManifest | null> {
  await ensureDriveAccess();
  try {
    const json = await CloudStorage.readFile(MANIFEST_PATH, SCOPE);
    return JSON.parse(json) as SyncManifest;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Delete operations
// ---------------------------------------------------------------------------

/**
 * Delete a file from Drive appdata.
 */
export async function deleteRemoteFile(remotePath: string): Promise<void> {
  await ensureDriveAccess();
  await CloudStorage.writeFile(remotePath, '', SCOPE);
}
