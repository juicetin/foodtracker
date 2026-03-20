/**
 * FTP sync operations -- mirrors driveSync pattern.
 *
 * Reads local backup file and uploads it to FTP server via ftpClient.
 */

import { Paths } from 'expo-file-system';
import { uploadToFtp } from './ftpClient';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BACKUP_DIR = '/backups/';

function getBackupDirUri(): string {
  return `${Paths.document.uri}${BACKUP_DIR}`;
}

// ---------------------------------------------------------------------------
// Sync operations
// ---------------------------------------------------------------------------

/**
 * Upload a backup result to FTP.
 * Mirrors the driveSync uploadIncremental pattern.
 */
export async function syncToFtp(
  backupResult: { filename: string },
): Promise<void> {
  const localPath = getBackupDirUri() + backupResult.filename;
  await uploadToFtp(localPath, backupResult.filename);
}
