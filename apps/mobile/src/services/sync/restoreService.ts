/**
 * Restore service -- discover and apply remote backups from Google Drive.
 *
 * discoverRemoteBackups() downloads the sync manifest from Drive.
 * restoreFromDrive() downloads full + incremental backups and replays them.
 */

import { downloadSyncManifest, downloadBackup } from './driveSync';
import { open } from '@op-engineering/op-sqlite';
import { Paths, File } from 'expo-file-system';
import type { SyncManifest } from './types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BACKUP_DIR = '/backups/';

function getBackupDirUri(): string {
  return `${Paths.document.uri}${BACKUP_DIR}`;
}

// ---------------------------------------------------------------------------
// Discover remote backups
// ---------------------------------------------------------------------------

/**
 * Download the sync manifest from Drive to discover available backups.
 * Returns the manifest or null if none exists.
 */
export async function discoverRemoteBackups(): Promise<SyncManifest | null> {
  return await downloadSyncManifest();
}

// ---------------------------------------------------------------------------
// Restore from Drive
// ---------------------------------------------------------------------------

/**
 * Download and apply the full backup + incrementals from Drive.
 *
 * Flow:
 * 1. Download sync manifest
 * 2. Download full backup .db file
 * 3. Download each incremental in order
 * 4. Open restored DB, replay incrementals
 * 5. Close restored DB (app restart required to use restored data)
 */
export async function restoreFromDrive(): Promise<void> {
  const manifest = await downloadSyncManifest();
  if (!manifest) {
    throw new Error('No remote backups found');
  }

  const backupDir = getBackupDirUri();

  // 1. Download full backup
  if (!manifest.lastFullBackupId) {
    throw new Error('No full backup in manifest');
  }

  const fullLocalPath = backupDir + manifest.lastFullBackupId;
  await downloadBackup(manifest.lastFullBackupId, fullLocalPath);

  // 2. Download incrementals in order
  for (const incId of manifest.incrementalIds) {
    const incLocalPath = backupDir + incId;
    await downloadBackup(incId, incLocalPath);
  }

  // 3. Open the downloaded full backup as a temporary DB connection
  const restoreDb = open({ name: 'restore-temp.db', location: backupDir });

  try {
    // 4. Replay incrementals into the restored DB
    for (const incId of manifest.incrementalIds) {
      const incLocalPath = backupDir + incId;
      try {
        const incFile = new File(incLocalPath);
        const json = await incFile.text();
        const changeset = JSON.parse(json) as {
          changes: Array<{
            table_name: string;
            row_id: number;
            operation: string;
            new_values?: Record<string, unknown>;
          }>;
        };

        for (const change of changeset.changes) {
          try {
            if (change.operation === 'DELETE') {
              restoreDb.executeSync(
                `DELETE FROM ${change.table_name} WHERE rowid = ?`,
                [change.row_id],
              );
            } else if (change.new_values) {
              const columns = Object.keys(change.new_values);
              const values = columns.map((col) => change.new_values![col]);
              const placeholders = columns.map(() => '?').join(', ');
              restoreDb.executeSync(
                `INSERT OR REPLACE INTO ${change.table_name} (${columns.join(', ')}) VALUES (${placeholders})`,
                values as Array<string | number | null>,
              );
            }
          } catch {
            // Skip individual row errors -- best effort replay
          }
        }
      } catch {
        // Skip unreadable incrementals
      }
    }
  } finally {
    restoreDb.close();
  }

  // 5. Copy restored DB over live DB
  // App restart required after this to pick up new data
  const restoredFile = new File(backupDir + 'restore-temp.db');
  const liveDbPath = `${Paths.document.uri}foodtracker.db`;
  const liveFile = new File(liveDbPath);

  try {
    if (liveFile.exists) {
      liveFile.delete();
    }
  } catch {
    // Best effort
  }

  restoredFile.copy(liveFile);
}
