/**
 * Backup service — incremental JSON changesets, full VACUUM INTO snapshots,
 * compaction via incremental replay, and retention management.
 *
 * Plain TS module with named exports following the exportService.ts pattern.
 * Uses expo-file-system v19 class-based API (Paths, File, Directory).
 */

import { opsqlite } from '../../../db/client';
import { open } from '@op-engineering/op-sqlite';
import { Paths, File, Directory } from 'expo-file-system';
import { getJournalEntries, clearJournal } from './changeJournal';
import {
  BACKUP_DIR,
  type IncrementalChangeset,
  type BackupMetadata,
} from './types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getBackupDirUri(): string {
  return `${Paths.document.uri}${BACKUP_DIR}`;
}

function ensureBackupDir(): string {
  const dirUri = getBackupDirUri();
  const dir = new Directory(dirUri);
  if (!dir.exists) {
    dir.create();
  }
  return dirUri;
}

function isoFileSafe(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function getFileSize(uri: string): number | null {
  try {
    const file = new File(uri);
    if (file.exists && 'size' in file) {
      return (file as unknown as { size: number }).size ?? null;
    }
  } catch {
    // Size unknown
  }
  return null;
}

function deleteFileIfExists(uri: string): void {
  try {
    const file = new File(uri);
    if (file.exists) {
      file.delete();
    }
  } catch {
    // Best effort
  }
}

// ---------------------------------------------------------------------------
// Incremental backup — JSON changeset from journal
// ---------------------------------------------------------------------------

export async function performIncrementalBackup(): Promise<{ filename: string; changeCount: number } | null> {
  const entries = getJournalEntries();
  if (entries.length === 0) return null;

  const backupDir = ensureBackupDir();
  const backupId = crypto.randomUUID();
  const createdAt = new Date().toISOString();

  // Find the timestamp of the most recent previous backup
  let sinceLastBackup: string | null = null;
  try {
    const rows = opsqlite.executeSync(
      'SELECT created_at FROM _backup_metadata ORDER BY created_at DESC LIMIT 1',
    ).rows as Array<Record<string, unknown>>;
    if (rows.length > 0) {
      sinceLastBackup = rows[0]!.created_at as string;
    }
  } catch {
    // No previous backups
  }

  // Build table change counts
  const tableChangeCounts: Record<string, { insert: number; update: number; delete: number }> = {};
  for (const entry of entries) {
    if (!tableChangeCounts[entry.table_name]) {
      tableChangeCounts[entry.table_name] = { insert: 0, update: 0, delete: 0 };
    }
    const counts = tableChangeCounts[entry.table_name]!;
    if (entry.operation === 'INSERT') counts.insert++;
    else if (entry.operation === 'UPDATE') counts.update++;
    else if (entry.operation === 'DELETE') counts.delete++;
  }

  const changeset: IncrementalChangeset = {
    app: 'Tastimate',
    backupId,
    type: 'incremental',
    createdAt,
    sinceLastBackup,
    tableChangeCounts,
    changes: entries,
  };

  const filename = `tastimate-backup-incremental-${isoFileSafe()}.json`;
  const filePath = backupDir + filename;

  const file = new File(filePath);
  file.write(JSON.stringify(changeset, null, 2));

  const sizeBytes = getFileSize(filePath);

  // Find latest full backup id for parent reference
  let parentFullId: string | null = null;
  try {
    const rows = opsqlite.executeSync(
      "SELECT id FROM _backup_metadata WHERE type = 'full' ORDER BY created_at DESC LIMIT 1",
    ).rows as Array<Record<string, unknown>>;
    if (rows.length > 0) {
      parentFullId = rows[0]!.id as string;
    }
  } catch {
    // No full backup yet
  }

  const journalFrom = entries[0]!.timestamp;
  const journalTo = entries[entries.length - 1]!.timestamp;

  // Record metadata
  opsqlite.executeSync(
    'INSERT INTO _backup_metadata (id, type, filename, created_at, journal_from, journal_to, size_bytes, parent_full_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
    [backupId, 'incremental', filename, createdAt, journalFrom, journalTo, sizeBytes, parentFullId],
  );

  // Clear processed journal entries
  clearJournal(journalTo);

  return { filename, changeCount: entries.length };
}

// ---------------------------------------------------------------------------
// Full backup — VACUUM INTO snapshot
// ---------------------------------------------------------------------------

export async function performFullBackup(): Promise<{ filename: string; sizeBytes: number }> {
  const backupDir = ensureBackupDir();
  const filename = `tastimate-backup-full-${isoFileSafe()}.db`;
  const fullPath = backupDir + filename;

  // VACUUM INTO requires the target file to not exist
  deleteFileIfExists(fullPath);

  await opsqlite.execute('VACUUM INTO ?', [fullPath]);

  const sizeBytes = getFileSize(fullPath) ?? 0;
  const backupId = crypto.randomUUID();

  // Record metadata
  opsqlite.executeSync(
    'INSERT INTO _backup_metadata (id, type, filename, created_at, size_bytes) VALUES (?, ?, ?, ?, ?)',
    [backupId, 'full', filename, new Date().toISOString(), sizeBytes],
  );

  // Clear entire journal — full backup captures everything
  clearJournal();

  return { filename, sizeBytes };
}

// ---------------------------------------------------------------------------
// Compaction — replay incremental JSON diffs onto last full backup copy
// ---------------------------------------------------------------------------

export async function compactBackups(): Promise<{ filename: string } | null> {
  // Find latest full backup
  let latestFull: { id: string; filename: string; created_at: string } | null = null;
  try {
    const rows = opsqlite.executeSync(
      "SELECT id, filename, created_at FROM _backup_metadata WHERE type = 'full' ORDER BY created_at DESC LIMIT 1",
    ).rows as Array<Record<string, unknown>>;
    if (rows.length > 0) {
      latestFull = {
        id: rows[0]!.id as string,
        filename: rows[0]!.filename as string,
        created_at: rows[0]!.created_at as string,
      };
    }
  } catch {
    return null;
  }

  if (!latestFull) return null;

  // Find incrementals after the latest full backup
  let incrementals: Array<{ id: string; filename: string; created_at: string }> = [];
  try {
    const rows = opsqlite.executeSync(
      "SELECT id, filename, created_at FROM _backup_metadata WHERE type = 'incremental' AND created_at > ? ORDER BY created_at ASC",
      [latestFull.created_at],
    ).rows as Array<Record<string, unknown>>;
    incrementals = rows.map((r) => ({
      id: r.id as string,
      filename: r.filename as string,
      created_at: r.created_at as string,
    }));
  } catch {
    return null;
  }

  // Need at least 7 incrementals to trigger compaction
  if (incrementals.length < 7) return null;

  const backupDir = ensureBackupDir();
  const newFilename = `tastimate-backup-full-${isoFileSafe()}.db`;

  // Copy the last full backup to a new file
  const sourceFile = new File(backupDir + latestFull.filename);
  sourceFile.copy(new File(backupDir + newFilename));

  // Open the copy with a separate op-sqlite connection
  const backupDb = open({ name: newFilename, location: backupDir });

  try {
    // Replay each incremental changeset
    for (const inc of incrementals) {
      const incFile = new File(backupDir + inc.filename);
      const json = await incFile.text();
      const changeset = JSON.parse(json) as IncrementalChangeset;

      for (const change of changeset.changes) {
        if (change.operation === 'DELETE') {
          backupDb.executeSync(
            `DELETE FROM ${change.table_name} WHERE rowid = ?`,
            [change.row_id],
          );
        } else {
          // INSERT or UPDATE — fetch current row state from live DB
          try {
            const liveRows = opsqlite.executeSync(
              `SELECT * FROM ${change.table_name} WHERE rowid = ?`,
              [change.row_id],
            ).rows as Array<Record<string, unknown>>;

            if (liveRows.length > 0) {
              const row = liveRows[0]!;
              const columns = Object.keys(row);
              const values = columns.map((col) => row[col]);
              const placeholders = columns.map(() => '?').join(', ');

              backupDb.executeSync(
                `INSERT OR REPLACE INTO ${change.table_name} (${columns.join(', ')}) VALUES (${placeholders})`,
                values as Array<string | number | null>,
              );
            }
            // If row no longer exists in live DB (deleted after this incremental), skip
          } catch {
            // Skip individual row errors — best effort replay
          }
        }
      }
    }

    // Clean up the compacted copy
    backupDb.executeSync('VACUUM');
  } finally {
    backupDb.close();
  }

  const sizeBytes = getFileSize(backupDir + newFilename);
  const newId = crypto.randomUUID();

  // Record the new full backup metadata
  opsqlite.executeSync(
    'INSERT INTO _backup_metadata (id, type, filename, created_at, size_bytes) VALUES (?, ?, ?, ?, ?)',
    [newId, 'full', newFilename, new Date().toISOString(), sizeBytes],
  );

  // Delete superseded incrementals from disk and metadata
  for (const inc of incrementals) {
    deleteFileIfExists(backupDir + inc.filename);
    opsqlite.executeSync('DELETE FROM _backup_metadata WHERE id = ?', [inc.id]);
  }

  // Run retention cleanup
  await cleanupOldBackups();

  return { filename: newFilename };
}

// ---------------------------------------------------------------------------
// Retention — keep last 3 full backups
// ---------------------------------------------------------------------------

export async function cleanupOldBackups(): Promise<void> {
  const backupDir = ensureBackupDir();

  try {
    const fullRows = opsqlite.executeSync(
      "SELECT id, filename, created_at FROM _backup_metadata WHERE type = 'full' ORDER BY created_at DESC",
    ).rows as Array<Record<string, unknown>>;

    // Keep first 3, delete the rest
    const toDelete = fullRows.slice(3);
    for (const row of toDelete) {
      const fullId = row.id as string;
      const filename = row.filename as string;
      const createdAt = row.created_at as string;

      // Delete the full backup file
      deleteFileIfExists(backupDir + filename);

      // Delete associated incrementals (by parent_full_id or created_at before next full)
      try {
        const incRows = opsqlite.executeSync(
          "SELECT id, filename FROM _backup_metadata WHERE type = 'incremental' AND (parent_full_id = ? OR created_at <= ?)",
          [fullId, createdAt],
        ).rows as Array<Record<string, unknown>>;

        for (const inc of incRows) {
          deleteFileIfExists(backupDir + (inc.filename as string));
          opsqlite.executeSync('DELETE FROM _backup_metadata WHERE id = ?', [inc.id as string]);
        }
      } catch {
        // Best effort
      }

      // Delete the full backup metadata
      opsqlite.executeSync('DELETE FROM _backup_metadata WHERE id = ?', [fullId]);
    }
  } catch {
    // Best effort cleanup
  }
}

// ---------------------------------------------------------------------------
// List backups
// ---------------------------------------------------------------------------

export function listBackups(): BackupMetadata[] {
  try {
    const rows = opsqlite.executeSync(
      'SELECT id, type, filename, created_at, journal_from, journal_to, size_bytes, parent_full_id FROM _backup_metadata ORDER BY created_at DESC',
    ).rows as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      id: r.id as string,
      type: r.type as BackupMetadata['type'],
      filename: r.filename as string,
      createdAt: r.created_at as string,
      journalFrom: (r.journal_from as string) ?? null,
      journalTo: (r.journal_to as string) ?? null,
      sizeBytes: (r.size_bytes as number) ?? null,
      parentFullId: (r.parent_full_id as string) ?? null,
    }));
  } catch {
    return [];
  }
}
