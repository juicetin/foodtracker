/**
 * Sync service type definitions.
 *
 * Defines manifest, conflict, resolution, preferences,
 * and remote backup entry types for Google Drive sync.
 */

export interface SyncManifest {
  deviceId: string;
  lastSyncedAt: string;
  lastFullBackupId: string | null;
  incrementalIds: string[];
  appVersion: string;
}

export interface RemoteBackupEntry {
  id: string;
  type: 'full' | 'incremental';
  filename: string;
  uploadedAt: string;
  sizeBytes: number | null;
}

export interface SyncConflict {
  table: string;
  rowId: number;
  field: string;
  localValue: unknown;
  localTimestamp: string;
  remoteValue: unknown;
  remoteTimestamp: string;
}

export interface SyncResolution {
  table: string;
  rowId: number;
  field: string;
  resolvedValue: unknown;
  source: 'local' | 'remote';
}

export type SyncStatus = 'idle' | 'syncing' | 'error' | 'conflict';
