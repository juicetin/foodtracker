/**
 * Sync scheduler -- extends background backup with Drive upload.
 *
 * triggerManualSync() performs an incremental backup, uploads it to Drive,
 * and updates the sync manifest. Respects WiFi-only preference and
 * Google Sign-In status.
 */

import NetInfo from '@react-native-community/netinfo';
import { isSignedIn } from './driveAuth';
import {
  uploadIncremental,
  downloadSyncManifest,
  uploadSyncManifest,
} from './driveSync';
import { performIncrementalBackup } from '../backup/backupService';
import { useSyncStore } from '../../store/useSyncStore';
import { Paths } from 'expo-file-system';
import type { SyncManifest } from './types';
import Constants from 'expo-constants';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BACKUP_DIR = '/backups/';

function getBackupDirUri(): string {
  return `${Paths.document.uri}${BACKUP_DIR}`;
}

// ---------------------------------------------------------------------------
// Manual sync trigger
// ---------------------------------------------------------------------------

/**
 * Perform incremental backup and upload to Drive.
 * Respects WiFi-only setting and sign-in status.
 */
export async function triggerManualSync(): Promise<void> {
  // 1. Check sign-in
  if (!isSignedIn()) return;

  const store = useSyncStore.getState();

  // 2. WiFi gate
  if (store.wifiOnly) {
    const netState = await NetInfo.fetch();
    if (netState.type !== 'wifi') return;
  }

  // 3. Set syncing status
  store.setSyncStatus('syncing');

  try {
    // 4. Create local backup
    const result = await performIncrementalBackup();

    if (result) {
      // 5. Upload to Drive
      const localPath = getBackupDirUri() + result.filename;
      await uploadIncremental(result.filename, localPath);

      // 6. Update sync manifest
      const existingManifest = await downloadSyncManifest();
      const manifest: SyncManifest = existingManifest ?? {
        deviceId: Constants.installationId ?? 'unknown',
        lastSyncedAt: '',
        lastFullBackupId: null,
        incrementalIds: [],
        appVersion: Constants.expoConfig?.version ?? '1.0.0',
      };

      manifest.incrementalIds.push(result.filename);
      manifest.lastSyncedAt = new Date().toISOString();
      await uploadSyncManifest(manifest);
    }

    // 7. Update store
    store.setLastSyncAt(new Date().toISOString());
    store.setSyncStatus('idle');
  } catch (err) {
    // 8. Error handling
    store.setSyncStatus('error');
    if (__DEV__) console.warn('Sync failed:', err);
  }
}

