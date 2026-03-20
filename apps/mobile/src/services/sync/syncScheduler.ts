/**
 * Sync scheduler -- extends background backup with Drive + FTP upload.
 *
 * triggerManualSync() performs an incremental backup, dispatches uploads
 * to enabled backends (Google Drive and/or FTP) via Promise.allSettled
 * so one backend failing does not block the other.
 * Respects WiFi-only preference.
 */

import NetInfo from '@react-native-community/netinfo';
import { isSignedIn } from './driveAuth';
import {
  uploadIncremental,
  downloadSyncManifest,
  uploadSyncManifest,
} from './driveSync';
import { syncToFtp } from './ftpSync';
import { loadFtpCredentials } from './ftpClient';
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
 * Perform incremental backup and upload to enabled backends.
 * Dispatches Drive and FTP uploads independently via Promise.allSettled.
 * Respects WiFi-only setting. At least one backend (or neither) may be enabled.
 */
export async function triggerManualSync(): Promise<void> {
  const store = useSyncStore.getState();

  // 1. Check if any backend is enabled
  const driveEnabled = isSignedIn();
  const ftpCreds = store.ftpEnabled ? await loadFtpCredentials() : null;
  const ftpEnabled = store.ftpEnabled && ftpCreds !== null;

  // 2. WiFi gate
  if (store.wifiOnly) {
    const netState = await NetInfo.fetch();
    if (netState.type !== 'wifi') return;
  }

  // 3. Set syncing status
  store.setSyncStatus('syncing');
  if (ftpEnabled) store.setFtpSyncStatus('syncing');

  try {
    // 4. Create local backup
    const result = await performIncrementalBackup();

    if (result) {
      const promises: Promise<void>[] = [];

      // 5a. Drive upload (if signed in)
      if (driveEnabled) {
        promises.push(
          (async () => {
            const localPath = getBackupDirUri() + result.filename;
            await uploadIncremental(result.filename, localPath);

            // Update sync manifest
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
          })(),
        );
      }

      // 5b. FTP upload (if enabled and credentials present)
      if (ftpEnabled) {
        promises.push(syncToFtp(result));
      }

      // 6. Dispatch independently -- one failing doesn't block the other
      const results = await Promise.allSettled(promises);

      // 7. Check results per backend
      let driveOk = true;
      let ftpOk = true;
      let idx = 0;
      if (driveEnabled) {
        if (results[idx]?.status === 'rejected') {
          driveOk = false;
          if (__DEV__) console.warn('Drive sync failed:', (results[idx] as PromiseRejectedResult).reason);
        }
        idx++;
      }
      if (ftpEnabled) {
        if (results[idx]?.status === 'rejected') {
          ftpOk = false;
          if (__DEV__) console.warn('FTP sync failed:', (results[idx] as PromiseRejectedResult).reason);
        }
      }

      // 8. Update store per backend
      if (driveEnabled) {
        store.setSyncStatus(driveOk ? 'idle' : 'error');
        if (driveOk) store.setLastSyncAt(new Date().toISOString());
      } else {
        store.setSyncStatus('idle');
      }

      if (ftpEnabled) {
        store.setFtpSyncStatus(ftpOk ? 'idle' : 'error');
        if (ftpOk) store.setLastFtpSyncAt(new Date().toISOString());
      }

      return;
    }

    // No backup produced -- still mark idle
    store.setSyncStatus('idle');
    if (ftpEnabled) store.setFtpSyncStatus('idle');
    store.setLastSyncAt(new Date().toISOString());
  } catch (err) {
    // Backup creation failed
    store.setSyncStatus('error');
    if (ftpEnabled) store.setFtpSyncStatus('error');
    if (__DEV__) console.warn('Sync failed:', err);
  }
}

