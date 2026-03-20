/**
 * Background auto-backup scheduler using expo-task-manager + expo-background-task.
 *
 * IMPORTANT: This module is imported as a side-effect in App.tsx so that
 * TaskManager.defineTask() runs at module load time — before React renders.
 */

import * as TaskManager from 'expo-task-manager';
import * as BackgroundTask from 'expo-background-task';

export const BACKUP_TASK_NAME = 'TASTIMATE_AUTO_BACKUP';

// Define task at module scope (must be called before React renders)
TaskManager.defineTask(BACKUP_TASK_NAME, async () => {
  try {
    // Dynamic import to avoid circular deps at module load
    const { performIncrementalBackup, compactBackups } = await import('./backupService');
    const result = await performIncrementalBackup();
    if (result) {
      // Check if compaction is needed (every 7 incrementals)
      await compactBackups();
    }

    // Attempt Drive sync after local backup (failure does not affect local backup)
    try {
      const { triggerManualSync } = await import('../sync/syncScheduler');
      await triggerManualSync();
    } catch {
      // Drive sync failure is non-fatal -- local backup succeeded
    }

    return BackgroundTask.BackgroundTaskResult.Success;
  } catch {
    return BackgroundTask.BackgroundTaskResult.Failed;
  }
});

export async function registerAutoBackup(): Promise<void> {
  try {
    await BackgroundTask.registerTaskAsync(BACKUP_TASK_NAME, {
      minimumInterval: 24 * 60 * 60, // 24 hours (daily)
    });
  } catch (err) {
    // Silently fail -- auto-backup is nice-to-have, not critical
    if (__DEV__) console.warn('Failed to register auto-backup task:', err);
  }
}

export async function unregisterAutoBackup(): Promise<void> {
  try {
    await BackgroundTask.unregisterTaskAsync(BACKUP_TASK_NAME);
  } catch {
    // Task may not be registered -- ignore
  }
}
