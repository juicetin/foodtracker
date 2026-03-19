/**
 * Backup system type definitions.
 *
 * Defines tracked tables, journal entries, changeset format,
 * and backup metadata for incremental + full backup operations.
 */

export const TRACKED_TABLES = new Set([
  'food_entries', 'ingredients', 'photos', 'scanned_dishes',
  'favourite_meals', 'correction_history',
  'off_product_cache', 'off_search_cache', 'installed_packs',
  'custom_recipes', 'user_settings',
]);

export const BACKUP_DIR = 'backups/';

export interface ChangeJournalEntry {
  id: number;
  table_name: string;
  row_id: number;
  operation: 'INSERT' | 'UPDATE' | 'DELETE';
  timestamp: string;
}

export interface IncrementalChangeset {
  app: 'Tastimate';
  backupId: string;
  type: 'incremental';
  createdAt: string;
  sinceLastBackup: string | null;
  tableChangeCounts: Record<string, { insert: number; update: number; delete: number }>;
  changes: ChangeJournalEntry[];
}

export interface BackupMetadata {
  id: string;
  type: 'full' | 'incremental';
  filename: string;
  createdAt: string;
  journalFrom: string | null;
  journalTo: string | null;
  sizeBytes: number | null;
  parentFullId: string | null;
}
