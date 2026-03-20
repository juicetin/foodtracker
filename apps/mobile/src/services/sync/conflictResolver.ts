/**
 * Per-field LWW (Last Write Wins) conflict resolver.
 *
 * Compares field-level timestamps between local and remote change
 * journal entries to detect divergent fields, then auto-resolves
 * by picking the value with the latest timestamp.
 */

import { opsqlite } from '../../../db/client';
import type { SyncConflict, SyncResolution } from './types';

// ---------------------------------------------------------------------------
// Types for field-level change records
// ---------------------------------------------------------------------------

export interface FieldRecord {
  table: string;
  rowId: number;
  field: string;
  value: unknown;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Conflict detection
// ---------------------------------------------------------------------------

/**
 * Compare local and remote field-level changes.
 * Returns conflicts for fields where values differ between local and remote.
 * Identical values at the same timestamp are not conflicts.
 */
export function detectConflicts(
  localChanges: FieldRecord[],
  remoteChanges: FieldRecord[],
): SyncConflict[] {
  const conflicts: SyncConflict[] = [];

  // Build a map of remote changes for quick lookup
  const remoteMap = new Map<string, FieldRecord>();
  for (const remote of remoteChanges) {
    const key = `${remote.table}:${remote.rowId}:${remote.field}`;
    remoteMap.set(key, remote);
  }

  // Compare each local change against its remote counterpart
  for (const local of localChanges) {
    const key = `${local.table}:${local.rowId}:${local.field}`;
    const remote = remoteMap.get(key);

    if (!remote) continue; // No remote counterpart, not a conflict

    // Skip if values and timestamps are identical
    if (local.value === remote.value && local.timestamp === remote.timestamp) {
      continue;
    }

    // Skip if values are identical (regardless of timestamp)
    if (local.value === remote.value) {
      continue;
    }

    // Values differ -> conflict
    conflicts.push({
      table: local.table,
      rowId: local.rowId,
      field: local.field,
      localValue: local.value,
      localTimestamp: local.timestamp,
      remoteValue: remote.value,
      remoteTimestamp: remote.timestamp,
    });
  }

  return conflicts;
}

// ---------------------------------------------------------------------------
// Auto-resolution (LWW per field)
// ---------------------------------------------------------------------------

/**
 * For each conflict, pick the value with the latest timestamp.
 * Ties go to local (optimistic local-first).
 */
export function autoResolveConflicts(
  conflicts: SyncConflict[],
): SyncResolution[] {
  return conflicts.map((c) => {
    const remoteNewer =
      new Date(c.remoteTimestamp).getTime() > new Date(c.localTimestamp).getTime();

    return {
      table: c.table,
      rowId: c.rowId,
      field: c.field,
      resolvedValue: remoteNewer ? c.remoteValue : c.localValue,
      source: remoteNewer ? 'remote' as const : 'local' as const,
    };
  });
}

// ---------------------------------------------------------------------------
// Apply resolutions to local DB
// ---------------------------------------------------------------------------

/**
 * Execute SQL updates for each resolved field.
 */
export function applyResolution(resolutions: SyncResolution[]): void {
  for (const r of resolutions) {
    opsqlite.executeSync(
      `UPDATE ${r.table} SET ${r.field} = ? WHERE rowid = ?`,
      [r.resolvedValue as string | number | null, r.rowId],
    );
  }
}
