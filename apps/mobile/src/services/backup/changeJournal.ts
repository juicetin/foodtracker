/**
 * Change journal helpers — read, count, and clear journal entries
 * written by the updateHook in db/client.ts.
 */

import { opsqlite } from '../../../db/client';
import type { ChangeJournalEntry } from './types';

/**
 * Read all journal entries ordered by timestamp, then id.
 */
export function getJournalEntries(): ChangeJournalEntry[] {
  try {
    const rows = opsqlite.executeSync(
      'SELECT id, table_name, row_id, operation, timestamp FROM _change_journal ORDER BY timestamp, id',
    ).rows as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      id: r.id as number,
      table_name: r.table_name as string,
      row_id: r.row_id as number,
      operation: r.operation as ChangeJournalEntry['operation'],
      timestamp: r.timestamp as string,
    }));
  } catch {
    return [];
  }
}

/**
 * Count pending journal entries.
 */
export function getJournalCount(): number {
  try {
    const rows = opsqlite.executeSync(
      'SELECT COUNT(*) AS cnt FROM _change_journal',
    ).rows as Array<Record<string, unknown>>;
    return (rows[0]?.cnt as number) ?? 0;
  } catch {
    return 0;
  }
}

/**
 * Clear journal entries. If beforeTimestamp is provided, only entries
 * with timestamp <= that value are deleted; otherwise all entries are removed.
 */
export function clearJournal(beforeTimestamp?: string): void {
  try {
    if (beforeTimestamp) {
      opsqlite.executeSync(
        'DELETE FROM _change_journal WHERE timestamp <= ?',
        [beforeTimestamp],
      );
    } else {
      opsqlite.executeSync('DELETE FROM _change_journal');
    }
  } catch {
    // Ignore errors — journal clear is best-effort
  }
}
