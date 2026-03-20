/**
 * conflictResolver tests — per-field LWW merge logic and conflict detection.
 */

import {
  detectConflicts,
  autoResolveConflicts,
  applyResolution,
} from '../conflictResolver';
import type { SyncConflict, SyncResolution } from '../types';

// ---------------------------------------------------------------------------
// Mock op-sqlite for applyResolution
// ---------------------------------------------------------------------------

const mockExecuteSync = jest.fn();
jest.mock('../../../../db/client', () => ({
  opsqlite: {
    executeSync: (...a: unknown[]) => mockExecuteSync(...a),
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

interface FieldRecord {
  table: string;
  rowId: number;
  field: string;
  value: unknown;
  timestamp: string;
}

const LOCAL_CHANGES: FieldRecord[] = [
  { table: 'food_entries', rowId: 1, field: 'calories', value: 500, timestamp: '2026-03-20T10:00:00Z' },
  { table: 'food_entries', rowId: 1, field: 'protein', value: 30, timestamp: '2026-03-20T10:00:00Z' },
  { table: 'food_entries', rowId: 2, field: 'food_name', value: 'Salad', timestamp: '2026-03-20T09:00:00Z' },
];

const REMOTE_CHANGES_IDENTICAL: FieldRecord[] = [
  { table: 'food_entries', rowId: 1, field: 'calories', value: 500, timestamp: '2026-03-20T10:00:00Z' },
  { table: 'food_entries', rowId: 1, field: 'protein', value: 30, timestamp: '2026-03-20T10:00:00Z' },
  { table: 'food_entries', rowId: 2, field: 'food_name', value: 'Salad', timestamp: '2026-03-20T09:00:00Z' },
];

const REMOTE_CHANGES_DIVERGENT: FieldRecord[] = [
  { table: 'food_entries', rowId: 1, field: 'calories', value: 600, timestamp: '2026-03-20T11:00:00Z' },
  { table: 'food_entries', rowId: 1, field: 'protein', value: 30, timestamp: '2026-03-20T10:00:00Z' },
  { table: 'food_entries', rowId: 2, field: 'food_name', value: 'Caesar Salad', timestamp: '2026-03-20T08:00:00Z' },
];

// ---------------------------------------------------------------------------
// detectConflicts
// ---------------------------------------------------------------------------

describe('detectConflicts', () => {
  it('returns empty array when local and remote are identical', () => {
    const conflicts = detectConflicts(LOCAL_CHANGES, REMOTE_CHANGES_IDENTICAL);
    expect(conflicts).toEqual([]);
  });

  it('returns conflict for field where remote is newer than local', () => {
    const conflicts = detectConflicts(LOCAL_CHANGES, REMOTE_CHANGES_DIVERGENT);

    // calories: remote 600 at 11:00 vs local 500 at 10:00 -> conflict
    // protein: identical -> no conflict
    // food_name: local 'Salad' at 09:00, remote 'Caesar Salad' at 08:00 -> conflict (divergent values)
    expect(conflicts.length).toBeGreaterThanOrEqual(1);

    const calorieConflict = conflicts.find(
      (c) => c.field === 'calories' && c.rowId === 1,
    );
    expect(calorieConflict).toBeDefined();
    expect(calorieConflict!.localValue).toBe(500);
    expect(calorieConflict!.remoteValue).toBe(600);
  });
});

// ---------------------------------------------------------------------------
// autoResolveConflicts
// ---------------------------------------------------------------------------

describe('autoResolveConflicts', () => {
  it('picks field with latest timestamp for each conflict', () => {
    const conflicts: SyncConflict[] = [
      {
        table: 'food_entries',
        rowId: 1,
        field: 'calories',
        localValue: 500,
        localTimestamp: '2026-03-20T10:00:00Z',
        remoteValue: 600,
        remoteTimestamp: '2026-03-20T11:00:00Z',
      },
      {
        table: 'food_entries',
        rowId: 2,
        field: 'food_name',
        localValue: 'Salad',
        localTimestamp: '2026-03-20T09:00:00Z',
        remoteValue: 'Caesar Salad',
        remoteTimestamp: '2026-03-20T08:00:00Z',
      },
    ];

    const resolutions = autoResolveConflicts(conflicts);

    expect(resolutions).toHaveLength(2);

    // calories: remote is newer (11:00 > 10:00) -> pick remote 600
    const calRes = resolutions.find((r) => r.field === 'calories');
    expect(calRes!.resolvedValue).toBe(600);
    expect(calRes!.source).toBe('remote');

    // food_name: local is newer (09:00 > 08:00) -> pick local 'Salad'
    const nameRes = resolutions.find((r) => r.field === 'food_name');
    expect(nameRes!.resolvedValue).toBe('Salad');
    expect(nameRes!.source).toBe('local');
  });
});

// ---------------------------------------------------------------------------
// applyResolution
// ---------------------------------------------------------------------------

describe('applyResolution', () => {
  it('executes SQL updates for each resolved field', () => {
    const resolutions: SyncResolution[] = [
      { table: 'food_entries', rowId: 1, field: 'calories', resolvedValue: 600, source: 'remote' },
      { table: 'food_entries', rowId: 2, field: 'food_name', resolvedValue: 'Salad', source: 'local' },
    ];

    applyResolution(resolutions);

    expect(mockExecuteSync).toHaveBeenCalledTimes(2);
    expect(mockExecuteSync).toHaveBeenCalledWith(
      'UPDATE food_entries SET calories = ? WHERE rowid = ?',
      [600, 1],
    );
    expect(mockExecuteSync).toHaveBeenCalledWith(
      'UPDATE food_entries SET food_name = ? WHERE rowid = ?',
      ['Salad', 2],
    );
  });
});
