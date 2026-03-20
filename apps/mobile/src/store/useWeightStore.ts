/**
 * Weight entries state management.
 *
 * Non-persisted Zustand store -- data lives in SQLite weight_entries table.
 * Supports manual entry, Health Connect sync with dedup, and EMA trend derivation.
 */

import { create } from 'zustand';
import { opsqlite } from '../../db/client';
import { readWeightRecords } from '../services/health/healthConnectService';
import {
  calculateWeightTrend,
  type WeightEntry,
  type WeightTrend,
} from '../services/health/weightTrendService';

interface WeightStoreState {
  entries: WeightEntry[];
  isLoading: boolean;
  lastSyncAt: string | null;

  /** Load all weight entries from SQLite, sorted by date ASC. */
  loadEntries: () => Promise<void>;

  /** Add a manual weight entry. Uses INSERT OR REPLACE on date for upsert. */
  addManualWeight: (date: string, weightKg: number) => Promise<void>;

  /**
   * Sync weight records from Health Connect.
   * @param days - Number of days to look back (default 90).
   * @returns Count of records processed.
   */
  syncFromHealthConnect: (days?: number) => Promise<number>;

  /** Delete a weight entry by id. */
  deleteWeightEntry: (id: number) => Promise<void>;

  /** Derive EMA-smoothed weight trend from current entries. */
  getWeightTrend: () => WeightTrend;
}

/** Ensure weight_entries table exists. Called once on first load. */
let tableEnsured = false;
function ensureTable(): void {
  if (tableEnsured) return;
  opsqlite.execute(`CREATE TABLE IF NOT EXISTS weight_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    weight_kg REAL NOT NULL,
    source TEXT NOT NULL,
    health_connect_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  )`);
  tableEnsured = true;
}

export const useWeightStore = create<WeightStoreState>((set, get) => ({
  entries: [],
  isLoading: false,
  lastSyncAt: null,

  loadEntries: async () => {
    ensureTable();
    set({ isLoading: true });
    try {
      const result = opsqlite.execute(
        'SELECT id, date, weight_kg, source, health_connect_id FROM weight_entries ORDER BY date ASC',
      );
      const entries: WeightEntry[] = (result.rows ?? []).map((row: any) => ({
        date: row.date,
        weightKg: row.weight_kg,
        source: row.source as 'manual' | 'health_connect',
      }));
      set({ entries });
    } finally {
      set({ isLoading: false });
    }
  },

  addManualWeight: async (date: string, weightKg: number) => {
    ensureTable();
    opsqlite.execute(
      'INSERT OR REPLACE INTO weight_entries (date, weight_kg, source) VALUES (?, ?, ?)',
      [date, weightKg, 'manual'],
    );
    await get().loadEntries();
  },

  syncFromHealthConnect: async (days: number = 90) => {
    ensureTable();
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const records = await readWeightRecords(startDate, endDate);

    for (const record of records) {
      opsqlite.execute(
        'INSERT OR REPLACE INTO weight_entries (date, weight_kg, source, health_connect_id) VALUES (?, ?, ?, ?)',
        [record.date, record.weightKg, 'health_connect', record.healthConnectId],
      );
    }

    const now = new Date().toISOString();
    set({ lastSyncAt: now });

    // Refresh entries
    await get().loadEntries();

    return records.length;
  },

  deleteWeightEntry: async (id: number) => {
    ensureTable();
    opsqlite.execute('DELETE FROM weight_entries WHERE id = ?', [id]);
    await get().loadEntries();
  },

  getWeightTrend: () => {
    return calculateWeightTrend(get().entries);
  },
}));
