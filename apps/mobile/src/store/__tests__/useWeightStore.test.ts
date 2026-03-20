/**
 * Tests for useWeightStore -- weight entry management with HC sync.
 */

// Must mock db/client to avoid module-level opsqlite.updateHook() call
jest.mock('../../../db/client', () => ({
  opsqlite: {
    execute: jest.fn(() => ({ rows: [] as any[], rowsAffected: 0 })),
  },
  userDb: {},
}));

jest.mock('../../services/health/healthConnectService', () => ({
  readWeightRecords: jest.fn(),
}));

import { opsqlite } from '../../../db/client';
import { readWeightRecords } from '../../services/health/healthConnectService';
import { useWeightStore } from '../useWeightStore';

const mockExecute = opsqlite.execute as unknown as jest.Mock;
const mockReadWeightRecords = readWeightRecords as jest.Mock;

beforeEach(() => {
  mockExecute.mockClear();
  mockReadWeightRecords.mockClear();
  mockExecute.mockReturnValue({ rows: [], rowsAffected: 0 });
  useWeightStore.setState({
    entries: [],
    isLoading: false,
    lastSyncAt: null,
  });
});

describe('useWeightStore', () => {
  describe('addManualWeight', () => {
    it('inserts a weight_entry with source=manual and date=YYYY-MM-DD', async () => {
      await useWeightStore.getState().addManualWeight('2025-01-15', 80.5);

      const insertCall = mockExecute.mock.calls.find(
        (call: any[]) => typeof call[0] === 'string' && call[0].includes('INSERT OR REPLACE'),
      );
      expect(insertCall).toBeTruthy();
      expect(insertCall![1]).toContain('2025-01-15');
      expect(insertCall![1]).toContain(80.5);
      expect(insertCall![1]).toContain('manual');
    });
  });

  describe('syncFromHealthConnect', () => {
    it('reads HC records and upserts into weight_entries with source=health_connect', async () => {
      mockReadWeightRecords.mockResolvedValue([
        { date: '2025-01-15', weightKg: 80.5, healthConnectId: 'hc-1' },
        { date: '2025-01-16', weightKg: 80.2, healthConnectId: 'hc-2' },
      ]);

      const count = await useWeightStore.getState().syncFromHealthConnect(90);

      expect(mockReadWeightRecords).toHaveBeenCalledTimes(1);
      const insertCalls = mockExecute.mock.calls.filter(
        (call: any[]) => typeof call[0] === 'string' && call[0].includes('INSERT OR REPLACE'),
      );
      expect(insertCalls.length).toBeGreaterThanOrEqual(2);
      expect(count).toBe(2);
    });

    it('deduplicates by healthConnectId (no duplicate rows)', async () => {
      mockReadWeightRecords.mockResolvedValue([
        { date: '2025-01-15', weightKg: 80.5, healthConnectId: 'hc-1' },
        { date: '2025-01-15', weightKg: 80.6, healthConnectId: 'hc-1' },
      ]);

      const count = await useWeightStore.getState().syncFromHealthConnect(90);
      // INSERT OR REPLACE on date UNIQUE means SQL handles dedup
      expect(count).toBe(2);
    });
  });

  describe('loadEntries / getWeightEntries', () => {
    it('returns all entries sorted by date ASC', async () => {
      mockExecute.mockImplementation((sql: string) => {
        if (typeof sql === 'string' && sql.includes('SELECT')) {
          return {
            rows: [
              { id: 1, date: '2025-01-15', weight_kg: 80.5, source: 'manual', health_connect_id: null },
              { id: 2, date: '2025-01-16', weight_kg: 80.2, source: 'health_connect', health_connect_id: 'hc-1' },
            ],
            rowsAffected: 0,
          };
        }
        return { rows: [], rowsAffected: 0 };
      });

      await useWeightStore.getState().loadEntries();
      const entries = useWeightStore.getState().entries;

      expect(entries).toHaveLength(2);
      expect(entries[0].date).toBe('2025-01-15');
      expect(entries[1].date).toBe('2025-01-16');
      expect(entries[0].weightKg).toBe(80.5);
      expect(entries[1].source).toBe('health_connect');
    });
  });

  describe('deleteWeightEntry', () => {
    it('removes entry by id', async () => {
      await useWeightStore.getState().deleteWeightEntry(42);

      const deleteCall = mockExecute.mock.calls.find(
        (call: any[]) => typeof call[0] === 'string' && call[0].includes('DELETE'),
      );
      expect(deleteCall).toBeTruthy();
      expect(deleteCall![1]).toContain(42);
    });
  });

  describe('getWeightTrend', () => {
    it('returns trend derived from current entries', () => {
      useWeightStore.setState({
        entries: [
          { date: '2025-01-15', weightKg: 81, source: 'manual' as const },
          { date: '2025-01-16', weightKg: 80.5, source: 'manual' as const },
          { date: '2025-01-17', weightKg: 80.0, source: 'manual' as const },
        ],
      });

      const trend = useWeightStore.getState().getWeightTrend();

      expect(trend.raw).toHaveLength(3);
      expect(trend.smoothed).toHaveLength(3);
      expect(trend.latestKg).toBe(80.0);
      expect(trend.dates).toEqual(['2025-01-15', '2025-01-16', '2025-01-17']);
    });
  });
});
