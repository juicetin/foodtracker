/**
 * Unit tests for containerService -- container tare weight CRUD + usage tracking.
 *
 * Mocks opsqlite.execute to avoid real DB access (same pattern as offCacheService).
 */

const mockExecute = jest.fn();
jest.mock('../../../../db/client', () => ({
  opsqlite: { execute: (...args: unknown[]) => mockExecute(...args) },
}));

import {
  addContainer,
  getContainers,
  updateContainer,
  deleteContainer,
  recordContainerUsage,
  applyTare,
} from '../containerService';
import type { Container } from '../containerService';

beforeEach(() => {
  mockExecute.mockReset();
});

describe('containerService', () => {
  describe('addContainer()', () => {
    it('inserts a new row into container_weights with name and weightGrams', async () => {
      const insertedRow = {
        id: 1,
        name: 'Blue Bowl',
        weight_grams: 350,
        times_used: 0,
        last_used_at: null,
        created_at: '2026-03-20 12:00:00',
      };

      // First call: INSERT, second call: SELECT to return the inserted row
      mockExecute
        .mockResolvedValueOnce({ insertId: 1, rows: { _array: [] } })
        .mockResolvedValueOnce({ rows: { _array: [insertedRow] } });

      const result = await addContainer('Blue Bowl', 350);

      expect(mockExecute).toHaveBeenCalledTimes(2);
      const insertCall = mockExecute.mock.calls[0];
      expect(insertCall[0]).toContain('INSERT INTO container_weights');
      expect(insertCall[1]).toContain('Blue Bowl');
      expect(insertCall[1]).toContain(350);

      expect(result).toEqual<Container>({
        id: 1,
        name: 'Blue Bowl',
        weightGrams: 350,
        timesUsed: 0,
        lastUsedAt: null,
        createdAt: '2026-03-20 12:00:00',
      });
    });
  });

  describe('getContainers()', () => {
    it('returns all containers sorted by timesUsed DESC', async () => {
      mockExecute.mockResolvedValueOnce({
        rows: {
          _array: [
            {
              id: 2,
              name: 'Glass Jar',
              weight_grams: 500,
              times_used: 10,
              last_used_at: '2026-03-20',
              created_at: '2026-03-01',
            },
            {
              id: 1,
              name: 'Blue Bowl',
              weight_grams: 350,
              times_used: 3,
              last_used_at: '2026-03-19',
              created_at: '2026-03-01',
            },
          ],
        },
      });

      const result = await getContainers();

      expect(mockExecute).toHaveBeenCalledTimes(1);
      const sql = mockExecute.mock.calls[0][0] as string;
      expect(sql).toContain('container_weights');
      expect(sql).toContain('ORDER BY');
      expect(sql).toContain('times_used');

      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('Glass Jar');
      expect(result[0].timesUsed).toBe(10);
      expect(result[1].name).toBe('Blue Bowl');
    });
  });

  describe('updateContainer()', () => {
    it('updates name and/or weightGrams for existing container', async () => {
      mockExecute.mockResolvedValueOnce({ rows: { _array: [] } });

      await updateContainer(1, { name: 'Red Bowl', weightGrams: 400 });

      expect(mockExecute).toHaveBeenCalledTimes(1);
      const sql = mockExecute.mock.calls[0][0] as string;
      expect(sql).toContain('UPDATE container_weights');
      expect(sql).toContain('name');
      expect(sql).toContain('weight_grams');
    });
  });

  describe('deleteContainer()', () => {
    it('removes a container by ID', async () => {
      mockExecute.mockResolvedValueOnce({ rows: { _array: [] } });

      await deleteContainer(5);

      expect(mockExecute).toHaveBeenCalledTimes(1);
      const sql = mockExecute.mock.calls[0][0] as string;
      expect(sql).toContain('DELETE FROM container_weights');
      expect(mockExecute.mock.calls[0][1]).toContain(5);
    });
  });

  describe('recordContainerUsage()', () => {
    it('increments timesUsed and sets lastUsedAt', async () => {
      mockExecute.mockResolvedValueOnce({ rows: { _array: [] } });

      await recordContainerUsage(3);

      expect(mockExecute).toHaveBeenCalledTimes(1);
      const sql = mockExecute.mock.calls[0][0] as string;
      expect(sql).toContain('UPDATE container_weights');
      expect(sql).toContain('times_used = times_used + 1');
      expect(sql).toContain("last_used_at = datetime('now')");
      expect(mockExecute.mock.calls[0][1]).toContain(3);
    });
  });

  describe('applyTare()', () => {
    it('subtracts container weight from gross weight', () => {
      const container: Container = {
        id: 1,
        name: 'Bowl',
        weightGrams: 200,
        timesUsed: 5,
        lastUsedAt: null,
        createdAt: '2026-01-01',
      };

      expect(applyTare(500, container)).toBe(300);
    });

    it('returns 0 when tare exceeds gross weight (minimum 0)', () => {
      const container: Container = {
        id: 1,
        name: 'Heavy Bowl',
        weightGrams: 600,
        timesUsed: 0,
        lastUsedAt: null,
        createdAt: '2026-01-01',
      };

      expect(applyTare(400, container)).toBe(0);
    });
  });
});
