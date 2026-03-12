/**
 * CorrectionStore test suite -- validates SQLite correction history persistence
 * and suggestion engine.
 */

// Mock db/client with chainable drizzle API
jest.mock('../../../../db/client', () => {
  const mockInsert = jest.fn().mockReturnValue({
    values: jest.fn().mockResolvedValue(undefined),
  });
  const mockSelect = jest.fn().mockReturnValue({
    from: jest.fn().mockReturnValue({
      where: jest.fn().mockResolvedValue([]),
    }),
  });

  return {
    userDb: {
      insert: mockInsert,
      select: mockSelect,
    },
  };
});

jest.mock('../../../../db/schema', () => ({
  correctionHistory: {
    id: 'id',
    originalClassName: 'original_class_name',
    correctedClassName: 'corrected_class_name',
    confidence: 'confidence',
    correctedAt: 'corrected_at',
  },
}));

jest.mock('drizzle-orm', () => ({
  eq: jest.fn((col, val) => ({ col, val, _type: 'eq' })),
  sql: jest.fn(),
}));

import { CorrectionStore } from '../correctionStore';

function getUserDb() {
  const { userDb } = require('../../../../db/client');
  return userDb;
}

describe('CorrectionStore', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    const userDb = getUserDb();
    // Reset insert chain
    userDb.insert.mockReturnValue({
      values: jest.fn().mockResolvedValue(undefined),
    });
    // Reset select chain
    userDb.select.mockReturnValue({
      from: jest.fn().mockReturnValue({
        where: jest.fn().mockResolvedValue([]),
      }),
    });
  });

  describe('recordCorrection', () => {
    it('persists a correction to SQLite via userDb.insert', async () => {
      const userDb = getUserDb();

      await CorrectionStore.recordCorrection('rice', 'fried rice', 0.85);

      expect(userDb.insert).toHaveBeenCalled();
      const valuesCall = userDb.insert.mock.results[0].value.values;
      expect(valuesCall).toHaveBeenCalledWith(
        expect.objectContaining({
          originalClassName: 'rice',
          correctedClassName: 'fried rice',
          confidence: 0.85,
        }),
      );
    });

    it('generates a UUID for the id field', async () => {
      const userDb = getUserDb();

      await CorrectionStore.recordCorrection('rice', 'fried rice', 0.85);

      const valuesCall = userDb.insert.mock.results[0].value.values;
      const insertedValues = valuesCall.mock.calls[0][0];
      expect(insertedValues.id).toBeDefined();
      expect(typeof insertedValues.id).toBe('string');
      expect(insertedValues.id.length).toBeGreaterThan(0);
    });
  });

  describe('getCorrections', () => {
    it('returns all corrections for a given original class name', async () => {
      const userDb = getUserDb();
      const mockRows = [
        {
          id: '1',
          originalClassName: 'rice',
          correctedClassName: 'fried rice',
          confidence: 0.85,
          correctedAt: '2026-03-12 10:00:00',
        },
        {
          id: '2',
          originalClassName: 'rice',
          correctedClassName: 'brown rice',
          confidence: 0.75,
          correctedAt: '2026-03-12 11:00:00',
        },
      ];

      const mockWhere = jest.fn().mockResolvedValue(mockRows);
      const mockFrom = jest.fn().mockReturnValue({ where: mockWhere });
      userDb.select.mockReturnValue({ from: mockFrom });

      const results = await CorrectionStore.getCorrections('rice');

      expect(userDb.select).toHaveBeenCalled();
      expect(mockFrom).toHaveBeenCalled();
      expect(mockWhere).toHaveBeenCalled();
      expect(results).toEqual(mockRows);
    });

    it('returns empty array when no corrections exist', async () => {
      const results = await CorrectionStore.getCorrections('unknown');
      expect(results).toEqual([]);
    });
  });

  describe('getSuggestion', () => {
    function setupMockCorrections(rows: Array<Record<string, unknown>>) {
      const userDb = getUserDb();
      const mockWhere = jest.fn().mockResolvedValue(rows);
      const mockFrom = jest.fn().mockReturnValue({ where: mockWhere });
      userDb.select.mockReturnValue({ from: mockFrom });
    }

    it('returns the most frequently corrected-to class when 3+ corrections exist', async () => {
      setupMockCorrections([
        { id: '1', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.85, correctedAt: '2026-03-12 10:00:00' },
        { id: '2', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.80, correctedAt: '2026-03-12 11:00:00' },
        { id: '3', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.78, correctedAt: '2026-03-12 12:00:00' },
        { id: '4', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.70, correctedAt: '2026-03-12 13:00:00' },
      ]);

      const suggestion = await CorrectionStore.getSuggestion('rice');
      expect(suggestion).toBe('fried rice');
    });

    it('returns null when fewer than 3 corrections exist', async () => {
      setupMockCorrections([
        { id: '1', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.85, correctedAt: '2026-03-12 10:00:00' },
        { id: '2', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.80, correctedAt: '2026-03-12 11:00:00' },
      ]);

      const suggestion = await CorrectionStore.getSuggestion('rice');
      expect(suggestion).toBeNull();
    });

    it('returns null when no corrections exist', async () => {
      const suggestion = await CorrectionStore.getSuggestion('unknown');
      expect(suggestion).toBeNull();
    });

    it('picks the most frequent correction when multiple patterns exist', async () => {
      setupMockCorrections([
        // 3x fried rice
        { id: '1', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.85, correctedAt: '2026-03-12 10:00:00' },
        { id: '2', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.80, correctedAt: '2026-03-12 11:00:00' },
        { id: '3', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.78, correctedAt: '2026-03-12 12:00:00' },
        // 4x brown rice (should win)
        { id: '4', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.70, correctedAt: '2026-03-12 13:00:00' },
        { id: '5', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.72, correctedAt: '2026-03-12 14:00:00' },
        { id: '6', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.74, correctedAt: '2026-03-12 15:00:00' },
        { id: '7', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.76, correctedAt: '2026-03-12 16:00:00' },
      ]);

      const suggestion = await CorrectionStore.getSuggestion('rice');
      expect(suggestion).toBe('brown rice');
    });

    it('returns null when the most frequent correction has fewer than 3 occurrences', async () => {
      // 2x fried rice, 2x brown rice -- neither reaches the 3-correction threshold
      setupMockCorrections([
        { id: '1', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.85, correctedAt: '2026-03-12 10:00:00' },
        { id: '2', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.80, correctedAt: '2026-03-12 11:00:00' },
        { id: '3', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.70, correctedAt: '2026-03-12 13:00:00' },
        { id: '4', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.72, correctedAt: '2026-03-12 14:00:00' },
      ]);

      const suggestion = await CorrectionStore.getSuggestion('rice');
      expect(suggestion).toBeNull();
    });
  });

  describe('multiple corrections tracked independently', () => {
    it('records multiple corrections for the same original class', async () => {
      const userDb = getUserDb();
      // Track all values calls
      const capturedValues: Array<Record<string, unknown>> = [];
      userDb.insert.mockImplementation(() => ({
        values: jest.fn((vals: Record<string, unknown>) => {
          capturedValues.push(vals);
          return Promise.resolve(undefined);
        }),
      }));

      await CorrectionStore.recordCorrection('rice', 'fried rice', 0.85);
      await CorrectionStore.recordCorrection('rice', 'brown rice', 0.75);

      expect(userDb.insert).toHaveBeenCalledTimes(2);
      expect(capturedValues).toHaveLength(2);

      // Check first call
      expect(capturedValues[0]).toEqual(
        expect.objectContaining({
          originalClassName: 'rice',
          correctedClassName: 'fried rice',
        }),
      );
      // Check second call
      expect(capturedValues[1]).toEqual(
        expect.objectContaining({
          originalClassName: 'rice',
          correctedClassName: 'brown rice',
        }),
      );
    });
  });
});
