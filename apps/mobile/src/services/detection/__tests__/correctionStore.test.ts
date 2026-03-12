/**
 * CorrectionStore test suite -- validates SQLite correction history persistence
 * and suggestion engine.
 */

// Mock db/client before imports
const mockInsert = jest.fn();
const mockSelect = jest.fn();
const mockValues = jest.fn();
const mockFrom = jest.fn();
const mockWhere = jest.fn();

jest.mock('../../../../db/client', () => ({
  userDb: {
    insert: mockInsert,
    select: mockSelect,
  },
}));

jest.mock('../../../../db/schema', () => ({
  correctionHistory: {
    id: 'id',
    originalClassName: 'original_class_name',
    correctedClassName: 'corrected_class_name',
    confidence: 'confidence',
    correctedAt: 'corrected_at',
  },
}));

// Provide a mock for drizzle-orm eq function
jest.mock('drizzle-orm', () => ({
  eq: jest.fn((col, val) => ({ col, val, _type: 'eq' })),
  sql: jest.fn(),
}));

import { CorrectionStore } from '../correctionStore';

describe('CorrectionStore', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Default mock chain: insert().values() resolves
    mockValues.mockResolvedValue(undefined);
    mockInsert.mockReturnValue({ values: mockValues });

    // Default mock chain: select().from().where() resolves to empty
    mockWhere.mockResolvedValue([]);
    mockFrom.mockReturnValue({ where: mockWhere });
    mockSelect.mockReturnValue({ from: mockFrom });
  });

  describe('recordCorrection', () => {
    it('persists a correction to SQLite via userDb.insert', async () => {
      await CorrectionStore.recordCorrection('rice', 'fried rice', 0.85);

      expect(mockInsert).toHaveBeenCalled();
      expect(mockValues).toHaveBeenCalledWith(
        expect.objectContaining({
          originalClassName: 'rice',
          correctedClassName: 'fried rice',
          confidence: 0.85,
        }),
      );
    });

    it('generates a UUID for the id field', async () => {
      await CorrectionStore.recordCorrection('rice', 'fried rice', 0.85);

      const insertedValues = mockValues.mock.calls[0][0];
      expect(insertedValues.id).toBeDefined();
      expect(typeof insertedValues.id).toBe('string');
      expect(insertedValues.id.length).toBeGreaterThan(0);
    });
  });

  describe('getCorrections', () => {
    it('returns all corrections for a given original class name', async () => {
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
      mockWhere.mockResolvedValue(mockRows);

      const results = await CorrectionStore.getCorrections('rice');

      expect(mockSelect).toHaveBeenCalled();
      expect(mockFrom).toHaveBeenCalled();
      expect(mockWhere).toHaveBeenCalled();
      expect(results).toEqual(mockRows);
    });

    it('returns empty array when no corrections exist', async () => {
      mockWhere.mockResolvedValue([]);

      const results = await CorrectionStore.getCorrections('unknown');

      expect(results).toEqual([]);
    });
  });

  describe('getSuggestion', () => {
    it('returns the most frequently corrected-to class when 3+ corrections exist', async () => {
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
          correctedClassName: 'fried rice',
          confidence: 0.80,
          correctedAt: '2026-03-12 11:00:00',
        },
        {
          id: '3',
          originalClassName: 'rice',
          correctedClassName: 'fried rice',
          confidence: 0.78,
          correctedAt: '2026-03-12 12:00:00',
        },
        {
          id: '4',
          originalClassName: 'rice',
          correctedClassName: 'brown rice',
          confidence: 0.70,
          correctedAt: '2026-03-12 13:00:00',
        },
      ];
      mockWhere.mockResolvedValue(mockRows);

      const suggestion = await CorrectionStore.getSuggestion('rice');

      expect(suggestion).toBe('fried rice');
    });

    it('returns null when fewer than 3 corrections exist', async () => {
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
          correctedClassName: 'fried rice',
          confidence: 0.80,
          correctedAt: '2026-03-12 11:00:00',
        },
      ];
      mockWhere.mockResolvedValue(mockRows);

      const suggestion = await CorrectionStore.getSuggestion('rice');

      expect(suggestion).toBeNull();
    });

    it('returns null when no corrections exist', async () => {
      mockWhere.mockResolvedValue([]);

      const suggestion = await CorrectionStore.getSuggestion('unknown');

      expect(suggestion).toBeNull();
    });

    it('picks the most frequent correction when multiple patterns exist', async () => {
      const mockRows = [
        // 3x fried rice
        { id: '1', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.85, correctedAt: '2026-03-12 10:00:00' },
        { id: '2', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.80, correctedAt: '2026-03-12 11:00:00' },
        { id: '3', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.78, correctedAt: '2026-03-12 12:00:00' },
        // 4x brown rice (should win)
        { id: '4', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.70, correctedAt: '2026-03-12 13:00:00' },
        { id: '5', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.72, correctedAt: '2026-03-12 14:00:00' },
        { id: '6', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.74, correctedAt: '2026-03-12 15:00:00' },
        { id: '7', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.76, correctedAt: '2026-03-12 16:00:00' },
      ];
      mockWhere.mockResolvedValue(mockRows);

      const suggestion = await CorrectionStore.getSuggestion('rice');

      expect(suggestion).toBe('brown rice');
    });

    it('returns null when the most frequent correction has fewer than 3 occurrences', async () => {
      // 2x fried rice, 2x brown rice -- neither reaches the 3-correction threshold
      const mockRows = [
        { id: '1', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.85, correctedAt: '2026-03-12 10:00:00' },
        { id: '2', originalClassName: 'rice', correctedClassName: 'fried rice', confidence: 0.80, correctedAt: '2026-03-12 11:00:00' },
        { id: '3', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.70, correctedAt: '2026-03-12 13:00:00' },
        { id: '4', originalClassName: 'rice', correctedClassName: 'brown rice', confidence: 0.72, correctedAt: '2026-03-12 14:00:00' },
      ];
      mockWhere.mockResolvedValue(mockRows);

      const suggestion = await CorrectionStore.getSuggestion('rice');

      expect(suggestion).toBeNull();
    });
  });

  describe('multiple corrections tracked independently', () => {
    it('records multiple corrections for the same original class', async () => {
      await CorrectionStore.recordCorrection('rice', 'fried rice', 0.85);
      await CorrectionStore.recordCorrection('rice', 'brown rice', 0.75);

      expect(mockInsert).toHaveBeenCalledTimes(2);
      expect(mockValues).toHaveBeenCalledTimes(2);

      // Check first call
      expect(mockValues.mock.calls[0][0]).toEqual(
        expect.objectContaining({
          originalClassName: 'rice',
          correctedClassName: 'fried rice',
        }),
      );
      // Check second call
      expect(mockValues.mock.calls[1][0]).toEqual(
        expect.objectContaining({
          originalClassName: 'rice',
          correctedClassName: 'brown rice',
        }),
      );
    });
  });
});
