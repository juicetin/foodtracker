import { getRecentHistory, searchHistory } from '../historyService';

// Mock opsqlite
jest.mock('../../../../db/client', () => ({
  opsqlite: {
    execute: jest.fn(),
  },
}));

import { opsqlite } from '../../../../db/client';

const mockExecute = opsqlite.execute as jest.Mock;

describe('getRecentHistory', () => {
  beforeEach(() => {
    mockExecute.mockReset();
  });

  it('returns sorted results from database', () => {
    mockExecute.mockReturnValue({
      rows: [
        {
          name: 'Chicken Rice',
          total_count: 5,
          last_logged: '2026-03-20T12:00:00Z',
          avg_calories: 450,
          avg_protein: 35,
          avg_carbs: 50,
          avg_fat: 12,
        },
        {
          name: 'Oatmeal',
          total_count: 3,
          last_logged: '2026-03-19T08:00:00Z',
          avg_calories: 300,
          avg_protein: 10,
          avg_carbs: 55,
          avg_fat: 8,
        },
      ],
    });

    const result = getRecentHistory(20);
    expect(result).toHaveLength(2);
    expect(result[0].name).toBe('Chicken Rice');
    expect(result[0].totalCount).toBe(5);
    expect(result[0].avgCalories).toBe(450);
    expect(result[1].name).toBe('Oatmeal');
    expect(mockExecute).toHaveBeenCalledTimes(1);
    // Verify SQL excludes deleted entries
    const sql = mockExecute.mock.calls[0][0] as string;
    expect(sql).toContain('is_deleted = 0');
  });

  it('handles empty database', () => {
    mockExecute.mockReturnValue({ rows: [] });

    const result = getRecentHistory();
    expect(result).toEqual([]);
  });

  it('strips prefixes from food names', () => {
    mockExecute.mockReturnValue({
      rows: [
        {
          name: 'Copied: Grilled Salmon',
          total_count: 2,
          last_logged: '2026-03-18T19:00:00Z',
          avg_calories: 350,
          avg_protein: 40,
          avg_carbs: 0,
          avg_fat: 18,
        },
        {
          name: 'Quick Add: 500 kcal',
          total_count: 1,
          last_logged: '2026-03-17T12:00:00Z',
          avg_calories: 500,
          avg_protein: 0,
          avg_carbs: 0,
          avg_fat: 0,
        },
      ],
    });

    const result = getRecentHistory();
    expect(result[0].name).toBe('Grilled Salmon');
    expect(result[1].name).toBe('500 kcal');
  });
});

describe('searchHistory', () => {
  beforeEach(() => {
    mockExecute.mockReset();
  });

  it('filters results by query (case-insensitive)', () => {
    mockExecute.mockReturnValue({
      rows: [
        {
          name: 'Chicken Rice',
          total_count: 5,
          last_logged: '2026-03-20T12:00:00Z',
          avg_calories: 450,
          avg_protein: 35,
          avg_carbs: 50,
          avg_fat: 12,
        },
        {
          name: 'Oatmeal',
          total_count: 3,
          last_logged: '2026-03-19T08:00:00Z',
          avg_calories: 300,
          avg_protein: 10,
          avg_carbs: 55,
          avg_fat: 8,
        },
        {
          name: 'Chicken Soup',
          total_count: 2,
          last_logged: '2026-03-18T19:00:00Z',
          avg_calories: 200,
          avg_protein: 15,
          avg_carbs: 20,
          avg_fat: 5,
        },
      ],
    });

    const result = searchHistory('chicken');
    expect(result).toHaveLength(2);
    expect(result[0].name).toBe('Chicken Rice');
    expect(result[1].name).toBe('Chicken Soup');
  });

  it('returns empty array when no matches', () => {
    mockExecute.mockReturnValue({ rows: [] });

    const result = searchHistory('pizza');
    expect(result).toEqual([]);
  });
});
