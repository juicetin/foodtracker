/**
 * Tests for diaryQueries service.
 * Mocks opsqlite.executeSync to avoid real DB access.
 */

const mockExecuteSync = jest.fn();
jest.mock('../../../../db/client', () => ({
  opsqlite: {
    executeSync: (...args: unknown[]) => mockExecuteSync(...args),
  },
}));

import {
  loadEntriesForDate,
  loadWeekEntryPresence,
  computeDayTotals,
  getTodayDateStr,
  dateToStr,
  formatDateLabel,
  type DiaryEntry,
} from '../diaryQueries';

beforeEach(() => {
  mockExecuteSync.mockReset();
});

describe('loadEntriesForDate', () => {
  it('returns entries with timePeriod field assigned from created_at', () => {
    // First call: food_entries query
    mockExecuteSync.mockReturnValueOnce({
      rows: [
        {
          id: 'e1',
          meal_type: 'lunch',
          total_calories: 500,
          total_protein: 30,
          total_carbs: 40,
          total_fat: 20,
          notes: null,
          created_at: '2024-01-15T13:30:00',
        },
      ],
    });
    // Second call: photos for e1
    mockExecuteSync.mockReturnValueOnce({
      rows: [{ uri: 'file://photo1.jpg' }],
    });
    // Third call: dishes for e1
    mockExecuteSync.mockReturnValueOnce({
      rows: [{ id: 'd1', name: 'Pasta', cuisine: 'Italian' }],
    });

    const entries = loadEntriesForDate('2024-01-15');
    expect(entries).toHaveLength(1);
    expect(entries[0].timePeriod).toBe('afternoon');
    expect(entries[0].id).toBe('e1');
    expect(entries[0].totalCalories).toBe(500);
    expect(entries[0].photoUri).toBe('file://photo1.jpg');
    expect(entries[0].dishes).toEqual([{ id: 'd1', name: 'Pasta', cuisine: 'Italian' }]);
  });

  it('orders entries by created_at ASC', () => {
    mockExecuteSync.mockReturnValueOnce({ rows: [] });
    loadEntriesForDate('2024-01-15');
    expect(mockExecuteSync.mock.calls[0][0]).toMatch(/ORDER BY created_at ASC/i);
  });

  it('assigns null photoUri when no photos exist', () => {
    mockExecuteSync.mockReturnValueOnce({
      rows: [
        {
          id: 'e2',
          meal_type: 'breakfast',
          total_calories: 200,
          total_protein: 10,
          total_carbs: 30,
          total_fat: 5,
          notes: 'Quick bite',
          created_at: '2024-01-15T07:00:00',
        },
      ],
    });
    mockExecuteSync.mockReturnValueOnce({ rows: [] }); // no photos
    mockExecuteSync.mockReturnValueOnce({ rows: [] }); // no dishes

    const entries = loadEntriesForDate('2024-01-15');
    expect(entries[0].photoUri).toBeNull();
    expect(entries[0].timePeriod).toBe('morning');
  });
});

describe('loadWeekEntryPresence', () => {
  it('returns Map with date strings as keys', () => {
    mockExecuteSync.mockReturnValueOnce({
      rows: [
        { entry_date: '2024-01-15', count: 3 },
        { entry_date: '2024-01-17', count: 1 },
      ],
    });

    const result = loadWeekEntryPresence('2024-01-17'); // Wednesday
    expect(result).toBeInstanceOf(Map);
    expect(result.get('2024-01-15')).toBe(3);
    expect(result.get('2024-01-17')).toBe(1);
    expect(result.has('2024-01-16')).toBe(false);
  });

  it('calculates week boundaries as Mon-Sun', () => {
    mockExecuteSync.mockReturnValueOnce({ rows: [] });
    loadWeekEntryPresence('2024-01-17'); // Wednesday
    // Monday of that week is 2024-01-15, Sunday is 2024-01-21
    const params = mockExecuteSync.mock.calls[0][1];
    expect(params![0]).toBe('2024-01-15'); // Monday
    expect(params![1]).toBe('2024-01-21'); // Sunday
  });
});

describe('computeDayTotals', () => {
  it('sums calories/protein/carbs/fat from entry array', () => {
    const entries: DiaryEntry[] = [
      { id: '1', timePeriod: 'morning', mealType: 'breakfast', totalCalories: 300, totalProtein: 20, totalCarbs: 30, totalFat: 10, notes: null, createdAt: '', photoUri: null, dishes: [] },
      { id: '2', timePeriod: 'afternoon', mealType: 'lunch', totalCalories: 500, totalProtein: 35, totalCarbs: 50, totalFat: 15, notes: null, createdAt: '', photoUri: null, dishes: [] },
    ];

    const totals = computeDayTotals(entries);
    expect(totals.calories).toBe(800);
    expect(totals.protein).toBe(55);
    expect(totals.carbs).toBe(80);
    expect(totals.fat).toBe(25);
  });

  it('returns zeros for empty array', () => {
    const totals = computeDayTotals([]);
    expect(totals).toEqual({ calories: 0, protein: 0, carbs: 0, fat: 0 });
  });
});

describe('utility functions', () => {
  it('getTodayDateStr returns YYYY-MM-DD format', () => {
    const result = getTodayDateStr();
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it('dateToStr formats a Date to YYYY-MM-DD', () => {
    expect(dateToStr(new Date('2024-06-15T12:00:00'))).toBe('2024-06-15');
  });

  it('formatDateLabel returns Today for today', () => {
    expect(formatDateLabel(getTodayDateStr())).toBe('Today');
  });
});
