/**
 * Trends service tests — load daily totals for variable time ranges with macro breakdown.
 */

import {
  loadDailyTotals,
  computeTrendStats,
  type DayTotals,
  type TrendStats,
} from '../trendsService';

// ---------------------------------------------------------------------------
// Mock opsqlite
// ---------------------------------------------------------------------------

const mockExecute = jest.fn();

jest.mock('../../../../db/client', () => ({
  opsqlite: {
    executeSync: (...args: unknown[]) => mockExecute(...args),
  },
}));

beforeEach(() => {
  mockExecute.mockReset();
});

// ---------------------------------------------------------------------------
// loadDailyTotals
// ---------------------------------------------------------------------------

describe('loadDailyTotals', () => {
  it('returns correct number of days for 7-day range', () => {
    // Mock each day query
    for (let i = 0; i < 7; i++) {
      mockExecute.mockReturnValueOnce({
        rows: [{ cal: 1800, pro: 120, carb: 200, fat: 60 }],
      });
    }

    const result = loadDailyTotals(7);
    expect(result).toHaveLength(7);
    expect(mockExecute).toHaveBeenCalledTimes(7);
  });

  it('returns correct number of days for 30-day range', () => {
    for (let i = 0; i < 30; i++) {
      mockExecute.mockReturnValueOnce({
        rows: [{ cal: 2000, pro: 150, carb: 220, fat: 65 }],
      });
    }

    const result = loadDailyTotals(30);
    expect(result).toHaveLength(30);
  });

  it('handles days with no entries (null values)', () => {
    mockExecute.mockReturnValueOnce({
      rows: [{ cal: null, pro: null, carb: null, fat: null }],
    });

    const result = loadDailyTotals(1);
    expect(result[0].calories).toBe(0);
    expect(result[0].protein).toBe(0);
    expect(result[0].carbs).toBe(0);
    expect(result[0].fat).toBe(0);
  });

  it('includes date and dayLabel for each entry', () => {
    mockExecute.mockReturnValueOnce({
      rows: [{ cal: 1500, pro: 100, carb: 180, fat: 50 }],
    });

    const result = loadDailyTotals(1);
    expect(result[0].date).toBeTruthy();
    expect(result[0].dayLabel).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// computeTrendStats
// ---------------------------------------------------------------------------

describe('computeTrendStats', () => {
  const sampleDays: DayTotals[] = [
    { date: '2026-03-13', dayLabel: 'Fri', calories: 2000, protein: 150, carbs: 200, fat: 65 },
    { date: '2026-03-14', dayLabel: 'Sat', calories: 2200, protein: 130, carbs: 250, fat: 70 },
    { date: '2026-03-15', dayLabel: 'Sun', calories: 1800, protein: 140, carbs: 190, fat: 55 },
    { date: '2026-03-16', dayLabel: 'Mon', calories: 0, protein: 0, carbs: 0, fat: 0 }, // no entries
    { date: '2026-03-17', dayLabel: 'Tue', calories: 2100, protein: 160, carbs: 210, fat: 68 },
  ];

  it('calculates average calories (excluding zero days)', () => {
    const stats = computeTrendStats(sampleDays, 2000);
    // Average of 2000+2200+1800+2100 = 8100/4 = 2025 (skip zero day)
    expect(stats.avgCalories).toBe(2025);
  });

  it('calculates average macros (excluding zero days)', () => {
    const stats = computeTrendStats(sampleDays, 2000);
    expect(stats.avgProtein).toBe(145); // (150+130+140+160)/4
    expect(stats.avgCarbs).toBeCloseTo(212.5);
    expect(stats.avgFat).toBeCloseTo(64.5);
  });

  it('calculates goal adherence percentage', () => {
    const stats = computeTrendStats(sampleDays, 2000);
    // Days within ±10% of goal (1800-2200): 2000, 2200, 1800, 2100 = 4 logged days
    // All 4 are within range → 100%
    expect(stats.goalAdherencePct).toBe(100);
  });

  it('counts logged days correctly', () => {
    const stats = computeTrendStats(sampleDays, 2000);
    expect(stats.daysLogged).toBe(4); // excludes the 0-calorie day
    expect(stats.totalDays).toBe(5);
  });

  it('handles all-zero data', () => {
    const zeroDays: DayTotals[] = [
      { date: '2026-03-19', dayLabel: 'Wed', calories: 0, protein: 0, carbs: 0, fat: 0 },
    ];
    const stats = computeTrendStats(zeroDays, 2000);
    expect(stats.avgCalories).toBe(0);
    expect(stats.daysLogged).toBe(0);
    expect(stats.goalAdherencePct).toBe(0);
  });

  it('calculates streak from recent consecutive logged days', () => {
    const stats = computeTrendStats(sampleDays, 2000);
    // Days: Fri(logged), Sat(logged), Sun(logged), Mon(zero), Tue(logged)
    // Streak from end: Tue=1 (Mon breaks it)
    expect(stats.currentStreak).toBe(1);
  });
});
