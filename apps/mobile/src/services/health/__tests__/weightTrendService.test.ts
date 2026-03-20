import {
  emaSmooth,
  calculateWeightTrend,
  type WeightEntry,
} from '../weightTrendService';

describe('weightTrendService', () => {
  describe('emaSmooth', () => {
    it('returns correct EMA values for [80, 79.5, 80.2, 79.8] with alpha=0.15', () => {
      const result = emaSmooth([80, 79.5, 80.2, 79.8], 0.15);

      // EMA: s[0]=80
      // s[1] = 0.15*79.5 + 0.85*80 = 11.925 + 68 = 79.925
      // s[2] = 0.15*80.2 + 0.85*79.925 = 12.03 + 67.93625 = 79.96625
      // s[3] = 0.15*79.8 + 0.85*79.96625 = 11.97 + 67.9713125 = 79.9413125
      expect(result).toHaveLength(4);
      expect(result[0]).toBeCloseTo(80, 5);
      expect(result[1]).toBeCloseTo(79.925, 3);
      expect(result[2]).toBeCloseTo(79.96625, 3);
      expect(result[3]).toBeCloseTo(79.94131, 3);
    });

    it('returns empty array for empty input', () => {
      expect(emaSmooth([])).toEqual([]);
    });

    it('returns single-element array unchanged', () => {
      expect(emaSmooth([75])).toEqual([75]);
    });

    it('uses default alpha=0.15', () => {
      const withDefault = emaSmooth([80, 79]);
      const withExplicit = emaSmooth([80, 79], 0.15);
      expect(withDefault).toEqual(withExplicit);
    });
  });

  describe('calculateWeightTrend', () => {
    it('returns { raw, smoothed, dates, latestKg, trendDirection } from weight entries', () => {
      const entries: WeightEntry[] = [
        { date: '2025-01-10', weightKg: 81, source: 'manual' },
        { date: '2025-01-11', weightKg: 80.5, source: 'health_connect' },
        { date: '2025-01-12', weightKg: 80.2, source: 'manual' },
        { date: '2025-01-13', weightKg: 79.8, source: 'health_connect' },
        { date: '2025-01-14', weightKg: 79.5, source: 'manual' },
        { date: '2025-01-15', weightKg: 79.3, source: 'manual' },
        { date: '2025-01-16', weightKg: 79.0, source: 'manual' },
        { date: '2025-01-17', weightKg: 78.8, source: 'health_connect' },
      ];

      const result = calculateWeightTrend(entries);

      expect(result.raw).toHaveLength(8);
      expect(result.smoothed).toHaveLength(8);
      expect(result.dates).toHaveLength(8);
      expect(result.latestKg).toBe(78.8);
      expect(result.trendDirection).toBe('down');
    });

    it('returns null latestKg for empty entries', () => {
      const result = calculateWeightTrend([]);
      expect(result.latestKg).toBeNull();
      expect(result.raw).toEqual([]);
      expect(result.smoothed).toEqual([]);
      expect(result.dates).toEqual([]);
      expect(result.trendDirection).toBe('stable');
    });

    it('returns stable when trend diff is less than 0.2kg', () => {
      const entries: WeightEntry[] = [
        { date: '2025-01-10', weightKg: 80.0, source: 'manual' },
        { date: '2025-01-11', weightKg: 80.05, source: 'manual' },
        { date: '2025-01-12', weightKg: 80.0, source: 'manual' },
        { date: '2025-01-13', weightKg: 79.95, source: 'manual' },
        { date: '2025-01-14', weightKg: 80.0, source: 'manual' },
        { date: '2025-01-15', weightKg: 80.05, source: 'manual' },
        { date: '2025-01-16', weightKg: 80.0, source: 'manual' },
        { date: '2025-01-17', weightKg: 79.95, source: 'manual' },
      ];

      const result = calculateWeightTrend(entries);
      expect(result.trendDirection).toBe('stable');
    });

    it('detects upward trend', () => {
      const entries: WeightEntry[] = [
        { date: '2025-01-10', weightKg: 78.0, source: 'manual' },
        { date: '2025-01-11', weightKg: 78.5, source: 'manual' },
        { date: '2025-01-12', weightKg: 79.0, source: 'manual' },
        { date: '2025-01-13', weightKg: 79.5, source: 'manual' },
        { date: '2025-01-14', weightKg: 80.0, source: 'manual' },
        { date: '2025-01-15', weightKg: 80.5, source: 'manual' },
        { date: '2025-01-16', weightKg: 81.0, source: 'manual' },
        { date: '2025-01-17', weightKg: 81.5, source: 'manual' },
      ];

      const result = calculateWeightTrend(entries);
      expect(result.trendDirection).toBe('up');
    });
  });
});
