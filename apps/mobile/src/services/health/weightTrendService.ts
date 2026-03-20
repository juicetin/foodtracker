/**
 * Weight trend calculation with EMA smoothing.
 * Pure functions, no side effects.
 */

export interface WeightEntry {
  date: string; // YYYY-MM-DD
  weightKg: number;
  source: 'manual' | 'health_connect';
}

export interface WeightTrend {
  raw: number[];
  smoothed: number[];
  dates: string[];
  latestKg: number | null;
  trendDirection: 'up' | 'down' | 'stable';
}

/**
 * Exponential Moving Average smoothing.
 * Standard EMA: smoothed[0] = weights[0],
 *               smoothed[i] = alpha * weights[i] + (1 - alpha) * smoothed[i-1]
 *
 * @param weights - Raw weight values in chronological order
 * @param alpha - Smoothing factor (0-1). Default 0.15.
 */
export function emaSmooth(weights: number[], alpha: number = 0.15): number[] {
  if (weights.length === 0) return [];

  const smoothed: number[] = [weights[0]];
  for (let i = 1; i < weights.length; i++) {
    smoothed.push(alpha * weights[i] + (1 - alpha) * smoothed[i - 1]);
  }
  return smoothed;
}

/**
 * Calculate weight trend from entries.
 * Trend direction: compare last smoothed vs smoothed 7 entries ago (or first if <7).
 * Stable if absolute difference < 0.2kg.
 */
export function calculateWeightTrend(entries: WeightEntry[]): WeightTrend {
  if (entries.length === 0) {
    return {
      raw: [],
      smoothed: [],
      dates: [],
      latestKg: null,
      trendDirection: 'stable',
    };
  }

  const raw = entries.map((e) => e.weightKg);
  const dates = entries.map((e) => e.date);
  const smoothed = emaSmooth(raw);
  const latestKg = raw[raw.length - 1];

  // Compare last smoothed value vs 7 entries ago (or first)
  const compareIdx = Math.max(0, smoothed.length - 8);
  const diff = smoothed[smoothed.length - 1] - smoothed[compareIdx];

  let trendDirection: 'up' | 'down' | 'stable';
  if (Math.abs(diff) < 0.2) {
    trendDirection = 'stable';
  } else if (diff > 0) {
    trendDirection = 'up';
  } else {
    trendDirection = 'down';
  }

  return { raw, smoothed, dates, latestKg, trendDirection };
}
