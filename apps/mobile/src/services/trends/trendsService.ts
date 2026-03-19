/**
 * Trends service — load daily nutrition totals and compute trend statistics.
 *
 * Supports 7/14/30/all-time ranges with macro breakdown.
 */

import { opsqlite } from '../../../db/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DayTotals {
  date: string;
  dayLabel: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface TrendStats {
  avgCalories: number;
  avgProtein: number;
  avgCarbs: number;
  avgFat: number;
  daysLogged: number;
  totalDays: number;
  goalAdherencePct: number;
  currentStreak: number;
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/**
 * Load daily nutrition totals for the last `days` days (most recent last).
 * Pass 0 for all-time data.
 */
export function loadDailyTotals(days: number): DayTotals[] {
  if (days === 0) {
    return loadAllTimeTotals();
  }

  const result: DayTotals[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const dateStr = d.toISOString().split('T')[0];

    const rows = opsqlite.executeSync(
      `SELECT
         COALESCE(SUM(total_calories), 0) AS cal,
         COALESCE(SUM(total_protein), 0) AS pro,
         COALESCE(SUM(total_carbs), 0) AS carb,
         COALESCE(SUM(total_fat), 0) AS fat
       FROM food_entries WHERE entry_date = ? AND is_deleted = 0`,
      [dateStr],
    ).rows as Array<Record<string, unknown>>;

    const row = rows[0] ?? {};
    result.push({
      date: dateStr,
      dayLabel: DAY_NAMES[d.getDay()],
      calories: (row.cal as number) ?? 0,
      protein: (row.pro as number) ?? 0,
      carbs: (row.carb as number) ?? 0,
      fat: (row.fat as number) ?? 0,
    });
  }
  return result;
}

function loadAllTimeTotals(): DayTotals[] {
  try {
    const rows = opsqlite.executeSync(
      `SELECT entry_date,
              SUM(total_calories) AS cal,
              SUM(total_protein) AS pro,
              SUM(total_carbs) AS carb,
              SUM(total_fat) AS fat
       FROM food_entries
       WHERE is_deleted = 0
       GROUP BY entry_date
       ORDER BY entry_date ASC`,
    ).rows as Array<Record<string, unknown>>;

    return rows.map((r) => {
      const d = new Date((r.entry_date as string) + 'T12:00:00');
      return {
        date: r.entry_date as string,
        dayLabel: DAY_NAMES[d.getDay()],
        calories: (r.cal as number) ?? 0,
        protein: (r.pro as number) ?? 0,
        carbs: (r.carb as number) ?? 0,
        fat: (r.fat as number) ?? 0,
      };
    });
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/**
 * Compute trend statistics from daily totals.
 * Goal adherence = % of logged days within ±10% of calorie goal.
 */
export function computeTrendStats(days: DayTotals[], calorieGoal: number): TrendStats {
  const loggedDays = days.filter((d) => d.calories > 0);
  const daysLogged = loggedDays.length;
  const totalDays = days.length;

  if (daysLogged === 0) {
    return {
      avgCalories: 0, avgProtein: 0, avgCarbs: 0, avgFat: 0,
      daysLogged: 0, totalDays, goalAdherencePct: 0, currentStreak: 0,
    };
  }

  const avgCalories = Math.round(loggedDays.reduce((s, d) => s + d.calories, 0) / daysLogged);
  const avgProtein = Math.round(loggedDays.reduce((s, d) => s + d.protein, 0) / daysLogged * 10) / 10;
  const avgCarbs = Math.round(loggedDays.reduce((s, d) => s + d.carbs, 0) / daysLogged * 10) / 10;
  const avgFat = Math.round(loggedDays.reduce((s, d) => s + d.fat, 0) / daysLogged * 10) / 10;

  // Goal adherence: days within ±10% of goal
  const lo = calorieGoal * 0.9;
  const hi = calorieGoal * 1.1;
  const adherentDays = loggedDays.filter((d) => d.calories >= lo && d.calories <= hi).length;
  const goalAdherencePct = Math.round((adherentDays / daysLogged) * 100);

  // Current streak: consecutive logged days from the end
  let currentStreak = 0;
  for (let i = days.length - 1; i >= 0; i--) {
    if (days[i].calories > 0) {
      currentStreak++;
    } else {
      break;
    }
  }

  return { avgCalories, avgProtein, avgCarbs, avgFat, daysLogged, totalDays, goalAdherencePct, currentStreak };
}
