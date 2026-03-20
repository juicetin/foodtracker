/**
 * Time-period assignment logic for diary entries.
 *
 * Maps hours to morning/afternoon/evening with configurable boundaries.
 * Default: 6am-12pm morning, 12pm-6pm afternoon, 6pm-6am evening.
 * Midnight-to-morning-start maps to evening (late night eating).
 */

export type TimePeriod = 'morning' | 'afternoon' | 'evening';

export interface TimePeriodBoundary {
  morning: number;
  afternoon: number;
  evening: number;
}

export const DEFAULT_BOUNDARIES: TimePeriodBoundary = {
  morning: 6,
  afternoon: 12,
  evening: 18,
};

export const TIME_PERIOD_ORDER: TimePeriod[] = ['morning', 'afternoon', 'evening'];

/**
 * Assign a time period based on the hour extracted from a created_at timestamp.
 *
 * Logic: hours before morning boundary or >= evening boundary map to 'evening'.
 * Hours >= afternoon boundary map to 'afternoon'. Everything else is 'morning'.
 */
export function assignTimePeriod(
  createdAt: string,
  boundaries: TimePeriodBoundary = DEFAULT_BOUNDARIES,
): TimePeriod {
  const hour = new Date(createdAt).getHours();

  if (hour >= boundaries.evening || hour < boundaries.morning) {
    return 'evening';
  }
  if (hour >= boundaries.afternoon) {
    return 'afternoon';
  }
  return 'morning';
}

const PERIOD_LABELS: Record<TimePeriod, string> = {
  morning: 'Morning',
  afternoon: 'Afternoon',
  evening: 'Evening',
};

const PERIOD_ICONS: Record<TimePeriod, string> = {
  morning: 'sunny-outline',
  afternoon: 'partly-sunny-outline',
  evening: 'moon-outline',
};

/** Human-readable label for a time period. */
export function getTimePeriodLabel(period: TimePeriod): string {
  return PERIOD_LABELS[period];
}

/** Ionicons icon name for a time period. */
export function getTimePeriodIcon(period: TimePeriod): string {
  return PERIOD_ICONS[period];
}
