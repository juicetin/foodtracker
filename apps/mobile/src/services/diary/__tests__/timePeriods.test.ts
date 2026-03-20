import {
  assignTimePeriod,
  getTimePeriodLabel,
  getTimePeriodIcon,
  DEFAULT_BOUNDARIES,
  TIME_PERIOD_ORDER,
  type TimePeriod,
  type TimePeriodBoundary,
} from '../timePeriods';

describe('assignTimePeriod', () => {
  it('maps 07:30 to morning with defaults', () => {
    expect(assignTimePeriod('2024-01-01T07:30:00')).toBe('morning');
  });

  it('maps 13:00 to afternoon with defaults', () => {
    expect(assignTimePeriod('2024-01-01T13:00:00')).toBe('afternoon');
  });

  it('maps 19:00 to evening with defaults', () => {
    expect(assignTimePeriod('2024-01-01T19:00:00')).toBe('evening');
  });

  it('maps 03:00 (midnight-6am) to evening with defaults', () => {
    expect(assignTimePeriod('2024-01-01T03:00:00')).toBe('evening');
  });

  it('respects custom boundaries', () => {
    const custom: TimePeriodBoundary = { morning: 5, afternoon: 11, evening: 17 };
    expect(assignTimePeriod('2024-01-01T10:00:00', custom)).toBe('morning');
  });

  it('maps boundary hours correctly (exactly at morning start)', () => {
    expect(assignTimePeriod('2024-01-01T06:00:00')).toBe('morning');
  });

  it('maps boundary hours correctly (exactly at afternoon start)', () => {
    expect(assignTimePeriod('2024-01-01T12:00:00')).toBe('afternoon');
  });

  it('maps boundary hours correctly (exactly at evening start)', () => {
    expect(assignTimePeriod('2024-01-01T18:00:00')).toBe('evening');
  });
});

describe('getTimePeriodLabel', () => {
  it('returns Morning for morning', () => {
    expect(getTimePeriodLabel('morning')).toBe('Morning');
  });

  it('returns Afternoon for afternoon', () => {
    expect(getTimePeriodLabel('afternoon')).toBe('Afternoon');
  });

  it('returns Evening for evening', () => {
    expect(getTimePeriodLabel('evening')).toBe('Evening');
  });
});

describe('getTimePeriodIcon', () => {
  it('returns sunny-outline for morning', () => {
    expect(getTimePeriodIcon('morning')).toBe('sunny-outline');
  });

  it('returns partly-sunny-outline for afternoon', () => {
    expect(getTimePeriodIcon('afternoon')).toBe('partly-sunny-outline');
  });

  it('returns moon-outline for evening', () => {
    expect(getTimePeriodIcon('evening')).toBe('moon-outline');
  });
});

describe('constants', () => {
  it('DEFAULT_BOUNDARIES has expected values', () => {
    expect(DEFAULT_BOUNDARIES).toEqual({ morning: 6, afternoon: 12, evening: 18 });
  });

  it('TIME_PERIOD_ORDER is morning, afternoon, evening', () => {
    expect(TIME_PERIOD_ORDER).toEqual(['morning', 'afternoon', 'evening']);
  });
});
