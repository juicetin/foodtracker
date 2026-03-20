/**
 * Tests for ExpandableEntryCard three-state tap cycle.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';

// Mock react-native-reanimated
jest.mock('react-native-reanimated', () => {
  const View = require('react-native').View;
  return {
    __esModule: true,
    default: {
      View,
      createAnimatedComponent: (component: unknown) => component,
    },
    useSharedValue: jest.fn((init: unknown) => ({ value: init })),
    useAnimatedStyle: jest.fn(() => ({})),
    withTiming: jest.fn((val: unknown) => val),
    Easing: { inOut: jest.fn(() => jest.fn()), ease: {} },
    View,
  };
});

// Mock opsqlite for ingredient loading
const mockExecuteSync = jest.fn();
jest.mock('../../../../db/client', () => ({
  opsqlite: {
    executeSync: (...args: unknown[]) => mockExecuteSync(...args),
  },
}));

// Mock the preferences store (used by StickyMacroHeader but not directly here)
jest.mock('../../../store/usePreferencesStore', () => ({
  usePreferencesStore: jest.fn((selector: Function) =>
    selector({
      diaryDisplayMode: 'consumed',
      setDiaryDisplayMode: jest.fn(),
      timePeriodBoundaries: { morning: 6, afternoon: 12, evening: 18 },
    }),
  ),
}));

import { ExpandableEntryCard } from '../ExpandableEntryCard';
import type { DiaryEntry } from '../../../services/diary/diaryQueries';

const MOCK_ENTRY: DiaryEntry = {
  id: 'entry-1',
  timePeriod: 'afternoon',
  mealType: 'lunch',
  totalCalories: 650,
  totalProtein: 35,
  totalCarbs: 60,
  totalFat: 22,
  notes: null,
  createdAt: '2024-01-15T13:00:00',
  photoUri: null,
  dishes: [
    { id: 'd1', name: 'Chicken Rice Bowl', cuisine: 'Asian' },
  ],
};

const MOCK_NAVIGATE = jest.fn();

beforeEach(() => {
  mockExecuteSync.mockReset();
  MOCK_NAVIGATE.mockReset();
});

describe('ExpandableEntryCard', () => {
  it('starts in summary state with macro row visible', () => {
    const { getByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );
    expect(getByTestId('macro-row')).toBeTruthy();
  });

  it('transitions to ingredients state on first tap', () => {
    mockExecuteSync.mockReturnValueOnce({
      rows: [
        { name: 'Chicken', amount_g: 150, calories: 250, protein: 30, carbs: 0, fat: 10 },
        { name: 'Rice', amount_g: 200, calories: 260, protein: 5, carbs: 58, fat: 1 },
      ],
    });

    const { getByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );

    fireEvent.press(getByTestId('expandable-entry-card'));
    expect(getByTestId('ingredients-list')).toBeTruthy();
  });

  it('transitions to collapsed state on second tap (no macro row)', () => {
    mockExecuteSync.mockReturnValueOnce({ rows: [] });

    const { getByTestId, queryByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );

    // summary -> ingredients
    fireEvent.press(getByTestId('expandable-entry-card'));
    // ingredients -> collapsed
    fireEvent.press(getByTestId('expandable-entry-card'));
    expect(queryByTestId('macro-row')).toBeNull();
  });

  it('cycles back to summary on third tap', () => {
    mockExecuteSync.mockReturnValueOnce({ rows: [] });

    const { getByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );

    // summary -> ingredients -> collapsed -> summary
    fireEvent.press(getByTestId('expandable-entry-card'));
    fireEvent.press(getByTestId('expandable-entry-card'));
    fireEvent.press(getByTestId('expandable-entry-card'));
    expect(getByTestId('macro-row')).toBeTruthy();
  });

  it('shows photo placeholder when photoUri is null', () => {
    const { getByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );
    expect(getByTestId('photo-placeholder')).toBeTruthy();
  });

  it('renders three state dots', () => {
    const { getByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );
    const dotsRow = getByTestId('state-dots');
    expect(dotsRow.children).toHaveLength(3);
  });

  it('calls onNavigateToDetail on long press', () => {
    const { getByTestId } = render(
      <ExpandableEntryCard entry={MOCK_ENTRY} onNavigateToDetail={MOCK_NAVIGATE} />,
    );
    fireEvent(getByTestId('expandable-entry-card'), 'longPress');
    expect(MOCK_NAVIGATE).toHaveBeenCalledWith('entry-1');
  });
});
