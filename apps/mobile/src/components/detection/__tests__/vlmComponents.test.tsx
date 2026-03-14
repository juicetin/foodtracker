/**
 * Render tests for VLM-related detection components.
 *
 * Tests MealTextInput and RefiningBadge rendering behavior
 * based on their props (disabled, visible).
 */

import React from 'react';
import { render } from '@testing-library/react-native';
import { MealTextInput } from '../MealTextInput';
import { RefiningBadge } from '../RefiningBadge';

// Mock react-native-reanimated to avoid native module errors in tests
jest.mock('react-native-reanimated', () => {
  const View = require('react-native').View;
  const Text = require('react-native').Text;
  return {
    __esModule: true,
    default: {
      View,
      Text,
      createAnimatedComponent: (component: unknown) => component,
    },
    useSharedValue: jest.fn((init: unknown) => ({ value: init })),
    useAnimatedStyle: jest.fn(() => ({})),
    withTiming: jest.fn((val: unknown) => val),
    withRepeat: jest.fn((val: unknown) => val),
    FadeIn: { duration: jest.fn(() => ({ duration: jest.fn() })) },
    // Animated components
    View,
    Text,
  };
});

describe('MealTextInput', () => {
  it('renders TextInput when not disabled', () => {
    const onChangeText = jest.fn();
    const { getByPlaceholderText } = render(
      <MealTextInput value="" onChangeText={onChangeText} />,
    );
    expect(getByPlaceholderText('Describe your meal (optional)')).toBeTruthy();
  });

  it('returns null when disabled', () => {
    const onChangeText = jest.fn();
    const { toJSON } = render(
      <MealTextInput value="" onChangeText={onChangeText} disabled />,
    );
    expect(toJSON()).toBeNull();
  });
});

describe('RefiningBadge', () => {
  it('renders text when visible', () => {
    const { getByText } = render(<RefiningBadge visible />);
    expect(getByText('Refining...')).toBeTruthy();
  });

  it('returns null when not visible', () => {
    const { toJSON } = render(<RefiningBadge visible={false} />);
    expect(toJSON()).toBeNull();
  });
});
