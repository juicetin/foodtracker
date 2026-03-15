/**
 * Render tests for VLM-related detection components.
 *
 * Tests MealTextInput and ShimmerPlaceholder rendering behavior.
 */

import React from 'react';
import { render } from '@testing-library/react-native';
import { MealTextInput } from '../MealTextInput';
import { ShimmerPlaceholder } from '../ShimmerPlaceholder';

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
    FadeOut: { duration: jest.fn(() => ({ duration: jest.fn() })) },
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

describe('ShimmerPlaceholder', () => {
  it('renders with given dimensions', () => {
    const { toJSON } = render(<ShimmerPlaceholder width={100} height={14} />);
    const tree = toJSON();
    expect(tree).not.toBeNull();
  });

  it('applies custom borderRadius', () => {
    const { toJSON } = render(
      <ShimmerPlaceholder width={80} height={12} borderRadius={10} />,
    );
    expect(toJSON()).not.toBeNull();
  });
});
