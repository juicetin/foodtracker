/**
 * Tests for ScaleInputScreen -- verifies onResult callback is invoked
 * with netWeight before goBack(), and that missing onResult does not throw.
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockGoBack = jest.fn();
const mockCanGoBack = jest.fn(() => true);
const mockOnResult = jest.fn();

jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({
    goBack: mockGoBack,
    canGoBack: mockCanGoBack,
  }),
  useRoute: () => ({
    params: {
      onResult: mockOnResult,
    },
  }),
}));

jest.mock('../../services/scale/scaleOcrService', () => ({
  readScaleWeight: jest.fn().mockResolvedValue(null),
}));

jest.mock('../../services/scale/containerService', () => ({
  addContainer: jest.fn(),
  applyTare: jest.fn((gross: number) => gross),
  deleteContainer: jest.fn(),
  getContainers: jest.fn().mockResolvedValue([]),
  recordContainerUsage: jest.fn(),
}));

jest.mock('react-native-safe-area-context', () => {
  const RN = require('react-native');
  const RR = require('react');
  return {
    SafeAreaView: (props: any) => RR.createElement(RN.View, props),
  };
});

import ScaleInputScreen from '../ScaleInputScreen';

describe('ScaleInputScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('calls onResult with netWeight before goBack on confirm', async () => {
    const { getByPlaceholderText, getByText } = render(<ScaleInputScreen />);

    // Enter a weight
    const input = getByPlaceholderText('0.0');
    fireEvent.changeText(input, '250');

    // Press confirm button
    const confirmBtn = getByText(/Confirm/);
    fireEvent.press(confirmBtn);

    await waitFor(() => {
      expect(mockOnResult).toHaveBeenCalledWith(250);
      expect(mockGoBack).toHaveBeenCalled();
    });

    // Verify order: onResult called before goBack
    const onResultOrder = mockOnResult.mock.invocationCallOrder[0];
    const goBackOrder = mockGoBack.mock.invocationCallOrder[0];
    expect(onResultOrder).toBeLessThan(goBackOrder);
  });

  it('does not throw when onResult is undefined', async () => {
    // Override useRoute to return no onResult
    jest.spyOn(
      require('@react-navigation/native'),
      'useRoute',
    ).mockReturnValue({
      params: {},
    });

    const { getByPlaceholderText, getByText } = render(<ScaleInputScreen />);

    const input = getByPlaceholderText('0.0');
    fireEvent.changeText(input, '100');

    const confirmBtn = getByText(/Confirm/);
    fireEvent.press(confirmBtn);

    await waitFor(() => {
      expect(mockGoBack).toHaveBeenCalled();
    });
  });
});
