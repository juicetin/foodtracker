import React, { useCallback } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Slider from '@react-native-community/slider';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PortionSliderProps {
  baseWeightG: number;
  multiplier: number; // 0.5 - 3.0
  onMultiplierChange: (multiplier: number) => void;
  confidence: string; // portion estimate confidence
  method?: string; // estimation method
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_MULTIPLIER = 0.5;
const MAX_MULTIPLIER = 3.0;
const STEP = 0.1;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Horizontal slider for adjusting portion size from 0.5x to 3.0x.
 *
 * Per locked decision: "Portion adjustment via slider in the detail card:
 * 0.5x to 3x of estimated portion, macros update in real-time."
 *
 * Displays current weight in grams and the multiplier value. When confidence
 * is "low" and method is not "geometry", shows an "estimated" badge.
 */
export function PortionSlider({
  baseWeightG,
  multiplier,
  onMultiplierChange,
  confidence,
  method,
}: PortionSliderProps) {
  const currentWeightG = Math.round(baseWeightG * multiplier);
  const showEstimatedBadge = confidence === 'low' && method !== 'geometry';

  const handleValueChange = useCallback(
    (value: number) => {
      // Round to nearest step to avoid floating point drift
      const rounded = Math.round(value * 10) / 10;
      onMultiplierChange(rounded);
    },
    [onMultiplierChange],
  );

  return (
    <View style={styles.container}>
      {/* Weight display */}
      <View style={styles.weightRow}>
        <Text style={styles.weightText}>{currentWeightG}g</Text>
        <Text style={styles.multiplierText}>{multiplier.toFixed(1)}x</Text>
        {showEstimatedBadge && (
          <View style={styles.estimatedBadge}>
            <Text style={styles.estimatedText}>estimated</Text>
          </View>
        )}
      </View>

      {/* Slider */}
      <View style={styles.sliderRow}>
        <Text style={styles.rangeLabel}>0.5x</Text>
        <Slider
          style={styles.slider}
          minimumValue={MIN_MULTIPLIER}
          maximumValue={MAX_MULTIPLIER}
          step={STEP}
          value={multiplier}
          onValueChange={handleValueChange}
          minimumTrackTintColor="#22C55E"
          maximumTrackTintColor="#E0E0E0"
          thumbTintColor="#22C55E"
        />
        <Text style={styles.rangeLabel}>3.0x</Text>
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    paddingVertical: 8,
  },
  weightRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  weightText: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1A1A1A',
    marginRight: 8,
  },
  multiplierText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
  },
  estimatedBadge: {
    marginLeft: 8,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    backgroundColor: '#FFF3E0',
  },
  estimatedText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#E65100',
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  slider: {
    flex: 1,
    height: 40,
  },
  rangeLabel: {
    fontSize: 11,
    color: '#999',
    width: 30,
    textAlign: 'center',
  },
});
