/**
 * Animated badge showing VLM refinement in progress.
 *
 * Displays a small pill-shaped "Refining..." badge with a pulsing
 * opacity animation. Returns null when not visible.
 */

import React, { useEffect } from 'react';
import { StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  FadeIn,
} from 'react-native-reanimated';

interface RefiningBadgeProps {
  visible: boolean;
}

export function RefiningBadge({ visible }: RefiningBadgeProps) {
  const opacity = useSharedValue(1);

  useEffect(() => {
    if (visible) {
      opacity.value = withRepeat(
        withTiming(0.5, { duration: 800 }),
        -1, // infinite
        true, // reverse
      );
    } else {
      opacity.value = 1;
    }
  }, [visible, opacity]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  if (!visible) return null;

  return (
    <Animated.View
      entering={FadeIn.duration(300)}
      style={[styles.badge, animatedStyle]}
    >
      <Animated.Text style={styles.text}>Refining...</Animated.Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  badge: {
    backgroundColor: '#E3F2FD',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    marginLeft: 8,
  },
  text: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1565C0',
  },
});
