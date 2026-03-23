/**
 * Reusable shimmer placeholder component.
 *
 * Uses react-native-reanimated to pulse opacity between 1.0 and 0.4
 * with an 800ms cycle. Renders as a rounded rectangle that can replace
 * text labels while VLM identification is in progress.
 *
 * Based on the same Reanimated patterns used by the now-deleted RefiningBadge.
 */

import React from 'react';
import type { ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  FadeIn,
  FadeOut,
} from 'react-native-reanimated';
import { useTheme } from '../../theme/ThemeProvider';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ShimmerPlaceholderProps {
  width: number | string;
  height: number;
  style?: ViewStyle;
  borderRadius?: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ShimmerPlaceholder({
  width,
  height,
  style,
  borderRadius = 6,
}: ShimmerPlaceholderProps) {
  const { colors } = useTheme();
  const opacity = useSharedValue(1);

  // Start infinite pulse on mount
  React.useEffect(() => {
    opacity.value = withRepeat(
      withTiming(0.4, { duration: 800 }),
      -1, // infinite
      true, // reverse
    );
  }, [opacity]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  return (
    <Animated.View
      entering={FadeIn.duration(200)}
      exiting={FadeOut.duration(200)}
      style={[
        {
          width,
          height,
          borderRadius,
          backgroundColor: colors.shimmer.base,
        },
        animatedStyle,
        style,
      ]}
    />
  );
}
