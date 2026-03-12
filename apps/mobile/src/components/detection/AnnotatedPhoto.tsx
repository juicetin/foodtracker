import React from 'react';
import { Image, StyleSheet, useWindowDimensions, View } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface AnnotatedPhotoProps {
  photoUri: string;
  photoWidth: number;
  photoHeight: number;
  children?: React.ReactNode; // BoundingBoxOverlay goes here
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_SCALE = 1;
const MAX_SCALE = 5;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Interactive photo display with pinch-to-zoom and pan.
 *
 * Children (typically BoundingBoxOverlay) are rendered as an absolute overlay
 * that scales and translates in sync with the image.
 */
export function AnnotatedPhoto({
  photoUri,
  photoWidth,
  photoHeight,
  children,
}: AnnotatedPhotoProps) {
  const { width: screenWidth } = useWindowDimensions();

  const aspectRatio = photoWidth > 0 ? photoHeight / photoWidth : 1;
  const displayWidth = screenWidth;
  const displayHeight = screenWidth * aspectRatio;

  // Shared values for gestures
  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);

  // -- Gestures ---------------------------------------------------------------

  const pinchGesture = Gesture.Pinch()
    .onUpdate((e) => {
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, savedScale.value * e.scale));
      scale.value = newScale;
    })
    .onEnd(() => {
      savedScale.value = scale.value;
      if (scale.value < MIN_SCALE) {
        scale.value = withTiming(MIN_SCALE);
        savedScale.value = MIN_SCALE;
      }
    });

  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      // Only allow panning when zoomed in
      if (scale.value <= 1) return;

      const maxX = ((scale.value - 1) * displayWidth) / 2;
      const maxY = ((scale.value - 1) * displayHeight) / 2;

      translateX.value = Math.min(
        maxX,
        Math.max(-maxX, savedTranslateX.value + e.translationX),
      );
      translateY.value = Math.min(
        maxY,
        Math.max(-maxY, savedTranslateY.value + e.translationY),
      );
    })
    .onEnd(() => {
      savedTranslateX.value = translateX.value;
      savedTranslateY.value = translateY.value;
    });

  const doubleTapGesture = Gesture.Tap()
    .numberOfTaps(2)
    .onEnd(() => {
      // Reset zoom on double tap
      scale.value = withTiming(1);
      savedScale.value = 1;
      translateX.value = withTiming(0);
      translateY.value = withTiming(0);
      savedTranslateX.value = 0;
      savedTranslateY.value = 0;
    });

  const composed = Gesture.Simultaneous(
    pinchGesture,
    panGesture,
    doubleTapGesture,
  );

  // -- Animated style ---------------------------------------------------------

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  return (
    <View style={[styles.container, { width: displayWidth, height: displayHeight }]}>
      <GestureDetector gesture={composed}>
        <Animated.View style={[styles.inner, animatedStyle]}>
          <Image
            source={{ uri: photoUri }}
            style={{ width: displayWidth, height: displayHeight }}
            resizeMode="contain"
          />
          {/* Bounding box overlay sits on top of the image */}
          <View style={StyleSheet.absoluteFill}>{children}</View>
        </Animated.View>
      </GestureDetector>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  inner: {
    flex: 1,
  },
});
