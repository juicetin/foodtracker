import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  runOnJS,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import {
  CONFIDENCE_COLORS,
  DetectedItem,
  getConfidenceLevel,
} from '../../services/detection/types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DetectionListItemProps {
  item: DetectedItem;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Distance (px) the user must swipe to trigger dismiss. */
const SWIPE_THRESHOLD = 120;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Single detection item row with swipe-to-dismiss.
 *
 * Layout: [confidence dot] [food name] [portion estimate] [confidence %]
 *
 * Per locked decision:
 * - Grams as primary unit, descriptive as secondary
 * - Swipe-to-dismiss as power-user shortcut
 * - Tapping highlights corresponding bounding box
 */
export function DetectionListItem({
  item,
  isSelected,
  onSelect,
  onRemove,
}: DetectionListItemProps) {
  const level = getConfidenceLevel(item.confidence);
  const color = CONFIDENCE_COLORS[level];
  const translateX = useSharedValue(0);

  const panGesture = Gesture.Pan()
    .activeOffsetX([-10, 10])
    .failOffsetY([-5, 5])
    .onUpdate((e) => {
      translateX.value = e.translationX;
    })
    .onEnd((e) => {
      if (Math.abs(e.translationX) > SWIPE_THRESHOLD) {
        // Animate off-screen then remove
        translateX.value = withTiming(
          e.translationX > 0 ? 400 : -400,
          { duration: 200 },
          () => runOnJS(onRemove)(item.id),
        );
      } else {
        // Snap back
        translateX.value = withSpring(0);
      }
    });

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
  }));

  const portionText = `~${Math.round(item.portionEstimate.weightG * item.portionMultiplier)}g`;

  return (
    <GestureDetector gesture={panGesture}>
      <Animated.View style={animatedStyle}>
        <Pressable
          onPress={() => onSelect(item.id)}
          style={[
            styles.container,
            isSelected && styles.selectedContainer,
          ]}
        >
          {/* Confidence dot */}
          <View style={[styles.dot, { backgroundColor: color }]} />

          {/* Food name */}
          <View style={styles.nameContainer}>
            <Text style={styles.name} numberOfLines={1}>
              {item.className}
            </Text>
            <Text style={styles.portion}>{portionText}</Text>
          </View>

          {/* Confidence percentage */}
          <Text style={[styles.confidence, { color }]}>
            {Math.round(item.confidence * 100)}%
          </Text>
        </Pressable>
      </Animated.View>
    </GestureDetector>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#fff',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E0E0E0',
  },
  selectedContainer: {
    backgroundColor: '#F0F7FF',
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 12,
  },
  nameContainer: {
    flex: 1,
    marginRight: 8,
  },
  name: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1A1A1A',
  },
  portion: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  confidence: {
    fontSize: 13,
    fontWeight: '600',
    minWidth: 36,
    textAlign: 'right',
  },
});
