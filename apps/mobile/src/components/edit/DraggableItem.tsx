/**
 * DraggableItem -- Draggable card for dishes and ingredients in merge view.
 *
 * Uses Gesture.Pan from react-native-gesture-handler with activeOffsetX
 * to require horizontal intent before drag activates (prevents conflict
 * with vertical scroll per project pattern from DiaryScreen).
 *
 * When dragged past midpoint of screen, snaps to other column on release.
 */

import React from 'react';
import { Dimensions, StyleSheet, Text, View } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import type { MergeItem } from '../../services/entryEditor/reidentifyService';

const SCREEN_WIDTH = Dimensions.get('window').width;
const SNAP_THRESHOLD = SCREEN_WIDTH * 0.25; // drag 25% of screen to snap

interface DraggableItemProps {
  item: MergeItem;
  onDragComplete: (item: MergeItem, side: 'left' | 'right') => void;
  currentSide: 'left' | 'right';
}

export function DraggableItem({ item, onDragComplete, currentSide }: DraggableItemProps) {
  const translateX = useSharedValue(0);

  const handleDragComplete = (side: 'left' | 'right') => {
    onDragComplete(item, side);
  };

  const pan = Gesture.Pan()
    .activeOffsetX([-10, 10])
    .onUpdate((e) => {
      translateX.value = e.translationX;
    })
    .onEnd((e) => {
      const draggedRight = e.translationX > SNAP_THRESHOLD;
      const draggedLeft = e.translationX < -SNAP_THRESHOLD;

      if (currentSide === 'left' && draggedRight) {
        // Dragged from left (discard) to right (keep)
        runOnJS(handleDragComplete)('right');
      } else if (currentSide === 'right' && draggedLeft) {
        // Dragged from right (keep) to left (discard)
        runOnJS(handleDragComplete)('left');
      }

      translateX.value = withSpring(0, { damping: 15, stiffness: 150 });
    });

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
  }));

  const sourceBadgeStyle = item.source === 'new' ? styles.sourceBadgeNew : styles.sourceBadgeExisting;
  const sourceBadgeTextStyle = item.source === 'new' ? styles.sourceBadgeTextNew : styles.sourceBadgeTextExisting;

  return (
    <GestureDetector gesture={pan}>
      <Animated.View style={[styles.card, animatedStyle]}>
        <View style={styles.cardHeader}>
          <Text style={styles.itemName} numberOfLines={1}>{item.name}</Text>
          <View style={sourceBadgeStyle}>
            <Text style={sourceBadgeTextStyle}>{item.source}</Text>
          </View>
        </View>
        <View style={styles.cardDetails}>
          <Text style={styles.detailText}>{Math.round(item.amountG)}g</Text>
          <Text style={styles.detailSep}> · </Text>
          <Text style={styles.detailText}>{Math.round(item.calories)} kcal</Text>
        </View>
        {item.dishName ? (
          <Text style={styles.dishLabel} numberOfLines={1}>{item.dishName}</Text>
        ) : null}
      </Animated.View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#FFF',
    borderRadius: 10,
    padding: 10,
    marginBottom: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  itemName: {
    fontSize: 13,
    fontWeight: '600',
    color: '#111827',
    flex: 1,
    marginRight: 6,
  },
  sourceBadgeNew: {
    backgroundColor: '#DCFCE7',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  sourceBadgeTextNew: {
    fontSize: 10,
    fontWeight: '600',
    color: '#16A34A',
  },
  sourceBadgeExisting: {
    backgroundColor: '#F3F4F6',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  sourceBadgeTextExisting: {
    fontSize: 10,
    fontWeight: '600',
    color: '#6B7280',
  },
  cardDetails: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  detailText: {
    fontSize: 12,
    color: '#6B7280',
  },
  detailSep: {
    fontSize: 12,
    color: '#D1D5DB',
  },
  dishLabel: {
    fontSize: 11,
    color: '#9CA3AF',
    marginTop: 2,
    fontStyle: 'italic',
  },
});
