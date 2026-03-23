/**
 * MealGroupSection -- collapsible meal group with entries.
 *
 * Renders MealGroupHeader + FoodItemCard for each entry.
 * Animated expand/collapse via Reanimated.
 */

import React, { useCallback, useState } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withTiming } from 'react-native-reanimated';
import { type MealGroup } from '../../services/diary/mealGroups';
import { computeMealGroupTotals } from '../../services/diary/mealGroups';
import type { DiaryEntry } from '../../services/diary/diaryQueries';
import { MealGroupHeader } from './MealGroupHeader';
import { FoodItemCard } from './FoodItemCard';

interface MealGroupSectionProps {
  mealGroup: MealGroup;
  entries: DiaryEntry[];
  onAddFood: (mealGroup: MealGroup) => void;
  onItemPress: (entryId: string) => void;
  onItemLongPress: (entry: DiaryEntry) => void;
  onHeaderLongPress: (mealGroup: MealGroup) => void;
}

export function MealGroupSection({
  mealGroup,
  entries,
  onAddFood,
  onItemPress,
  onItemLongPress,
  onHeaderLongPress,
}: MealGroupSectionProps) {
  const [expanded, setExpanded] = useState(true);
  const animProgress = useSharedValue(1);

  const subtotals = computeMealGroupTotals(entries);

  const handleToggle = useCallback(() => {
    const nextExpanded = !expanded;
    animProgress.value = withTiming(nextExpanded ? 1 : 0, { duration: 250 });
    setExpanded(nextExpanded);
  }, [expanded, animProgress]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: animProgress.value,
    maxHeight: animProgress.value === 0 ? 0 : undefined,
    overflow: 'hidden' as const,
  }));

  return (
    <View style={styles.container}>
      <MealGroupHeader
        mealGroup={mealGroup}
        subtotals={subtotals}
        expanded={expanded}
        onToggle={handleToggle}
        onLongPress={() => onHeaderLongPress(mealGroup)}
        onAddFood={() => onAddFood(mealGroup)}
      />

      <Animated.View style={animatedStyle}>
        {entries.map((entry) => (
          <FoodItemCard
            key={entry.id}
            entry={entry}
            onPress={() => onItemPress(entry.id)}
            onLongPress={() => onItemLongPress(entry)}
          />
        ))}

        {entries.length === 0 && (
          <Pressable
            onPress={() => onAddFood(mealGroup)}
            style={styles.addFoodEmpty}
          >
            <Text style={styles.addFoodText}>+ Add Food</Text>
          </Pressable>
        )}
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 24,
  },
  addFoodEmpty: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  addFoodText: {
    fontSize: 14,
    color: '#16A34A',
    fontWeight: '500',
  },
});
