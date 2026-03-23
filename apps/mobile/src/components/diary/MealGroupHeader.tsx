/**
 * MealGroupHeader -- meal group header with expand/collapse and add food button.
 */

import React, { useMemo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { type MealGroup, MEAL_GROUP_CONFIG } from '../../services/diary/mealGroups';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface MealGroupHeaderProps {
  mealGroup: MealGroup;
  subtotals: { calories: number; protein: number; carbs: number; fat: number };
  expanded: boolean;
  onToggle: () => void;
  onLongPress: () => void;
  onAddFood: () => void;
}

export function MealGroupHeader({
  mealGroup,
  subtotals,
  expanded,
  onToggle,
  onLongPress,
  onAddFood,
}: MealGroupHeaderProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const config = MEAL_GROUP_CONFIG[mealGroup];
  const iconName = config.icon as keyof typeof Ionicons.glyphMap;

  return (
    <Pressable
      onPress={onToggle}
      onLongPress={onLongPress}
      delayLongPress={500}
      style={styles.container}
    >
      <View style={styles.leftSection}>
        <Ionicons name={iconName} size={20} color={colors.text.tertiary} style={styles.icon} />
        <Text style={styles.label}>{config.label}</Text>
      </View>

      <View style={styles.rightSection}>
        {subtotals.calories > 0 && (
          <Text style={styles.subtotalText}>
            {Math.round(subtotals.calories)} kcal
          </Text>
        )}
        <Pressable
          onPress={(e) => {
            e.stopPropagation?.();
            onAddFood();
          }}
          hitSlop={8}
          style={styles.addButton}
        >
          <Ionicons name="add-circle-outline" size={24} color={colors.accent.green} />
        </Pressable>
        <Ionicons
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size={18}
          color={colors.text.tertiary}
        />
      </View>
    </Pressable>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      backgroundColor: colors.background.elevated,
      minHeight: 44,
      paddingHorizontal: 16,
      paddingVertical: 8,
    },
    leftSection: {
      flexDirection: 'row',
      alignItems: 'center',
      flex: 1,
    },
    icon: {
      marginRight: 8,
    },
    label: {
      fontSize: 16,
      fontWeight: '600',
      color: colors.text.secondary,
    },
    rightSection: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
    },
    subtotalText: {
      fontSize: 14,
      color: colors.text.tertiary,
    },
    addButton: {
      minWidth: 44,
      minHeight: 44,
      alignItems: 'center',
      justifyContent: 'center',
    },
  });
}
