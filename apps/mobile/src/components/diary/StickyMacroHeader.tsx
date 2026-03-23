/**
 * StickyMacroHeader -- pinned daily macro summary with consumed/remaining toggle.
 *
 * Reads diaryDisplayMode from preferences store. Shows total or remaining macros
 * (goals minus consumed, clamped to zero).
 */

import React, { useMemo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { usePreferencesStore } from '../../store/usePreferencesStore';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface MacroTotals {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

interface StickyMacroHeaderProps {
  totals: MacroTotals;
  goals: MacroTotals;
}

export function StickyMacroHeader({ totals, goals }: StickyMacroHeaderProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const diaryDisplayMode = usePreferencesStore((s) => s.diaryDisplayMode);
  const setDiaryDisplayMode = usePreferencesStore((s) => s.setDiaryDisplayMode);

  const isConsumed = diaryDisplayMode === 'consumed';

  const displayCalories = isConsumed
    ? totals.calories
    : Math.max(0, goals.calories - totals.calories);
  const displayProtein = isConsumed
    ? totals.protein
    : Math.max(0, goals.protein - totals.protein);
  const displayCarbs = isConsumed
    ? totals.carbs
    : Math.max(0, goals.carbs - totals.carbs);
  const displayFat = isConsumed
    ? totals.fat
    : Math.max(0, goals.fat - totals.fat);

  const toggleMode = () => {
    setDiaryDisplayMode(isConsumed ? 'remaining' : 'consumed');
  };

  return (
    <View style={styles.container}>
      {/* Top row: calorie number + toggle */}
      <View style={styles.topRow}>
        <View style={styles.calorieBlock}>
          <Text style={styles.calorieNumber}>{Math.round(displayCalories)}</Text>
          <Text style={styles.calorieLabel}>kcal</Text>
        </View>
        <View style={styles.toggleBlock}>
          <Pressable onPress={toggleMode} style={styles.toggleButton} testID="macro-toggle">
            <Ionicons name="swap-horizontal-outline" size={20} color={colors.text.tertiary} />
          </Pressable>
          <Text style={styles.modeLabel}>{isConsumed ? 'Consumed' : 'Remaining'}</Text>
        </View>
      </View>

      {/* Bottom row: P/C/F pills */}
      <View style={styles.macroRow}>
        <MacroPill value={displayProtein} label="Protein" color={colors.accent.blue} bgColor={colors.accentTint.blue} colors={colors} />
        <MacroPill value={displayCarbs} label="Carbs" color={colors.accent.amber} bgColor={colors.accentTint.amber} colors={colors} />
        <MacroPill value={displayFat} label="Fat" color={colors.accent.green} bgColor={colors.accentTint.green} colors={colors} />
      </View>
    </View>
  );
}

function MacroPill({
  value,
  label,
  color,
  bgColor,
  colors,
}: {
  value: number;
  label: string;
  color: string;
  bgColor: string;
  colors: ThemeColors;
}) {
  return (
    <View style={[pillStyles.macroPill, { backgroundColor: bgColor }]}>
      <Text style={[pillStyles.macroPillNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={[pillStyles.macroPillLabel, { color: colors.text.tertiary }]}>{label}</Text>
    </View>
  );
}

const pillStyles = StyleSheet.create({
  macroPill: {
    flex: 1,
    borderRadius: 10,
    paddingVertical: 8,
    alignItems: 'center',
  },
  macroPillNum: {
    fontSize: 16,
    fontWeight: '700',
  },
  macroPillLabel: {
    fontSize: 11,
    fontWeight: '500',
    marginTop: 2,
  },
});

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      backgroundColor: colors.background.elevated,
      borderRadius: 16,
      padding: 16,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.06,
      shadowRadius: 8,
      elevation: 3,
    },
    topRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: 12,
    },
    calorieBlock: {
      flexDirection: 'row',
      alignItems: 'baseline',
      gap: 4,
    },
    calorieNumber: {
      fontSize: 32,
      fontWeight: '800',
      color: colors.text.primary,
    },
    calorieLabel: {
      fontSize: 16,
      color: colors.text.tertiary,
      fontWeight: '500',
    },
    toggleBlock: {
      alignItems: 'center',
    },
    toggleButton: {
      width: 36,
      height: 36,
      borderRadius: 18,
      backgroundColor: colors.background.surface,
      alignItems: 'center',
      justifyContent: 'center',
    },
    modeLabel: {
      fontSize: 10,
      color: colors.text.tertiary,
      fontWeight: '500',
      marginTop: 4,
    },
    macroRow: {
      flexDirection: 'row',
      gap: 8,
    },
  });
}
