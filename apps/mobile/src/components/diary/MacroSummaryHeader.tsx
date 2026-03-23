/**
 * MacroSummaryHeader -- remaining calories and P/C/F progress bars.
 *
 * Shows remaining calories prominently with three horizontal progress bars
 * for protein, carbs, and fat. Red text when over goal.
 */

import React, { useMemo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface MacroTotals {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

interface MacroSummaryHeaderProps {
  totals: MacroTotals;
  goals: MacroTotals;
}

export function MacroSummaryHeader({ totals, goals }: MacroSummaryHeaderProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const remainingCal = goals.calories - totals.calories;
  const isOverGoal = remainingCal < 0;

  return (
    <View style={styles.container}>
      {/* Remaining calories */}
      <Text style={[styles.calorieNumber, isOverGoal && { color: colors.accent.red }]}>
        {Math.round(remainingCal)}
      </Text>
      <Text style={styles.remainingLabel}>Remaining</Text>

      {/* P/C/F progress bars */}
      <View style={styles.barsContainer}>
        <MacroBar
          label="Protein"
          current={totals.protein}
          goal={goals.protein}
          fillColor={colors.accent.blue}
          trackColor={colors.accentTint.blue}
          colors={colors}
        />
        <MacroBar
          label="Carbs"
          current={totals.carbs}
          goal={goals.carbs}
          fillColor={colors.accent.amber}
          trackColor={colors.accentTint.amber}
          colors={colors}
        />
        <MacroBar
          label="Fat"
          current={totals.fat}
          goal={goals.fat}
          fillColor={colors.accent.green}
          trackColor={colors.accentTint.green}
          colors={colors}
        />
      </View>
    </View>
  );
}

function MacroBar({
  label,
  current,
  goal,
  fillColor,
  trackColor,
  colors,
}: {
  label: string;
  current: number;
  goal: number;
  fillColor: string;
  trackColor: string;
  colors: ThemeColors;
}) {
  const pct = goal > 0 ? Math.min(1, current / goal) : 0;

  return (
    <View style={barStyles.barRow}>
      <View style={barStyles.barLabelRow}>
        <Text style={[barStyles.barLabel, { color: colors.text.secondary }]}>{label}</Text>
        <Text style={[barStyles.barValue, { color: colors.text.tertiary }]}>
          {Math.round(current)}g / {Math.round(goal)}g
        </Text>
      </View>
      <View style={[barStyles.barTrack, { backgroundColor: trackColor }]}>
        <View
          style={[
            barStyles.barFill,
            {
              backgroundColor: fillColor,
              width: `${pct * 100}%`,
            },
          ]}
        />
      </View>
    </View>
  );
}

const barStyles = StyleSheet.create({
  barRow: {
    gap: 4,
  },
  barLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  barLabel: {
    fontSize: 14,
    fontWeight: '500',
  },
  barValue: {
    fontSize: 14,
  },
  barTrack: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  barFill: {
    height: 8,
    borderRadius: 4,
  },
});

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      backgroundColor: colors.background.elevated,
      padding: 16,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border.subtle,
    },
    calorieNumber: {
      fontSize: 32,
      fontWeight: '600',
      color: colors.text.primary,
      textAlign: 'center',
    },
    remainingLabel: {
      fontSize: 14,
      color: colors.text.tertiary,
      textAlign: 'center',
      marginBottom: 16,
    },
    barsContainer: {
      gap: 12,
    },
  });
}
