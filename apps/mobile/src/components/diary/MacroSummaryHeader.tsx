/**
 * MacroSummaryHeader -- remaining calories and P/C/F progress bars.
 *
 * Shows remaining calories prominently with three horizontal progress bars
 * for protein, carbs, and fat. Red text when over goal.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

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
  const remainingCal = goals.calories - totals.calories;
  const isOverGoal = remainingCal < 0;

  return (
    <View style={styles.container}>
      {/* Remaining calories */}
      <Text style={[styles.calorieNumber, isOverGoal && styles.calorieOver]}>
        {Math.round(remainingCal)}
      </Text>
      <Text style={styles.remainingLabel}>Remaining</Text>

      {/* P/C/F progress bars */}
      <View style={styles.barsContainer}>
        <MacroBar
          label="Protein"
          current={totals.protein}
          goal={goals.protein}
          fillColor="#3B82F6"
          trackColor="#EFF6FF"
        />
        <MacroBar
          label="Carbs"
          current={totals.carbs}
          goal={goals.carbs}
          fillColor="#D97706"
          trackColor="#FFFBEB"
        />
        <MacroBar
          label="Fat"
          current={totals.fat}
          goal={goals.fat}
          fillColor="#059669"
          trackColor="#ECFDF5"
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
}: {
  label: string;
  current: number;
  goal: number;
  fillColor: string;
  trackColor: string;
}) {
  const pct = goal > 0 ? Math.min(1, current / goal) : 0;

  return (
    <View style={styles.barRow}>
      <View style={styles.barLabelRow}>
        <Text style={styles.barLabel}>{label}</Text>
        <Text style={styles.barValue}>
          {Math.round(current)}g / {Math.round(goal)}g
        </Text>
      </View>
      <View style={[styles.barTrack, { backgroundColor: trackColor }]}>
        <View
          style={[
            styles.barFill,
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

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E5E7EB',
  },
  calorieNumber: {
    fontSize: 32,
    fontWeight: '600',
    color: '#111827',
    textAlign: 'center',
  },
  calorieOver: {
    color: '#EF4444',
  },
  remainingLabel: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    marginBottom: 16,
  },
  barsContainer: {
    gap: 12,
  },
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
    color: '#374151',
  },
  barValue: {
    fontSize: 14,
    color: '#6B7280',
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
