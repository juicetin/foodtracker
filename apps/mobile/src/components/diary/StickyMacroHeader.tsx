/**
 * StickyMacroHeader — pinned daily macro summary with consumed/remaining toggle.
 *
 * Reads diaryDisplayMode from preferences store. Shows total or remaining macros
 * (goals minus consumed, clamped to zero).
 */

import React from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { usePreferencesStore } from '../../store/usePreferencesStore';

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
            <Ionicons name="swap-horizontal-outline" size={20} color="#6B7280" />
          </Pressable>
          <Text style={styles.modeLabel}>{isConsumed ? 'Consumed' : 'Remaining'}</Text>
        </View>
      </View>

      {/* Bottom row: P/C/F pills */}
      <View style={styles.macroRow}>
        <MacroPill value={displayProtein} label="Protein" color="#3B82F6" bgColor="#EFF6FF" />
        <MacroPill value={displayCarbs} label="Carbs" color="#D97706" bgColor="#FFFBEB" />
        <MacroPill value={displayFat} label="Fat" color="#16A34A" bgColor="#F0FDF4" />
      </View>
    </View>
  );
}

function MacroPill({
  value,
  label,
  color,
  bgColor,
}: {
  value: number;
  label: string;
  color: string;
  bgColor: string;
}) {
  return (
    <View style={[styles.macroPill, { backgroundColor: bgColor }]}>
      <Text style={[styles.macroPillNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={styles.macroPillLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#FFFFFF',
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
    color: '#111827',
  },
  calorieLabel: {
    fontSize: 16,
    color: '#6B7280',
    fontWeight: '500',
  },
  toggleBlock: {
    alignItems: 'center',
  },
  toggleButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modeLabel: {
    fontSize: 10,
    color: '#9CA3AF',
    fontWeight: '500',
    marginTop: 4,
  },
  macroRow: {
    flexDirection: 'row',
    gap: 8,
  },
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
    color: '#9CA3AF',
    fontWeight: '500',
    marginTop: 2,
  },
});
