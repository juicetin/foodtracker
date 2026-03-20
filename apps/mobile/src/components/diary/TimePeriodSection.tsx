/**
 * TimePeriodSection — groups diary entries under a time period header.
 *
 * Shows period icon + label + calorie/macro subtotals. Renders ExpandableEntryCard
 * for each entry. Returns null if no entries.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { type TimePeriod, getTimePeriodIcon, getTimePeriodLabel } from '../../services/diary/timePeriods';
import type { DiaryEntry } from '../../services/diary/diaryQueries';
import { ExpandableEntryCard } from './ExpandableEntryCard';

interface TimePeriodSectionProps {
  period: TimePeriod;
  entries: DiaryEntry[];
  onNavigateToDetail: (entryId: string) => void;
}

export function TimePeriodSection({ period, entries, onNavigateToDetail }: TimePeriodSectionProps) {
  if (entries.length === 0) return null;

  const subtotals = entries.reduce(
    (acc, e) => ({
      calories: acc.calories + e.totalCalories,
      protein: acc.protein + e.totalProtein,
      carbs: acc.carbs + e.totalCarbs,
      fat: acc.fat + e.totalFat,
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0 },
  );

  const iconName = getTimePeriodIcon(period) as keyof typeof Ionicons.glyphMap;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons name={iconName} size={18} color="#6B7280" style={styles.icon} />
        <Text style={styles.title}>{getTimePeriodLabel(period)}</Text>
        <View style={styles.subtotalBlock}>
          <Text style={styles.subtotalCalories}>{Math.round(subtotals.calories)} kcal</Text>
          <Text style={styles.subtotalMacros}>
            P{Math.round(subtotals.protein)} C{Math.round(subtotals.carbs)} F{Math.round(subtotals.fat)}
          </Text>
        </View>
      </View>
      {entries.map((entry) => (
        <ExpandableEntryCard
          key={entry.id}
          entry={entry}
          onNavigateToDetail={onNavigateToDetail}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  icon: {
    marginRight: 6,
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
    flex: 1,
  },
  subtotalBlock: {
    alignItems: 'flex-end',
  },
  subtotalCalories: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6B7280',
  },
  subtotalMacros: {
    fontSize: 10,
    color: '#9CA3AF',
    fontWeight: '500',
    marginTop: 1,
  },
});
