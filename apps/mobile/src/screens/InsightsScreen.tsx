/**
 * InsightsScreen — trends tab showing calorie bar chart, macro averages, and streak.
 *
 * Extracted from DiaryScreen's TrendsCard section. Reloads data on tab focus.
 */

import React, {useCallback, useState, useMemo} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { loadDailyTotals, computeTrendStats, type DayTotals } from '../services/trends/trendsService';
import { getTodayDateStr } from '../services/diary/diaryQueries';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TrendRange = 7 | 14 | 30 | 0;

const TREND_RANGES: { value: TrendRange; label: string }[] = [
  { value: 7, label: '7D' },
  { value: 14, label: '14D' },
  { value: 30, label: '30D' },
  { value: 0, label: 'All' },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function InsightsScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const [trendRange, setTrendRange] = useState<TrendRange>(7);
  const [trendDays, setTrendDays] = useState<DayTotals[]>([]);
  const { nutritionGoals } = usePreferencesStore();

  useFocusEffect(
    useCallback(() => {
      setTrendDays(loadDailyTotals(trendRange));
    }, [trendRange]),
  );

  const stats = computeTrendStats(trendDays, nutritionGoals.calories);
  const todayStr = getTodayDateStr();
  const maxCal = Math.max(...trendDays.map((d) => d.calories), 1);
  const showLabels = trendDays.length <= 14;

  // Empty state
  if (trendDays.length === 0 || stats.daysLogged === 0) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.screenTitle}>Insights</Text>
        </View>
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>Start tracking to see insights</Text>
          <Text style={styles.emptyBody}>
            Log a few meals and your trends will appear here.
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.screenTitle}>Insights</Text>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Range toggle */}
        <View style={styles.rangeRow}>
          <Text style={styles.trendsTitle}>Trends</Text>
          <View style={styles.rangePills}>
            {TREND_RANGES.map((r) => (
              <Pressable
                key={r.value}
                style={[styles.rangePill, trendRange === r.value && styles.rangePillActive]}
                onPress={() => setTrendRange(r.value)}
              >
                <Text style={[styles.rangePillText, trendRange === r.value && styles.rangePillTextActive]}>
                  {r.label}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* Calorie bars */}
        <View style={styles.barsContainer}>
          {trendDays.map((day) => {
            const pct = maxCal > 0 ? Math.min(1, day.calories / maxCal) : 0;
            const isDayToday = day.date === todayStr;
            const overGoal = day.calories > nutritionGoals.calories;
            return (
              <View
                key={day.date}
                style={[
                  styles.barCol,
                  {
                    flex: trendDays.length <= 14 ? 1 : undefined,
                    width: trendDays.length > 14 ? Math.max(4, 300 / trendDays.length) : undefined,
                  },
                ]}
              >
                <View style={styles.barTrack}>
                  <View
                    style={[
                      styles.barFill,
                      {
                        height: `${Math.max(pct * 100, day.calories > 0 ? 4 : 0)}%`,
                        backgroundColor: overGoal ? colors.accent.red : isDayToday ? colors.accent.green : '#93C5FD',
                      },
                    ]}
                  />
                </View>
                {showLabels && (
                  <Text style={[styles.barDayLabel, isDayToday && styles.barDayLabelToday]}>
                    {day.dayLabel}
                  </Text>
                )}
              </View>
            );
          })}
        </View>

        {/* Stats row */}
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.avgCalories}</Text>
            <Text style={styles.statLabel}>Avg kcal</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.goalAdherencePct}%</Text>
            <Text style={styles.statLabel}>On target</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.daysLogged}/{stats.totalDays}</Text>
            <Text style={styles.statLabel}>Days logged</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.currentStreak}</Text>
            <Text style={styles.statLabel}>Streak</Text>
          </View>
        </View>

        {/* Macro averages */}
        <View style={styles.macroAvgRow}>
          <View style={[styles.macroAvgPill, { backgroundColor: colors.accentTint.blue }]}>
            <Text style={[styles.macroAvgNum, { color: colors.accent.blue }]}>{Math.round(stats.avgProtein)}g</Text>
            <Text style={styles.macroAvgLabel}>Avg P</Text>
          </View>
          <View style={[styles.macroAvgPill, { backgroundColor: colors.accentTint.amber }]}>
            <Text style={[styles.macroAvgNum, { color: '#D97706' }]}>{Math.round(stats.avgCarbs)}g</Text>
            <Text style={styles.macroAvgLabel}>Avg C</Text>
          </View>
          <View style={[styles.macroAvgPill, { backgroundColor: colors.accentTint.green }]}>
            <Text style={[styles.macroAvgNum, { color: colors.accent.green }]}>{Math.round(stats.avgFat)}g</Text>
            <Text style={styles.macroAvgLabel}>Avg F</Text>
          </View>
        </View>

        <View style={{ height: 100 }} />
      </ScrollView>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background.primary },
  header: {
    paddingTop: 56,
    paddingHorizontal: 16,
    paddingBottom: 12,
    backgroundColor: colors.background.elevated,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border.subtle,
  },
  screenTitle: { fontSize: 28, fontWeight: '800', color: colors.text.primary },
  scroll: { flex: 1 },
  scrollContent: { padding: 16 },

  // Empty state
  emptyState: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingTop: 120 },
  emptyTitle: { fontSize: 20, fontWeight: '700', color: colors.text.secondary, marginBottom: 8 },
  emptyBody: { fontSize: 15, color: colors.text.tertiary, textAlign: 'center', maxWidth: 260 },

  // Range toggle
  rangeRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  trendsTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary },
  rangePills: { flexDirection: 'row', gap: 4 },
  rangePill: {
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8,
    backgroundColor: colors.background.surface,
  },
  rangePillActive: { backgroundColor: colors.accent.green },
  rangePillText: { fontSize: 13, fontWeight: '600', color: colors.text.tertiary },
  rangePillTextActive: { color: colors.text.inverse },

  // Calorie bars
  barsContainer: {
    flexDirection: 'row', gap: 2, height: 140,
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 12,
    marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  barCol: { alignItems: 'center' },
  barTrack: {
    flex: 1, width: '100%', backgroundColor: colors.background.surface, borderRadius: 3,
    justifyContent: 'flex-end', overflow: 'hidden', minWidth: 4,
  },
  barFill: { width: '100%', borderRadius: 3 },
  barDayLabel: { fontSize: 9, color: colors.text.tertiary, fontWeight: '500', marginTop: 3 },
  barDayLabelToday: { color: colors.accent.green, fontWeight: '700' },

  // Stats
  statsRow: {
    flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 14,
    backgroundColor: colors.background.elevated, borderRadius: 16, paddingHorizontal: 12,
    marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  statItem: { alignItems: 'center', flex: 1 },
  statValue: { fontSize: 18, fontWeight: '800', color: colors.text.primary },
  statLabel: { fontSize: 10, color: colors.text.tertiary, fontWeight: '500', marginTop: 2 },

  // Macro averages
  macroAvgRow: {
    flexDirection: 'row', gap: 8,
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  macroAvgPill: {
    flex: 1, borderRadius: 10, paddingVertical: 10, alignItems: 'center',
  },
  macroAvgNum: { fontSize: 16, fontWeight: '700' },
  macroAvgLabel: { fontSize: 10, color: colors.text.tertiary, fontWeight: '500', marginTop: 2 },
});
}
