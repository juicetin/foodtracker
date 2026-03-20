/**
 * DiaryScreen — daily food diary with time-period grouping and sticky macro header.
 *
 * Queries food_entries + scanned_dishes + photos from SQLite.
 * Auto-refreshes when navigating back from DetectionScreen.
 */

import React, { useCallback, useState } from 'react';
import {
  Alert,
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { runOnJS } from 'react-native-reanimated';
import type { RootStackParamList } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { loadDailyTotals, computeTrendStats, type DayTotals } from '../services/trends/trendsService';
import {
  loadEntriesForDate,
  loadWeekEntryPresence,
  computeDayTotals,
  getTodayDateStr,
  dateToStr,
  formatDateLabel,
} from '../services/diary/diaryQueries';
import type { DiaryEntry } from '../services/diary/diaryQueries';
import { TIME_PERIOD_ORDER } from '../services/diary/timePeriods';
import { StickyMacroHeader, WeekOverviewBar, TimePeriodSection, SearchBar } from '../components/diary';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getPreviousDayStr(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  d.setDate(d.getDate() - 1);
  return dateToStr(d);
}

type TrendRange = 7 | 14 | 30 | 0;
const TREND_RANGES: { value: TrendRange; label: string }[] = [
  { value: 7, label: '7D' },
  { value: 14, label: '14D' },
  { value: 30, label: '30D' },
  { value: 0, label: 'All' },
];

const SWIPE_THRESHOLD = 50;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DiaryScreen() {
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [selectedDate, setSelectedDate] = useState(getTodayDateStr());
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [weekPresence, setWeekPresence] = useState<Map<string, number>>(new Map());
  const [trendRange, setTrendRange] = useState<TrendRange>(7);
  const [trendDays, setTrendDays] = useState<DayTotals[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const { addEntry } = useFoodLogStore();
  const { nutritionGoals, timePeriodBoundaries } = usePreferencesStore();

  const isToday = selectedDate === getTodayDateStr();
  const dayTotals = computeDayTotals(entries);

  const refresh = useCallback(() => {
    setEntries(loadEntriesForDate(selectedDate, timePeriodBoundaries));
    setWeekPresence(loadWeekEntryPresence(selectedDate));
    setTrendDays(loadDailyTotals(trendRange));
  }, [selectedDate, trendRange, timePeriodBoundaries]);

  const goToPreviousDay = useCallback(() => {
    setSelectedDate((prev) => {
      const d = new Date(prev + 'T12:00:00');
      d.setDate(d.getDate() - 1);
      return dateToStr(d);
    });
  }, []);

  const goToNextDay = useCallback(() => {
    setSelectedDate((prev) => {
      const d = new Date(prev + 'T12:00:00');
      d.setDate(d.getDate() + 1);
      const next = dateToStr(d);
      // Don't go past today
      return next > getTodayDateStr() ? prev : next;
    });
  }, []);

  const goToToday = useCallback(() => {
    setSelectedDate(getTodayDateStr());
  }, []);

  // Refresh on screen focus or date change
  useFocusEffect(
    useCallback(() => {
      refresh();
    }, [refresh]),
  );

  function onRefresh() {
    setRefreshing(true);
    refresh();
    setRefreshing(false);
  }

  // Swipe gesture for date navigation
  const swipeGesture = Gesture.Pan()
    .activeOffsetX([-20, 20])
    .failOffsetY([-10, 10])
    .onEnd((event) => {
      if (event.translationX > SWIPE_THRESHOLD) {
        runOnJS(goToPreviousDay)();
      } else if (event.translationX < -SWIPE_THRESHOLD) {
        runOnJS(goToNextDay)();
      }
    });

  return (
    <View style={styles.container}>
      {/* Sticky macro header -- OUTSIDE ScrollView */}
      <StickyMacroHeader
        totals={dayTotals}
        goals={nutritionGoals}
      />

      {/* Persistent search bar -- always visible below header */}
      <SearchBar onSearchPress={() => nav.navigate('FoodSearch')} />

      {/* Swipe gesture wraps the scrollable content */}
      <GestureDetector gesture={swipeGesture}>
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#16A34A" />
          }
        >
          {/* Date navigation */}
          <View style={styles.dateNav}>
            <Pressable onPress={goToPreviousDay} style={styles.dateArrow}>
              <Ionicons name="chevron-back" size={24} color="#374151" />
            </Pressable>
            <Pressable onPress={goToToday} style={styles.dateLabelBtn}>
              <Text style={styles.dateTitle}>{formatDateLabel(selectedDate)}</Text>
              {!isToday && (
                <Text style={styles.dateSubtitle}>Tap for today</Text>
              )}
            </Pressable>
            <Pressable onPress={goToNextDay} style={[styles.dateArrow, isToday && { opacity: 0.3 }]} disabled={isToday}>
              <Ionicons name="chevron-forward" size={24} color="#374151" />
            </Pressable>
          </View>

          {/* Week overview bar */}
          <WeekOverviewBar
            selectedDate={selectedDate}
            onSelectDate={setSelectedDate}
            entryPresence={weekPresence}
          />

          {/* Copy previous day */}
          {isToday && (
            <Pressable
              style={styles.copyBtn}
              onPress={async () => {
                const prevDayStr = getPreviousDayStr(selectedDate);
                const yesterday = loadEntriesForDate(prevDayStr, timePeriodBoundaries);
                if (yesterday.length === 0) {
                  Alert.alert('No meals', 'No meals logged yesterday to copy.');
                  return;
                }
                for (const entry of yesterday) {
                  await addEntry({
                    mealType: entry.mealType,
                    totalCalories: entry.totalCalories,
                    totalProtein: entry.totalProtein,
                    totalCarbs: entry.totalCarbs,
                    totalFat: entry.totalFat,
                    notes: `Copied: ${entry.dishes.map((d) => d.name).join(', ') || entry.notes || 'meal'}`,
                  });
                }
                refresh();
                Alert.alert('Copied', `${yesterday.length} meal(s) from yesterday added.`);
              }}
            >
              <Ionicons name="copy-outline" size={14} color="#3B82F6" />
              <Text style={styles.copyBtnText}>Copy Yesterday</Text>
            </Pressable>
          )}

          {/* Time-period sections (morning, afternoon, evening) */}
          {TIME_PERIOD_ORDER.map((period) => {
            const periodEntries = entries.filter((e) => e.timePeriod === period);
            return (
              <TimePeriodSection
                key={period}
                period={period}
                entries={periodEntries}
                onNavigateToDetail={(id) => nav.navigate('EntryDetail', { entryId: id })}
              />
            );
          })}

          {/* Empty state */}
          {entries.length === 0 && (
            <View style={styles.emptyState}>
              <Text style={styles.emptyIcon}>🍽️</Text>
              <Text style={styles.emptyText}>No meals logged today</Text>
              <Text style={styles.emptySubtext}>Tap the + button to scan your food</Text>
            </View>
          )}

          {/* Trends */}
          <TrendsCard
            days={trendDays}
            range={trendRange}
            onRangeChange={(r) => setTrendRange(r)}
            calorieGoal={nutritionGoals.calories}
          />

          <View style={{ height: 100 }} />
        </ScrollView>
      </GestureDetector>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function TrendsCard({
  days,
  range,
  onRangeChange,
  calorieGoal,
}: {
  days: DayTotals[];
  range: TrendRange;
  onRangeChange: (r: TrendRange) => void;
  calorieGoal: number;
}) {
  const stats = computeTrendStats(days, calorieGoal);
  const maxCal = Math.max(...days.map((d) => d.calories), 1);
  const todayStr = getTodayDateStr();
  const showLabels = days.length <= 14; // Only show day labels for 7/14 range

  if (days.length === 0 || stats.daysLogged === 0) return null;

  return (
    <View style={styles.trendsCard}>
      {/* Range toggle */}
      <View style={styles.rangeRow}>
        <Text style={styles.trendsTitle}>Trends</Text>
        <View style={styles.rangePills}>
          {TREND_RANGES.map((r) => (
            <Pressable
              key={r.value}
              style={[styles.rangePill, range === r.value && styles.rangePillActive]}
              onPress={() => onRangeChange(r.value)}
            >
              <Text style={[styles.rangePillText, range === r.value && styles.rangePillTextActive]}>
                {r.label}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>

      {/* Calorie bars */}
      <View style={styles.barsContainer}>
        {days.map((day) => {
          const pct = maxCal > 0 ? Math.min(1, day.calories / maxCal) : 0;
          const isDayToday = day.date === todayStr;
          const overGoal = day.calories > calorieGoal;
          return (
            <View key={day.date} style={[styles.barCol, { flex: days.length <= 14 ? 1 : undefined, width: days.length > 14 ? Math.max(4, 300 / days.length) : undefined }]}>
              <View style={styles.barTrack}>
                <View
                  style={[
                    styles.barFill,
                    {
                      height: `${Math.max(pct * 100, day.calories > 0 ? 4 : 0)}%`,
                      backgroundColor: overGoal ? '#EF4444' : isDayToday ? '#16A34A' : '#93C5FD',
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
        <View style={[styles.macroAvgPill, { backgroundColor: '#EFF6FF' }]}>
          <Text style={[styles.macroAvgNum, { color: '#3B82F6' }]}>{Math.round(stats.avgProtein)}g</Text>
          <Text style={styles.macroAvgLabel}>Avg P</Text>
        </View>
        <View style={[styles.macroAvgPill, { backgroundColor: '#FFFBEB' }]}>
          <Text style={[styles.macroAvgNum, { color: '#D97706' }]}>{Math.round(stats.avgCarbs)}g</Text>
          <Text style={styles.macroAvgLabel}>Avg C</Text>
        </View>
        <View style={[styles.macroAvgPill, { backgroundColor: '#F0FDF4' }]}>
          <Text style={[styles.macroAvgNum, { color: '#16A34A' }]}>{Math.round(stats.avgFat)}g</Text>
          <Text style={styles.macroAvgLabel}>Avg F</Text>
        </View>
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  scroll: { flex: 1 },
  scrollContent: { paddingTop: 16, paddingHorizontal: 16 },
  dateNav: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 12,
  },
  dateArrow: { padding: 8 },
  dateLabelBtn: { alignItems: 'center', flex: 1 },
  dateTitle: { fontSize: 20, fontWeight: '800', color: '#111827' },
  dateSubtitle: { fontSize: 11, color: '#16A34A', fontWeight: '500', marginTop: 2 },
  copyBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    backgroundColor: '#EFF6FF', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8,
    marginBottom: 12, alignSelf: 'center',
  },
  copyBtnText: { fontSize: 13, fontWeight: '600', color: '#3B82F6' },

  // Trends card
  trendsCard: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  rangeRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  trendsTitle: { fontSize: 16, fontWeight: '700', color: '#111827' },
  rangePills: { flexDirection: 'row', gap: 4 },
  rangePill: {
    paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8,
    backgroundColor: '#F3F4F6',
  },
  rangePillActive: { backgroundColor: '#16A34A' },
  rangePillText: { fontSize: 12, fontWeight: '600', color: '#6B7280' },
  rangePillTextActive: { color: '#FFF' },
  barsContainer: { flexDirection: 'row', gap: 2, height: 100, marginBottom: 12 },
  barCol: { alignItems: 'center' },
  barTrack: {
    flex: 1, width: '100%', backgroundColor: '#F3F4F6', borderRadius: 3,
    justifyContent: 'flex-end', overflow: 'hidden', minWidth: 4,
  },
  barFill: { width: '100%', borderRadius: 3 },
  barDayLabel: { fontSize: 9, color: '#9CA3AF', fontWeight: '500', marginTop: 3 },
  barDayLabelToday: { color: '#16A34A', fontWeight: '700' },
  statsRow: {
    flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10,
    borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: '#F3F4F6',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
    marginBottom: 10,
  },
  statItem: { alignItems: 'center', flex: 1 },
  statValue: { fontSize: 16, fontWeight: '800', color: '#111827' },
  statLabel: { fontSize: 10, color: '#9CA3AF', fontWeight: '500', marginTop: 2 },
  macroAvgRow: { flexDirection: 'row', gap: 8 },
  macroAvgPill: {
    flex: 1, borderRadius: 10, paddingVertical: 8, alignItems: 'center',
  },
  macroAvgNum: { fontSize: 15, fontWeight: '700' },
  macroAvgLabel: { fontSize: 10, color: '#9CA3AF', fontWeight: '500', marginTop: 2 },

  // Empty
  emptyState: { alignItems: 'center', paddingVertical: 60 },
  emptyIcon: { fontSize: 48, marginBottom: 12 },
  emptyText: { fontSize: 18, fontWeight: '600', color: '#6B7280', marginBottom: 6 },
  emptySubtext: { fontSize: 14, color: '#9CA3AF', textAlign: 'center' },
});
