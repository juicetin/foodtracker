/**
 * DiaryScreen — shows today's logged food entries grouped by meal type.
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
  Image,
  Pressable,
  RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { opsqlite } from '../../db/client';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { autoDetectMealType, type MealType } from '../services/detection/types';
import { loadDailyTotals, computeTrendStats, type DayTotals } from '../services/trends/trendsService';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DiaryDish {
  id: string;
  name: string;
  cuisine: string | null;
}

interface DiaryEntry {
  id: string;
  mealType: MealType;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  notes: string | null;
  createdAt: string;
  photoUri: string | null;
  dishes: DiaryDish[];
}

const MEAL_ORDER: MealType[] = ['breakfast', 'lunch', 'snack', 'dinner'];
const MEAL_LABELS: Record<MealType, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  snack: 'Snack',
  dinner: 'Dinner',
};
const MEAL_ICONS: Record<MealType, string> = {
  breakfast: '🌅',
  lunch: '☀️',
  snack: '🍎',
  dinner: '🌙',
};

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

function getTodayDateStr(): string {
  return new Date().toISOString().split('T')[0];
}

function dateToStr(d: Date): string {
  return d.toISOString().split('T')[0];
}

function formatDateLabel(dateStr: string): string {
  const today = getTodayDateStr();
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = dateToStr(yesterday);

  if (dateStr === today) return 'Today';
  if (dateStr === yesterdayStr) return 'Yesterday';

  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
}

function loadEntriesForDate(dateStr: string): DiaryEntry[] {
  const entryRows = opsqlite.execute(
    `SELECT id, meal_type, total_calories, total_protein, total_carbs, total_fat, notes, created_at
     FROM food_entries
     WHERE entry_date = ? AND is_deleted = 0
     ORDER BY created_at DESC`,
    [dateStr],
  ).rows as Array<Record<string, unknown>>;

  return entryRows.map((row) => {
    const entryId = row.id as string;

    const photoRows = opsqlite.execute(
      'SELECT uri FROM photos WHERE entry_id = ? LIMIT 1',
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    const dishRows = opsqlite.execute(
      'SELECT id, name, cuisine FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at',
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    return {
      id: entryId,
      mealType: row.meal_type as MealType,
      totalCalories: (row.total_calories as number) ?? 0,
      totalProtein: (row.total_protein as number) ?? 0,
      totalCarbs: (row.total_carbs as number) ?? 0,
      totalFat: (row.total_fat as number) ?? 0,
      notes: (row.notes as string) ?? null,
      createdAt: row.created_at as string,
      photoUri: photoRows.length > 0 ? (photoRows[0].uri as string) : null,
      dishes: dishRows.map((d) => ({
        id: d.id as string,
        name: d.name as string,
        cuisine: (d.cuisine as string) ?? null,
      })),
    };
  });
}

// ---------------------------------------------------------------------------
// Component
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

export default function DiaryScreen() {
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [selectedDate, setSelectedDate] = useState(getTodayDateStr());
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [trendRange, setTrendRange] = useState<TrendRange>(7);
  const [trendDays, setTrendDays] = useState<DayTotals[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const { addEntry } = useFoodLogStore();
  const { nutritionGoals } = usePreferencesStore();

  const isToday = selectedDate === getTodayDateStr();

  const refresh = useCallback(() => {
    setEntries(loadEntriesForDate(selectedDate));
    setTrendDays(loadDailyTotals(trendRange));
  }, [selectedDate, trendRange]);

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

  // Group entries by meal type
  const grouped = MEAL_ORDER.map((type) => ({
    type,
    entries: entries.filter((e) => e.mealType === type),
  })).filter((g) => g.entries.length > 0);

  // Day totals
  const dayTotals = entries.reduce(
    (acc, e) => ({
      calories: acc.calories + e.totalCalories,
      protein: acc.protein + e.totalProtein,
      carbs: acc.carbs + e.totalCarbs,
      fat: acc.fat + e.totalFat,
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0 },
  );

  return (
    <View style={styles.container}>
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

        {/* Copy previous day */}
        {isToday && (
          <Pressable
            style={styles.copyBtn}
            onPress={async () => {
              const prevDayStr = getPreviousDayStr(selectedDate);
              const yesterday = loadEntriesForDate(prevDayStr);
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

        {/* Day summary card */}
        {entries.length > 0 && (
          <View style={styles.summaryCard}>
            <View style={styles.summaryCalBlock}>
              <Text style={styles.summaryCalNum}>{Math.round(dayTotals.calories)}</Text>
              <Text style={styles.summaryCalLabel}>kcal</Text>
            </View>
            <View style={styles.summaryMacros}>
              <MacroPill value={dayTotals.protein} label="Protein" color="#3B82F6" />
              <MacroPill value={dayTotals.carbs} label="Carbs" color="#D97706" />
              <MacroPill value={dayTotals.fat} label="Fat" color="#16A34A" />
            </View>
          </View>
        )}

        {/* Trends */}
        <TrendsCard
          days={trendDays}
          range={trendRange}
          onRangeChange={(r) => setTrendRange(r)}
          calorieGoal={nutritionGoals.calories}
        />

        {/* Grouped entries */}
        {grouped.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>🍽️</Text>
            <Text style={styles.emptyText}>No meals logged today</Text>
            <Text style={styles.emptySubtext}>Tap the + button to scan your food</Text>
          </View>
        ) : (
          grouped.map((group) => (
            <View key={group.type} style={styles.mealSection}>
              <View style={styles.mealHeader}>
                <Text style={styles.mealIcon}>{MEAL_ICONS[group.type]}</Text>
                <Text style={styles.mealTitle}>{MEAL_LABELS[group.type]}</Text>
                <Text style={styles.mealCalories}>
                  {Math.round(group.entries.reduce((s, e) => s + e.totalCalories, 0))} kcal
                </Text>
              </View>
              {group.entries.map((entry) => (
                <Pressable key={entry.id} onPress={() => nav.navigate('EntryDetail', { entryId: entry.id })}>
                  <EntryCard entry={entry} />
                </Pressable>
              ))}
            </View>
          ))
        )}

        <View style={{ height: 100 }} />
      </ScrollView>
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

function EntryCard({ entry }: { entry: DiaryEntry }) {
  const time = new Date(entry.createdAt).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
  const dishNames =
    entry.dishes.length > 0
      ? entry.dishes.map((d) => d.name).join(', ')
      : entry.notes ?? 'Logged meal';

  return (
    <View style={styles.entryCard}>
      {entry.photoUri && (
        <Image source={{ uri: entry.photoUri }} style={styles.entryPhoto} resizeMode="cover" />
      )}
      <View style={styles.entryInfo}>
        <Text style={styles.entryDishes} numberOfLines={2}>
          {dishNames}
        </Text>
        <Text style={styles.entryTime}>{time}</Text>
      </View>
      <View style={styles.entryNutrition}>
        <Text style={styles.entryCal}>{Math.round(entry.totalCalories)}</Text>
        <Text style={styles.entryCalLabel}>kcal</Text>
      </View>
    </View>
  );
}

function MacroPill({
  value,
  label,
  color,
}: {
  value: number;
  label: string;
  color: string;
}) {
  return (
    <View style={styles.macroPill}>
      <Text style={[styles.macroPillNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={styles.macroPillLabel}>{label}</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  scroll: { flex: 1 },
  scrollContent: { paddingTop: 60, paddingHorizontal: 16 },
  dateNav: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 16,
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

  // Summary
  summaryCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  summaryCalBlock: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 4,
    marginBottom: 12,
  },
  summaryCalNum: { fontSize: 32, fontWeight: '800', color: '#111827' },
  summaryCalLabel: { fontSize: 16, color: '#6B7280', fontWeight: '500' },
  summaryMacros: { flexDirection: 'row', gap: 8 },
  macroPill: {
    flex: 1,
    backgroundColor: '#F9FAFB',
    borderRadius: 10,
    paddingVertical: 8,
    alignItems: 'center',
  },
  macroPillNum: { fontSize: 16, fontWeight: '700' },
  macroPillLabel: {
    fontSize: 11,
    color: '#9CA3AF',
    fontWeight: '500',
    marginTop: 2,
  },

  // Meal sections
  mealSection: { marginBottom: 20 },
  mealHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  mealIcon: { fontSize: 16 },
  mealTitle: { fontSize: 16, fontWeight: '700', color: '#111827', flex: 1 },
  mealCalories: { fontSize: 14, fontWeight: '600', color: '#6B7280' },

  // Entry cards
  entryCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.04,
    shadowRadius: 4,
    elevation: 2,
  },
  entryPhoto: { width: 52, height: 52, borderRadius: 10, marginRight: 12 },
  entryInfo: { flex: 1 },
  entryDishes: { fontSize: 14, fontWeight: '600', color: '#111827', marginBottom: 2 },
  entryTime: { fontSize: 12, color: '#9CA3AF' },
  entryNutrition: { alignItems: 'flex-end', marginLeft: 8 },
  entryCal: { fontSize: 18, fontWeight: '700', color: '#111827' },
  entryCalLabel: { fontSize: 11, color: '#9CA3AF' },

  // Empty
  emptyState: { alignItems: 'center', paddingVertical: 60 },
  emptyIcon: { fontSize: 48, marginBottom: 12 },
  emptyText: { fontSize: 18, fontWeight: '600', color: '#6B7280', marginBottom: 6 },
  emptySubtext: { fontSize: 14, color: '#9CA3AF', textAlign: 'center' },
});
