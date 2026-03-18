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
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { opsqlite } from '../../db/client';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { autoDetectMealType, type MealType } from '../services/detection/types';

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

function loadTodayEntries(): DiaryEntry[] {
  const todayStr = getTodayDateStr();

  const entryRows = opsqlite.execute(
    `SELECT id, meal_type, total_calories, total_protein, total_carbs, total_fat, notes, created_at
     FROM food_entries
     WHERE entry_date = ? AND is_deleted = 0
     ORDER BY created_at DESC`,
    [todayStr],
  ).rows as Array<Record<string, unknown>>;

  return entryRows.map((row) => {
    const entryId = row.id as string;

    // Load photo
    const photoRows = opsqlite.execute(
      'SELECT uri FROM photos WHERE entry_id = ? LIMIT 1',
      [entryId],
    ).rows as Array<Record<string, unknown>>;

    // Load dishes
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

function getYesterdayDateStr(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().split('T')[0];
}

function loadYesterdayEntries(): DiaryEntry[] {
  const yesterdayStr = getYesterdayDateStr();
  const entryRows = opsqlite.execute(
    `SELECT id, meal_type, total_calories, total_protein, total_carbs, total_fat, notes, created_at
     FROM food_entries WHERE entry_date = ? AND is_deleted = 0 ORDER BY created_at`,
    [yesterdayStr],
  ).rows as Array<Record<string, unknown>>;

  return entryRows.map((row) => {
    const entryId = row.id as string;
    const photoRows = opsqlite.execute('SELECT uri FROM photos WHERE entry_id = ? LIMIT 1', [entryId]).rows as Array<Record<string, unknown>>;
    const dishRows = opsqlite.execute('SELECT id, name, cuisine FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at', [entryId]).rows as Array<Record<string, unknown>>;
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
        id: d.id as string, name: d.name as string, cuisine: (d.cuisine as string) ?? null,
      })),
    };
  });
}

interface DayTotal {
  date: string;
  dayLabel: string;
  calories: number;
}

function loadWeeklyTotals(): DayTotal[] {
  const days: DayTotal[] = [];
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const dateStr = d.toISOString().split('T')[0];
    const result = opsqlite.execute(
      'SELECT SUM(total_calories) AS cal FROM food_entries WHERE entry_date = ? AND is_deleted = 0',
      [dateStr],
    ).rows as Array<Record<string, unknown>>;
    days.push({
      date: dateStr,
      dayLabel: dayNames[d.getDay()],
      calories: (result[0]?.cal as number) ?? 0,
    });
  }
  return days;
}

export default function DiaryScreen() {
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [weeklyTotals, setWeeklyTotals] = useState<DayTotal[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const { addEntry } = useFoodLogStore();
  const { nutritionGoals } = usePreferencesStore();

  const refresh = useCallback(() => {
    setEntries(loadTodayEntries());
    setWeeklyTotals(loadWeeklyTotals());
  }, []);

  // Refresh on screen focus (e.g. after logging a meal)
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
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.title}>Food Diary</Text>
            <Text style={styles.dateLabel}>Today</Text>
          </View>
          <Pressable
            style={styles.copyBtn}
            onPress={async () => {
              const yesterday = loadYesterdayEntries();
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
            <Text style={styles.copyBtnText}>📋 Copy Yesterday</Text>
          </Pressable>
        </View>

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

        {/* Weekly trends */}
        {weeklyTotals.some((d) => d.calories > 0) && (
          <View style={styles.weeklyCard}>
            <Text style={styles.weeklyTitle}>This Week</Text>
            <View style={styles.weeklyBars}>
              {weeklyTotals.map((day) => {
                const pct = nutritionGoals.calories > 0
                  ? Math.min(1, day.calories / nutritionGoals.calories)
                  : 0;
                const isToday = day.date === getTodayDateStr();
                return (
                  <View key={day.date} style={styles.barCol}>
                    <Text style={styles.barCalLabel}>
                      {day.calories > 0 ? Math.round(day.calories) : ''}
                    </Text>
                    <View style={styles.barTrack}>
                      <View
                        style={[
                          styles.barFill,
                          {
                            height: `${Math.max(pct * 100, day.calories > 0 ? 4 : 0)}%`,
                            backgroundColor: pct >= 1 ? '#EF4444' : isToday ? '#16A34A' : '#93C5FD',
                          },
                        ]}
                      />
                    </View>
                    <Text style={[styles.barDayLabel, isToday && styles.barDayLabelToday]}>
                      {day.dayLabel}
                    </Text>
                  </View>
                );
              })}
            </View>
          </View>
        )}

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
  headerRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16,
  },
  title: { fontSize: 28, fontWeight: '800', color: '#111827', marginBottom: 4 },
  dateLabel: { fontSize: 15, color: '#6B7280', fontWeight: '500' },
  copyBtn: {
    backgroundColor: '#EFF6FF', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8, marginTop: 4,
  },
  copyBtnText: { fontSize: 13, fontWeight: '600', color: '#3B82F6' },

  // Weekly trends
  weeklyCard: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  weeklyTitle: { fontSize: 16, fontWeight: '700', color: '#111827', marginBottom: 12 },
  weeklyBars: { flexDirection: 'row', gap: 4, height: 120 },
  barCol: { flex: 1, alignItems: 'center' },
  barCalLabel: { fontSize: 9, color: '#9CA3AF', marginBottom: 4, height: 12 },
  barTrack: {
    flex: 1, width: '100%', backgroundColor: '#F3F4F6', borderRadius: 4,
    justifyContent: 'flex-end', overflow: 'hidden',
  },
  barFill: { width: '100%', borderRadius: 4 },
  barDayLabel: { fontSize: 11, color: '#9CA3AF', fontWeight: '500', marginTop: 4 },
  barDayLabelToday: { color: '#16A34A', fontWeight: '700' },

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
