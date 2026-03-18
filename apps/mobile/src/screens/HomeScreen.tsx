/**
 * HomeScreen — calorie/macro dashboard with progress, recent meals, quick actions.
 *
 * Shows: calorie remaining ring, macro progress bars, recent meals for quick re-log,
 * and action buttons (scan, quick-add).
 */

import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Modal,
  TextInput,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { opsqlite } from '../../db/client';
import { autoDetectMealType } from '../services/detection/types';
import { loadFavourites, incrementFavouriteUsage, type FavouriteMeal } from '../services/favourites';

// ---------------------------------------------------------------------------
// Recent meals loader
// ---------------------------------------------------------------------------

interface RecentMeal {
  dishName: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  mealType: string;
  entryDate: string;
  entryId: string;
}

function loadRecentMeals(limit: number = 5): RecentMeal[] {
  try {
    const rows = opsqlite.execute(
      `SELECT sd.name AS dish_name, fe.total_calories, fe.total_protein, fe.total_carbs, fe.total_fat,
              fe.meal_type, fe.entry_date, fe.id AS entry_id
       FROM scanned_dishes sd
       JOIN food_entries fe ON fe.id = sd.entry_id
       WHERE fe.is_deleted = 0
       ORDER BY fe.created_at DESC
       LIMIT ?`,
      [limit],
    ).rows as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      dishName: r.dish_name as string,
      calories: (r.total_calories as number) ?? 0,
      protein: (r.total_protein as number) ?? 0,
      carbs: (r.total_carbs as number) ?? 0,
      fat: (r.total_fat as number) ?? 0,
      mealType: r.meal_type as string,
      entryDate: r.entry_date as string,
      entryId: r.entry_id as string,
    }));
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Streak counter
// ---------------------------------------------------------------------------

function calculateStreak(): number {
  try {
    const rows = opsqlite.execute(
      `SELECT DISTINCT entry_date FROM food_entries WHERE is_deleted = 0 ORDER BY entry_date DESC LIMIT 60`,
    ).rows as Array<Record<string, unknown>>;

    if (rows.length === 0) return 0;

    const dates = rows.map((r) => r.entry_date as string);
    const today = new Date().toISOString().split('T')[0];

    // If today hasn't been logged yet, start from yesterday
    let checkDate = today;
    if (dates[0] !== today) {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      checkDate = yesterday.toISOString().split('T')[0];
      if (dates[0] !== checkDate) return 0; // No recent logging
    }

    let streak = 0;
    const dateSet = new Set(dates);
    const d = new Date(checkDate);
    while (dateSet.has(d.toISOString().split('T')[0])) {
      streak++;
      d.setDate(d.getDate() - 1);
    }
    return streak;
  } catch {
    return 0;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function HomeScreen() {
  const navigation = useNavigation<any>();
  const { nutritionGoals } = usePreferencesStore();
  const { getTodayTotals, addEntry, loadTodayEntries } = useFoodLogStore();

  const [recentMeals, setRecentMeals] = useState<RecentMeal[]>([]);
  const [favourites, setFavourites] = useState<FavouriteMeal[]>([]);
  const [quickAddVisible, setQuickAddVisible] = useState(false);
  const [quickAddCal, setQuickAddCal] = useState('');
  const [streak, setStreak] = useState(0);

  const totals = getTodayTotals();

  useFocusEffect(
    useCallback(() => {
      loadTodayEntries();
      setRecentMeals(loadRecentMeals());
      setFavourites(loadFavourites(5));
      setStreak(calculateStreak());
    }, [loadTodayEntries]),
  );

  async function handleLogFavourite(fav: FavouriteMeal) {
    await addEntry({
      mealType: autoDetectMealType(),
      totalCalories: fav.totalCalories,
      totalProtein: fav.totalProtein,
      totalCarbs: fav.totalCarbs,
      totalFat: fav.totalFat,
      notes: `Favourite: ${fav.name}`,
    });
    incrementFavouriteUsage(fav.id);
    await loadTodayEntries();
    setFavourites(loadFavourites(5));
    setStreak(calculateStreak());
  }

  async function handleRelogMeal(meal: RecentMeal) {
    await addEntry({
      mealType: autoDetectMealType(),
      totalCalories: meal.calories,
      totalProtein: meal.protein,
      totalCarbs: meal.carbs,
      totalFat: meal.fat,
      notes: `Re-logged: ${meal.dishName}`,
    });
    await loadTodayEntries();
    setStreak(calculateStreak());
  }

  const calRemaining = Math.max(0, nutritionGoals.calories - totals.calories);
  const calPct = Math.min(1, totals.calories / nutritionGoals.calories);
  const proteinPct = Math.min(1, totals.protein / nutritionGoals.protein);
  const carbsPct = Math.min(1, totals.carbs / nutritionGoals.carbs);
  const fatPct = Math.min(1, totals.fat / nutritionGoals.fat);

  async function handleQuickAdd() {
    const cal = parseInt(quickAddCal, 10);
    if (isNaN(cal) || cal <= 0) return;
    await addEntry({
      mealType: autoDetectMealType(),
      totalCalories: cal,
      totalProtein: 0,
      totalCarbs: 0,
      totalFat: 0,
      notes: `Quick add: ${cal} kcal`,
    });
    setQuickAddVisible(false);
    setQuickAddCal('');
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.headerRow}>
        <View>
          <Text style={styles.greeting}>Tastimate</Text>
          <Text style={styles.dateLabel}>
            {new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' })}
          </Text>
        </View>
        {streak > 0 && (
          <View style={styles.streakBadge}>
            <Text style={styles.streakNum}>{streak}</Text>
            <Text style={styles.streakLabel}>day{streak !== 1 ? 's' : ''}</Text>
          </View>
        )}
      </View>

      {/* Calorie ring card */}
      <View style={styles.calorieCard}>
        <View style={styles.calorieRing}>
          {/* Simplified ring using a border-based approach */}
          <View style={[styles.ringOuter, { borderColor: calPct >= 1 ? '#EF4444' : '#16A34A' }]}>
            <Text style={styles.ringCalNum}>{Math.round(calRemaining)}</Text>
            <Text style={styles.ringCalLabel}>remaining</Text>
          </View>
        </View>
        <View style={styles.calorieDetails}>
          <CalorieRow label="Goal" value={nutritionGoals.calories} color="#6B7280" />
          <CalorieRow label="Eaten" value={Math.round(totals.calories)} color="#111827" />
          <CalorieRow label="Remaining" value={Math.round(calRemaining)} color="#16A34A" />
        </View>
      </View>

      {/* Macro progress bars */}
      <View style={styles.macroCard}>
        <MacroBar label="Protein" current={totals.protein} goal={nutritionGoals.protein} color="#3B82F6" unit="g" />
        <MacroBar label="Carbs" current={totals.carbs} goal={nutritionGoals.carbs} color="#D97706" unit="g" />
        <MacroBar label="Fat" current={totals.fat} goal={nutritionGoals.fat} color="#16A34A" unit="g" />
      </View>

      {/* Action buttons */}
      <View style={styles.actionRow}>
        <Pressable style={styles.actionBtn} onPress={() => navigation.navigate('Detection')}>
          <Text style={styles.actionIcon}>📷</Text>
          <Text style={styles.actionLabel}>Scan Food</Text>
        </Pressable>
        <Pressable style={[styles.actionBtn, styles.actionBtnSecondary]} onPress={() => setQuickAddVisible(true)}>
          <Text style={styles.actionIcon}>⚡</Text>
          <Text style={styles.actionLabel}>Quick Add</Text>
        </Pressable>
      </View>

      {/* Favourites */}
      {favourites.length > 0 && (
        <View style={styles.recentSection}>
          <Text style={styles.sectionTitle}>⭐ Favourites</Text>
          {favourites.map((fav) => (
            <Pressable
              key={fav.id}
              style={styles.recentRow}
              onPress={() => handleLogFavourite(fav)}
            >
              <View style={styles.recentInfo}>
                <Text style={styles.recentName} numberOfLines={1}>{fav.name}</Text>
                <Text style={styles.recentMeta}>Used {fav.timesUsed}× · tap to log</Text>
              </View>
              <View style={styles.recentRight}>
                <Text style={styles.recentCal}>{Math.round(fav.totalCalories)} kcal</Text>
              </View>
            </Pressable>
          ))}
        </View>
      )}

      {/* Recent meals */}
      {recentMeals.length > 0 && (
        <View style={styles.recentSection}>
          <Text style={styles.sectionTitle}>Recent Meals</Text>
          {recentMeals.map((meal, i) => (
            <Pressable
              key={`${meal.entryId}-${i}`}
              style={styles.recentRow}
              onPress={() => handleRelogMeal(meal)}
            >
              <View style={styles.recentInfo}>
                <Text style={styles.recentName} numberOfLines={1}>{meal.dishName}</Text>
                <Text style={styles.recentMeta}>{meal.mealType} · {meal.entryDate}</Text>
              </View>
              <View style={styles.recentRight}>
                <Text style={styles.recentCal}>{Math.round(meal.calories)} kcal</Text>
                <Text style={styles.relogHint}>tap to re-log</Text>
              </View>
            </Pressable>
          ))}
        </View>
      )}

      <View style={{ height: 100 }} />

      {/* Quick Add Modal */}
      <Modal visible={quickAddVisible} transparent animationType="fade" onRequestClose={() => setQuickAddVisible(false)}>
        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.modalOverlay}>
          <Pressable style={styles.modalBackdrop} onPress={() => setQuickAddVisible(false)} />
          <View style={styles.quickAddSheet}>
            <Text style={styles.quickAddTitle}>Quick Add Calories</Text>
            <TextInput
              style={styles.quickAddInput}
              value={quickAddCal}
              onChangeText={setQuickAddCal}
              keyboardType="number-pad"
              placeholder="Enter calories"
              placeholderTextColor="#9CA3AF"
              autoFocus
              returnKeyType="done"
              onSubmitEditing={handleQuickAdd}
            />
            <View style={styles.quickAddActions}>
              <Pressable style={styles.quickAddCancel} onPress={() => setQuickAddVisible(false)}>
                <Text style={styles.quickAddCancelText}>Cancel</Text>
              </Pressable>
              <Pressable style={styles.quickAddConfirm} onPress={handleQuickAdd}>
                <Text style={styles.quickAddConfirmText}>Add</Text>
              </Pressable>
            </View>
          </View>
        </KeyboardAvoidingView>
      </Modal>
    </ScrollView>
  );
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function CalorieRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <View style={styles.calRow}>
      <Text style={[styles.calRowLabel, { color }]}>{label}</Text>
      <Text style={[styles.calRowValue, { color }]}>{Math.round(value)}</Text>
    </View>
  );
}

function MacroBar({ label, current, goal, color, unit }: {
  label: string; current: number; goal: number; color: string; unit: string;
}) {
  const pct = Math.min(1, current / goal);
  return (
    <View style={styles.macroRow}>
      <View style={styles.macroLabelRow}>
        <Text style={styles.macroLabel}>{label}</Text>
        <Text style={styles.macroValues}>
          <Text style={{ fontWeight: '700', color }}>{Math.round(current)}{unit}</Text>
          <Text style={{ color: '#9CA3AF' }}> / {goal}{unit}</Text>
        </Text>
      </View>
      <View style={styles.barBg}>
        <View style={[styles.barFill, { width: `${pct * 100}%`, backgroundColor: color }]} />
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingTop: 60, paddingHorizontal: 16 },
  headerRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20,
  },
  greeting: { fontSize: 28, fontWeight: '800', color: '#111827', marginBottom: 4 },
  dateLabel: { fontSize: 14, color: '#6B7280' },
  streakBadge: {
    backgroundColor: '#FEF3C7', borderRadius: 14, paddingHorizontal: 14, paddingVertical: 8,
    alignItems: 'center', borderWidth: 1, borderColor: '#FDE68A',
  },
  streakNum: { fontSize: 20, fontWeight: '800', color: '#D97706' },
  streakLabel: { fontSize: 10, fontWeight: '600', color: '#92400E' },

  // Calorie card
  calorieCard: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 20, marginBottom: 12,
    flexDirection: 'row', alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  calorieRing: { marginRight: 20 },
  ringOuter: {
    width: 100, height: 100, borderRadius: 50, borderWidth: 8,
    alignItems: 'center', justifyContent: 'center',
  },
  ringCalNum: { fontSize: 22, fontWeight: '800', color: '#111827' },
  ringCalLabel: { fontSize: 11, color: '#6B7280' },
  calorieDetails: { flex: 1 },
  calRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  calRowLabel: { fontSize: 14, fontWeight: '500' },
  calRowValue: { fontSize: 14, fontWeight: '700' },

  // Macro card
  macroCard: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  macroRow: { marginBottom: 12 },
  macroLabelRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  macroLabel: { fontSize: 14, fontWeight: '600', color: '#374151' },
  macroValues: { fontSize: 13 },
  barBg: { height: 8, backgroundColor: '#F3F4F6', borderRadius: 4, overflow: 'hidden' },
  barFill: { height: 8, borderRadius: 4 },

  // Actions
  actionRow: { flexDirection: 'row', gap: 12, marginBottom: 24 },
  actionBtn: {
    flex: 1, backgroundColor: '#16A34A', borderRadius: 14, paddingVertical: 16,
    alignItems: 'center', gap: 4,
  },
  actionBtnSecondary: { backgroundColor: '#3B82F6' },
  actionIcon: { fontSize: 20 },
  actionLabel: { fontSize: 15, fontWeight: '700', color: '#FFF' },

  // Recent
  recentSection: { marginBottom: 20 },
  sectionTitle: { fontSize: 18, fontWeight: '700', color: '#111827', marginBottom: 12 },
  recentRow: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#FFF',
    borderRadius: 12, padding: 14, marginBottom: 8,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.03, shadowRadius: 4, elevation: 1,
  },
  recentInfo: { flex: 1 },
  recentName: { fontSize: 14, fontWeight: '600', color: '#111827', marginBottom: 2 },
  recentMeta: { fontSize: 12, color: '#9CA3AF' },
  recentRight: { alignItems: 'flex-end', marginLeft: 8 },
  recentCal: { fontSize: 15, fontWeight: '700', color: '#111827' },
  relogHint: { fontSize: 10, color: '#16A34A', fontWeight: '500', marginTop: 2 },

  // Quick Add Modal
  modalOverlay: { flex: 1, justifyContent: 'flex-end' },
  modalBackdrop: { flex: 1 },
  quickAddSheet: {
    backgroundColor: '#FFF', borderTopLeftRadius: 20, borderTopRightRadius: 20,
    padding: 24, paddingBottom: 40,
    shadowColor: '#000', shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 10,
  },
  quickAddTitle: { fontSize: 18, fontWeight: '700', color: '#111827', marginBottom: 16, textAlign: 'center' },
  quickAddInput: {
    backgroundColor: '#F3F4F6', borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14,
    fontSize: 24, fontWeight: '700', color: '#111827', textAlign: 'center', marginBottom: 16,
  },
  quickAddActions: { flexDirection: 'row', gap: 12 },
  quickAddCancel: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: '#F3F4F6', alignItems: 'center' },
  quickAddCancelText: { fontSize: 16, fontWeight: '600', color: '#6B7280' },
  quickAddConfirm: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: '#16A34A', alignItems: 'center' },
  quickAddConfirmText: { fontSize: 16, fontWeight: '700', color: '#FFF' },
});
