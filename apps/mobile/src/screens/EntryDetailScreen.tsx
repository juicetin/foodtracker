/**
 * EntryDetailScreen — view a logged food entry's dishes and ingredients.
 *
 * Shows: photo, dish names, all ingredients with weights and nutrition,
 * macro totals. Read-only for now (editing is a future feature).
 */

import React, { useEffect, useState } from 'react';
import {
  Alert,
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Pressable,
} from 'react-native';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import { opsqlite } from '../../db/client';
import type { RootStackParamList } from '../types';
import { addFavourite, isFavourited } from '../services/favourites';
import { useFoodLogStore } from '../store/useFoodLogStore';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DetailIngredient {
  id: string;
  name: string;
  amountG: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

interface DetailDish {
  id: string;
  name: string;
  cuisine: string | null;
  portionScale: number;
  ingredients: DetailIngredient[];
}

interface EntryDetail {
  id: string;
  mealType: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  notes: string | null;
  createdAt: string;
  photoUri: string | null;
  dishes: DetailDish[];
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

function loadEntry(entryId: string): EntryDetail | null {
  const entryRows = opsqlite.execute(
    `SELECT id, meal_type, total_calories, total_protein, total_carbs, total_fat, notes, created_at
     FROM food_entries WHERE id = ?`,
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  if (entryRows.length === 0) return null;
  const row = entryRows[0];

  const photoRows = opsqlite.execute(
    'SELECT uri FROM photos WHERE entry_id = ? LIMIT 1',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const dishRows = opsqlite.execute(
    'SELECT id, name, cuisine, portion_scale FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const dishes: DetailDish[] = dishRows.map((d) => {
    const dishId = d.id as string;
    const ingRows = opsqlite.execute(
      `SELECT id, name, amount_g, calories, protein, carbs, fat
       FROM ingredients WHERE dish_id = ? ORDER BY created_at`,
      [dishId],
    ).rows as Array<Record<string, unknown>>;

    return {
      id: dishId,
      name: d.name as string,
      cuisine: (d.cuisine as string) ?? null,
      portionScale: (d.portion_scale as number) ?? 1,
      ingredients: ingRows.map((i) => ({
        id: i.id as string,
        name: i.name as string,
        amountG: (i.amount_g as number) ?? 0,
        calories: (i.calories as number) ?? 0,
        protein: (i.protein as number) ?? 0,
        carbs: (i.carbs as number) ?? 0,
        fat: (i.fat as number) ?? 0,
      })),
    };
  });

  return {
    id: row.id as string,
    mealType: row.meal_type as string,
    totalCalories: (row.total_calories as number) ?? 0,
    totalProtein: (row.total_protein as number) ?? 0,
    totalCarbs: (row.total_carbs as number) ?? 0,
    totalFat: (row.total_fat as number) ?? 0,
    notes: (row.notes as string) ?? null,
    createdAt: row.created_at as string,
    photoUri: photoRows.length > 0 ? (photoRows[0].uri as string) : null,
    dishes,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function EntryDetailScreen() {
  const navigation = useNavigation();
  const route = useRoute<RouteProp<RootStackParamList, 'EntryDetail'>>();
  const { deleteEntry } = useFoodLogStore();
  const [entry, setEntry] = useState<EntryDetail | null>(null);
  const [alreadyFaved, setAlreadyFaved] = useState(false);

  useEffect(() => {
    const loaded = loadEntry(route.params.entryId);
    setEntry(loaded);
    if (loaded && loaded.dishes.length > 0) {
      setAlreadyFaved(isFavourited(loaded.dishes.map((d) => d.name).join(', ')));
    }
  }, [route.params.entryId]);

  if (!entry) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Loading…</Text>
      </View>
    );
  }

  const time = new Date(entry.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const mealLabel = entry.mealType.charAt(0).toUpperCase() + entry.mealType.slice(1);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Photo */}
        {entry.photoUri && (
          <Image source={{ uri: entry.photoUri }} style={styles.photo} resizeMode="cover" />
        )}

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.mealBadge}>
            <Text style={styles.mealBadgeText}>{mealLabel}</Text>
          </View>
          <Text style={styles.timeText}>{time}</Text>
        </View>

        {/* Totals */}
        <View style={styles.totalsCard}>
          <View style={styles.totalMain}>
            <Text style={styles.totalCalNum}>{Math.round(entry.totalCalories)}</Text>
            <Text style={styles.totalCalLabel}>kcal</Text>
          </View>
          <View style={styles.totalMacros}>
            <MacroPill value={entry.totalProtein} label="P" color="#3B82F6" />
            <MacroPill value={entry.totalCarbs} label="C" color="#D97706" />
            <MacroPill value={entry.totalFat} label="F" color="#16A34A" />
          </View>
        </View>

        {/* Save to favourites */}
        {entry.dishes.length > 0 && !alreadyFaved && (
          <Pressable
            style={styles.favBtn}
            onPress={() => {
              const name = entry.dishes.map((d) => d.name).join(', ');
              addFavourite({
                name,
                totalCalories: entry.totalCalories,
                totalProtein: entry.totalProtein,
                totalCarbs: entry.totalCarbs,
                totalFat: entry.totalFat,
              });
              setAlreadyFaved(true);
              Alert.alert('Saved', `"${name}" added to favourites.`);
            }}
          >
            <Text style={styles.favBtnText}>⭐ Save to Favourites</Text>
          </Pressable>
        )}
        {alreadyFaved && (
          <View style={styles.favedBadge}>
            <Text style={styles.favedBadgeText}>⭐ In your favourites</Text>
          </View>
        )}

        {/* Dishes */}
        {entry.dishes.map((dish) => (
          <View key={dish.id} style={styles.dishCard}>
            <View style={styles.dishHeader}>
              <Text style={styles.dishName}>{dish.name}</Text>
              {dish.cuisine && (
                <View style={styles.cuisinePill}>
                  <Text style={styles.cuisineText}>{dish.cuisine}</Text>
                </View>
              )}
              {dish.portionScale !== 1 && (
                <Text style={styles.scaleText}>{dish.portionScale}×</Text>
              )}
            </View>

            {dish.ingredients.map((ing) => (
              <View key={ing.id} style={styles.ingRow}>
                <View style={styles.ingLeft}>
                  <Text style={styles.ingName}>{ing.name}</Text>
                  <Text style={styles.ingCal}>{Math.round(ing.calories)} kcal</Text>
                </View>
                <View style={styles.ingWeightChip}>
                  <Text style={styles.ingWeightText}>{Math.round(ing.amountG)}g</Text>
                </View>
              </View>
            ))}

            {dish.ingredients.length === 0 && (
              <Text style={styles.noIngText}>No ingredients recorded</Text>
            )}
          </View>
        ))}

        {entry.dishes.length === 0 && entry.notes && (
          <View style={styles.notesCard}>
            <Text style={styles.notesText}>{entry.notes}</Text>
          </View>
        )}

        {/* Delete */}
        <Pressable
          style={styles.deleteBtn}
          onPress={() => {
            Alert.alert('Delete Meal', 'Are you sure? This cannot be undone.', [
              { text: 'Cancel', style: 'cancel' },
              {
                text: 'Delete',
                style: 'destructive',
                onPress: async () => {
                  await deleteEntry(entry.id);
                  if (navigation.canGoBack()) navigation.goBack();
                },
              },
            ]);
          }}
        >
          <Text style={styles.deleteBtnText}>Delete Meal</Text>
        </Pressable>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

function MacroPill({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <View style={styles.macroPill}>
      <Text style={[styles.macroPillNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={[styles.macroPillLabel, { color }]}> {label}</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  scrollContent: { paddingBottom: 20 },
  loadingText: { marginTop: 100, textAlign: 'center', color: '#6B7280', fontSize: 16 },

  photo: { width: '100%', height: 240 },
  header: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    paddingHorizontal: 16, paddingVertical: 14, backgroundColor: '#FFF',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  mealBadge: {
    backgroundColor: '#16A34A', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 4,
  },
  mealBadgeText: { fontSize: 13, fontWeight: '600', color: '#FFF' },
  timeText: { fontSize: 14, color: '#6B7280' },

  totalsCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  totalMain: { flexDirection: 'row', alignItems: 'baseline', gap: 4 },
  totalCalNum: { fontSize: 28, fontWeight: '800', color: '#111827' },
  totalCalLabel: { fontSize: 14, color: '#6B7280' },
  totalMacros: { flexDirection: 'row', gap: 6 },
  macroPill: {
    flexDirection: 'row', backgroundColor: '#F9FAFB', borderRadius: 8,
    paddingHorizontal: 8, paddingVertical: 4, alignItems: 'center',
  },
  macroPillNum: { fontSize: 13, fontWeight: '700' },
  macroPillLabel: { fontSize: 11, fontWeight: '600' },

  favBtn: {
    backgroundColor: '#FEF3C7', borderRadius: 12, paddingVertical: 12, marginHorizontal: 16,
    marginTop: 12, alignItems: 'center', borderWidth: 1, borderColor: '#FDE68A',
  },
  favBtnText: { fontSize: 15, fontWeight: '600', color: '#92400E' },
  favedBadge: {
    backgroundColor: '#F0FDF4', borderRadius: 12, paddingVertical: 10, marginHorizontal: 16,
    marginTop: 12, alignItems: 'center', borderWidth: 1, borderColor: '#BBF7D0',
  },
  favedBadgeText: { fontSize: 14, fontWeight: '500', color: '#16A34A' },

  dishCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  dishHeader: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: 16, paddingVertical: 12, backgroundColor: '#F9FAFB',
  },
  dishName: { fontSize: 16, fontWeight: '700', color: '#111827', flex: 1 },
  cuisinePill: { backgroundColor: '#DCFCE7', borderRadius: 20, paddingHorizontal: 8, paddingVertical: 2 },
  cuisineText: { fontSize: 11, fontWeight: '500', color: '#16A34A' },
  scaleText: { fontSize: 13, fontWeight: '600', color: '#6B7280' },

  ingRow: {
    flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F9FAFB',
  },
  ingLeft: { flex: 1 },
  ingName: { fontSize: 14, fontWeight: '500', color: '#111827' },
  ingCal: { fontSize: 12, color: '#9CA3AF', marginTop: 1 },
  ingWeightChip: {
    backgroundColor: '#F0FDF4', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4,
    borderWidth: 1, borderColor: '#BBF7D0',
  },
  ingWeightText: { fontSize: 13, fontWeight: '600', color: '#16A34A' },
  noIngText: { textAlign: 'center', padding: 16, color: '#9CA3AF', fontSize: 13 },

  notesCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
  },
  notesText: { fontSize: 14, color: '#374151', lineHeight: 20 },
  deleteBtn: {
    marginHorizontal: 16, marginTop: 24, paddingVertical: 14, borderRadius: 12,
    backgroundColor: '#FEF2F2', alignItems: 'center', borderWidth: 1, borderColor: '#FECACA',
  },
  deleteBtnText: { fontSize: 15, fontWeight: '600', color: '#DC2626' },
});
