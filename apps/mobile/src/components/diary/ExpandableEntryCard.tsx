/**
 * ExpandableEntryCard — three-state tap cycle entry card with photo thumbnail.
 *
 * States: summary (macros visible) -> ingredients (per-ingredient breakdown) -> collapsed (minimal)
 * Visual affordance: three dots at bottom-right showing cycle position.
 * Long-press navigates to entry detail.
 */

import React, { useState, useCallback } from 'react';
import { View, Text, Pressable, Image, StyleSheet } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withTiming, Easing } from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { opsqlite } from '../../../db/client';
import type { DiaryEntry } from '../../services/diary/diaryQueries';

type CardState = 'summary' | 'ingredients' | 'collapsed';
const STATE_CYCLE: CardState[] = ['summary', 'ingredients', 'collapsed'];

interface ExpandableEntryCardProps {
  entry: DiaryEntry;
  onNavigateToDetail: (entryId: string) => void;
}

interface IngredientRow {
  name: string;
  amount_g: number | null;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export function ExpandableEntryCard({ entry, onNavigateToDetail }: ExpandableEntryCardProps) {
  const [cardState, setCardState] = useState<CardState>('summary');
  const [ingredients, setIngredients] = useState<IngredientRow[] | null>(null);
  const [photoError, setPhotoError] = useState(false);

  const animatedHeight = useSharedValue(1);
  const animStyle = useAnimatedStyle(() => ({
    opacity: animatedHeight.value,
  }));

  const handlePress = useCallback(() => {
    const currentIndex = STATE_CYCLE.indexOf(cardState);
    const nextState = STATE_CYCLE[(currentIndex + 1) % STATE_CYCLE.length];

    // Load ingredients when entering ingredients state for first time
    if (nextState === 'ingredients' && ingredients === null) {
      try {
        const rows = opsqlite.executeSync(
          'SELECT name, amount_g, calories, protein, carbs, fat FROM ingredients WHERE entry_id = ? ORDER BY calories DESC',
          [entry.id],
        ).rows as Array<Record<string, unknown>>;
        setIngredients(
          rows.map((r) => ({
            name: r.name as string,
            amount_g: (r.amount_g as number) ?? null,
            calories: (r.calories as number) ?? 0,
            protein: (r.protein as number) ?? 0,
            carbs: (r.carbs as number) ?? 0,
            fat: (r.fat as number) ?? 0,
          })),
        );
      } catch {
        setIngredients([]);
      }
    }

    animatedHeight.value = withTiming(1, {
      duration: 200,
      easing: Easing.inOut(Easing.ease),
    });
    setCardState(nextState);
  }, [cardState, ingredients, entry.id, animatedHeight]);

  const handleLongPress = useCallback(() => {
    onNavigateToDetail(entry.id);
  }, [entry.id, onNavigateToDetail]);

  const time = new Date(entry.createdAt).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  const dishNames =
    entry.dishes.length > 0
      ? entry.dishes.map((d) => d.name).join(', ')
      : entry.notes ?? 'Logged meal';

  const showPhoto = entry.photoUri && !photoError;

  return (
    <Pressable onPress={handlePress} onLongPress={handleLongPress} testID="expandable-entry-card">
      <Animated.View style={[styles.card, animStyle]}>
        {/* Top row: photo + dish names + time */}
        <View style={styles.topRow}>
          {showPhoto ? (
            <Image
              source={{ uri: entry.photoUri! }}
              style={styles.photo}
              resizeMode="cover"
              onError={() => setPhotoError(true)}
            />
          ) : (
            <View style={styles.photoPlaceholder} testID="photo-placeholder">
              <Ionicons name="restaurant-outline" size={24} color="#9CA3AF" />
            </View>
          )}
          <View style={styles.infoBlock}>
            <Text style={styles.dishNames} numberOfLines={1} ellipsizeMode="tail">
              {dishNames}
            </Text>
            <Text style={styles.time}>{time}</Text>
          </View>
        </View>

        {/* Summary state: calorie + macro row */}
        {cardState !== 'collapsed' && (
          <View style={styles.macroRow} testID="macro-row">
            <Text style={styles.calorieText}>{Math.round(entry.totalCalories)} kcal</Text>
            <View style={styles.macroPills}>
              <Text style={[styles.macroPill, { color: '#3B82F6' }]}>
                P {Math.round(entry.totalProtein)}g
              </Text>
              <Text style={[styles.macroPill, { color: '#D97706' }]}>
                C {Math.round(entry.totalCarbs)}g
              </Text>
              <Text style={[styles.macroPill, { color: '#16A34A' }]}>
                F {Math.round(entry.totalFat)}g
              </Text>
            </View>
          </View>
        )}

        {/* Ingredients state: per-ingredient list */}
        {cardState === 'ingredients' && ingredients && ingredients.length > 0 && (
          <View style={styles.ingredientsList} testID="ingredients-list">
            {ingredients.map((ing, i) => (
              <View key={`${ing.name}-${i}`} style={styles.ingredientRow}>
                <Text style={styles.ingredientName} numberOfLines={1}>
                  {ing.name}
                  {ing.amount_g != null ? ` (${Math.round(ing.amount_g)}g)` : ''}
                </Text>
                <Text style={styles.ingredientMacros}>
                  {Math.round(ing.calories)} kcal
                  {' '}P{Math.round(ing.protein)}
                  {' '}C{Math.round(ing.carbs)}
                  {' '}F{Math.round(ing.fat)}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Dot indicator */}
        <View style={styles.dotsRow} testID="state-dots">
          {STATE_CYCLE.map((state) => (
            <View
              key={state}
              style={[
                styles.dot,
                state === cardState ? styles.dotActive : styles.dotInactive,
              ]}
            />
          ))}
        </View>
      </Animated.View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
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
  topRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  photo: {
    width: 64,
    height: 64,
    borderRadius: 10,
    marginRight: 12,
  },
  photoPlaceholder: {
    width: 64,
    height: 64,
    borderRadius: 10,
    marginRight: 12,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoBlock: {
    flex: 1,
  },
  dishNames: {
    fontSize: 15,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 4,
  },
  time: {
    fontSize: 12,
    color: '#9CA3AF',
  },
  macroRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#F3F4F6',
  },
  calorieText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  macroPills: {
    flexDirection: 'row',
    gap: 10,
  },
  macroPill: {
    fontSize: 12,
    fontWeight: '600',
  },
  ingredientsList: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#F3F4F6',
  },
  ingredientRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  ingredientName: {
    fontSize: 13,
    color: '#374151',
    flex: 1,
    marginRight: 8,
  },
  ingredientMacros: {
    fontSize: 11,
    color: '#9CA3AF',
    fontWeight: '500',
  },
  dotsRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 4,
    marginTop: 8,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  dotActive: {
    backgroundColor: '#16A34A',
  },
  dotInactive: {
    backgroundColor: '#D1D5DB',
  },
});
