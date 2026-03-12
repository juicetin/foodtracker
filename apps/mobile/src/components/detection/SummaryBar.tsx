import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { DetectedItem, MealType } from '../../services/detection/types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface SummaryBarProps {
  items: DetectedItem[]; // active (non-removed) items only
  mealType: MealType;
  onChangeMealType: (type: MealType) => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MEAL_TYPES: MealType[] = ['breakfast', 'lunch', 'snack', 'dinner'];

const MEAL_TYPE_LABELS: Record<MealType, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  snack: 'Snack',
  dinner: 'Dinner',
};

/**
 * Very rough kcal-per-gram estimate used as a Phase 2 proxy.
 *
 * Real macro calculation will be wired in Phase 3 when the nutrition service
 * is connected.  Until then we use a ballpark ~1.5 kcal/g (mixed food avg).
 */
const KCAL_PER_GRAM = 1.5;

/** Rough protein-per-gram estimate (mixed food average). */
const PROTEIN_PER_GRAM = 0.1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function nextMealType(current: MealType): MealType {
  const idx = MEAL_TYPES.indexOf(current);
  return MEAL_TYPES[(idx + 1) % MEAL_TYPES.length];
}

function computeTotals(items: DetectedItem[]) {
  let totalWeightG = 0;
  for (const item of items) {
    totalWeightG += item.portionEstimate.weightG * item.portionMultiplier;
  }
  return {
    calories: Math.round(totalWeightG * KCAL_PER_GRAM),
    protein: Math.round(totalWeightG * PROTEIN_PER_GRAM),
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Horizontal summary bar displayed below the photo.
 *
 * Format (per locked decision):
 *   "4 items detected ~ ~650 cal ~ 45g protein"
 *
 * Right side: tappable meal-type chip that cycles through options.
 */
export function SummaryBar({ items, mealType, onChangeMealType }: SummaryBarProps) {
  const { calories, protein } = computeTotals(items);
  const count = items.length;

  return (
    <View style={styles.container}>
      <View style={styles.stats}>
        <Text style={styles.statText}>
          {count} {count === 1 ? 'item' : 'items'} detected
        </Text>
        <Text style={styles.separator}>{'\u00B7'}</Text>
        <Text style={styles.statText}>~{calories} cal</Text>
        <Text style={styles.separator}>{'\u00B7'}</Text>
        <Text style={styles.statText}>{protein}g protein</Text>
      </View>

      <Pressable
        style={styles.mealChip}
        onPress={() => onChangeMealType(nextMealType(mealType))}
      >
        <Text style={styles.mealChipText}>{MEAL_TYPE_LABELS[mealType]}</Text>
      </Pressable>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#F8F9FA',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E0E0E0',
  },
  stats: {
    flexDirection: 'row',
    alignItems: 'center',
    flexShrink: 1,
  },
  statText: {
    fontSize: 13,
    color: '#333',
  },
  separator: {
    marginHorizontal: 6,
    fontSize: 13,
    color: '#999',
  },
  mealChip: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#E8F5E9',
    marginLeft: 8,
  },
  mealChipText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2E7D32',
  },
});
