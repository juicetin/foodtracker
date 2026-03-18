import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import type { ScannedDish, ScannedIngredient } from '../../types';
import IngredientRow from './IngredientRow';

interface Props {
  dish: ScannedDish;
  onScaleChange: (dishId: string, scale: number) => void;
  onNameChange: (dishId: string, name: string) => void;
  /** Opens the KG ingredient search sheet for a specific ingredient. */
  onIngredientNameTap: (dishId: string, ingId: string, currentName: string) => void;
  onIngredientWeightChange: (dishId: string, ingId: string, amount_g: number) => void;
  onRemoveIngredient: (dishId: string, ingId: string) => void;
  onRemove: (dishId: string) => void;
}

const MIN_SCALE = 0.25;
const MAX_SCALE = 3.0;
const SCALE_STEP = 0.25;

function formatScale(s: number): string {
  const rounded = Math.round(s * 100) / 100;
  return rounded % 1 === 0 ? `${rounded}×` : `${rounded.toFixed(2)}×`;
}

export default function DishCard({
  dish,
  onScaleChange,
  onNameChange,
  onIngredientNameTap,
  onIngredientWeightChange,
  onRemoveIngredient,
  onRemove,
}: Props) {
  const [editingName, setEditingName] = useState(false);
  const [nameValue, setNameValue] = useState(dish.name);

  const scale = dish.portionScale;

  // Totals: nutrition stored at originalAmount_g, scale by current amount_g ratio
  const totals = dish.ingredients.reduce(
    (acc, ing) => {
      const s = ing.originalAmount_g > 0 ? ing.amount_g / ing.originalAmount_g : 1;
      return {
        calories: acc.calories + ing.calories * s,
        protein:  acc.protein  + ing.protein  * s,
        carbs:    acc.carbs    + ing.carbs    * s,
        fat:      acc.fat      + ing.fat      * s,
      };
    },
    { calories: 0, protein: 0, carbs: 0, fat: 0 },
  );

  function handleScaleDown() {
    const next = Math.max(MIN_SCALE, Math.round((scale - SCALE_STEP) * 4) / 4);
    onScaleChange(dish.id, next);
  }

  function handleScaleUp() {
    const next = Math.min(MAX_SCALE, Math.round((scale + SCALE_STEP) * 4) / 4);
    onScaleChange(dish.id, next);
  }

  function handleNameSubmit() {
    setEditingName(false);
    const trimmed = nameValue.trim();
    if (trimmed) {
      onNameChange(dish.id, trimmed);
    } else {
      setNameValue(dish.name);
    }
  }

  return (
    <View style={styles.card}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          {editingName ? (
            <TextInput
              style={styles.nameInput}
              value={nameValue}
              onChangeText={setNameValue}
              onBlur={handleNameSubmit}
              onSubmitEditing={handleNameSubmit}
              returnKeyType="done"
              autoFocus
            />
          ) : (
            <TouchableOpacity onPress={() => setEditingName(true)} activeOpacity={0.7}>
              <Text style={styles.dishName}>{dish.name}</Text>
            </TouchableOpacity>
          )}
          {dish.cuisine && (
            <View style={styles.cuisinePill}>
              <Text style={styles.cuisineText}>{dish.cuisine}</Text>
            </View>
          )}
        </View>
        <TouchableOpacity
          onPress={() => onRemove(dish.id)}
          style={styles.removeBtn}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Text style={styles.removeBtnText}>✕</Text>
        </TouchableOpacity>
      </View>

      {/* Scale control */}
      <View style={styles.scaleRow}>
        <Text style={styles.scaleLabel}>Portion</Text>
        <View style={styles.scaleControls}>
          <TouchableOpacity
            onPress={handleScaleDown}
            style={[styles.scaleBtn, scale <= MIN_SCALE && styles.scaleBtnDisabled]}
            disabled={scale <= MIN_SCALE}
          >
            <Text style={[styles.scaleBtnText, scale <= MIN_SCALE && styles.scaleBtnTextDisabled]}>−</Text>
          </TouchableOpacity>
          <Text style={styles.scaleValue}>{formatScale(scale)}</Text>
          <TouchableOpacity
            onPress={handleScaleUp}
            style={[styles.scaleBtn, scale >= MAX_SCALE && styles.scaleBtnDisabled]}
            disabled={scale >= MAX_SCALE}
          >
            <Text style={[styles.scaleBtnText, scale >= MAX_SCALE && styles.scaleBtnTextDisabled]}>+</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.divider} />

      {/* Ingredients */}
      {dish.ingredients.map((ing) => (
        <IngredientRow
          key={ing.id}
          ingredient={ing}
          onNameTap={() => onIngredientNameTap(dish.id, ing.id, ing.name)}
          onWeightChange={(g) => onIngredientWeightChange(dish.id, ing.id, g)}
          onRemove={() => onRemoveIngredient(dish.id, ing.id)}
        />
      ))}

      {dish.ingredients.length === 0 && (
        <Text style={styles.emptyText}>No ingredients</Text>
      )}

      <View style={styles.divider} />

      {/* Nutrition totals */}
      <View style={styles.nutritionRow}>
        <View style={styles.caloriesBlock}>
          <Text style={styles.caloriesNum}>{Math.round(totals.calories)}</Text>
          <Text style={styles.caloriesLabel}>kcal</Text>
        </View>
        <View style={styles.macroChips}>
          <MacroChip value={totals.protein} label="P" color="#3B82F6" bg="#EFF6FF" />
          <MacroChip value={totals.carbs}   label="C" color="#D97706" bg="#FFFBEB" />
          <MacroChip value={totals.fat}     label="F" color="#16A34A" bg="#F0FDF4" />
        </View>
      </View>
    </View>
  );
}

function MacroChip({ value, label, color, bg }: {
  value: number; label: string; color: string; bg: string;
}) {
  return (
    <View style={[styles.macroChip, { backgroundColor: bg }]}>
      <Text style={[styles.macroNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={[styles.macroLabel, { color }]}> {label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    marginHorizontal: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 14,
    paddingBottom: 8,
    backgroundColor: '#F9FAFB',
  },
  headerLeft: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
    marginRight: 8,
  },
  dishName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  nameInput: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
    borderBottomWidth: 1.5,
    borderBottomColor: '#16A34A',
    paddingVertical: 0,
    minWidth: 120,
  },
  cuisinePill: {
    backgroundColor: '#DCFCE7',
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  cuisineText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#16A34A',
  },
  removeBtn: {
    padding: 4,
  },
  removeBtnText: {
    fontSize: 14,
    color: '#D1D5DB',
    fontWeight: '600',
  },
  scaleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#F9FAFB',
  },
  scaleLabel: {
    fontSize: 13,
    color: '#6B7280',
    fontWeight: '500',
  },
  scaleControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  scaleBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#FFFFFF',
    borderWidth: 1.5,
    borderColor: '#16A34A',
    alignItems: 'center',
    justifyContent: 'center',
  },
  scaleBtnDisabled: {
    borderColor: '#E5E7EB',
    backgroundColor: '#F9FAFB',
  },
  scaleBtnText: {
    fontSize: 18,
    color: '#16A34A',
    fontWeight: '600',
    lineHeight: 20,
  },
  scaleBtnTextDisabled: {
    color: '#D1D5DB',
  },
  scaleValue: {
    fontSize: 15,
    fontWeight: '700',
    color: '#111827',
    minWidth: 40,
    textAlign: 'center',
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: '#F3F4F6',
    marginHorizontal: 16,
  },
  emptyText: {
    fontSize: 13,
    color: '#9CA3AF',
    textAlign: 'center',
    paddingVertical: 12,
  },
  nutritionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  caloriesBlock: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 3,
  },
  caloriesNum: {
    fontSize: 22,
    fontWeight: '800',
    color: '#111827',
  },
  caloriesLabel: {
    fontSize: 13,
    color: '#6B7280',
    fontWeight: '500',
  },
  macroChips: {
    flexDirection: 'row',
    gap: 6,
  },
  macroChip: {
    flexDirection: 'row',
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    alignItems: 'center',
  },
  macroNum: {
    fontSize: 13,
    fontWeight: '700',
  },
  macroLabel: {
    fontSize: 11,
    fontWeight: '600',
  },
});
