import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import type { ScannedDish, ScannedIngredient } from '../../types';
import IngredientRow from './IngredientRow';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

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
  return rounded % 1 === 0 ? `${rounded}x` : `${rounded.toFixed(2)}x`;
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
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
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
          <MacroChip value={totals.protein} label="P" color={colors.accent.blue} bg={colors.accentTint.blue} />
          <MacroChip value={totals.carbs}   label="C" color={colors.accent.amber} bg={colors.accentTint.amber} />
          <MacroChip value={totals.fat}     label="F" color={colors.accent.green} bg={colors.accentTint.green} />
        </View>
      </View>
    </View>
  );
}

function MacroChip({ value, label, color, bg }: {
  value: number; label: string; color: string; bg: string;
}) {
  return (
    <View style={[chipStyles.macroChip, { backgroundColor: bg }]}>
      <Text style={[chipStyles.macroNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={[chipStyles.macroLabel, { color }]}> {label}</Text>
    </View>
  );
}

const chipStyles = StyleSheet.create({
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

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    card: {
      backgroundColor: colors.background.elevated,
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
      backgroundColor: colors.background.surface,
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
      color: colors.text.primary,
    },
    nameInput: {
      fontSize: 16,
      fontWeight: '700',
      color: colors.text.primary,
      borderBottomWidth: 1.5,
      borderBottomColor: colors.accent.green,
      paddingVertical: 0,
      minWidth: 120,
    },
    cuisinePill: {
      backgroundColor: colors.accentTint.green,
      borderRadius: 20,
      paddingHorizontal: 10,
      paddingVertical: 3,
    },
    cuisineText: {
      fontSize: 12,
      fontWeight: '500',
      color: colors.accent.green,
    },
    removeBtn: {
      padding: 4,
    },
    removeBtnText: {
      fontSize: 14,
      color: colors.border.default,
      fontWeight: '600',
    },
    scaleRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 16,
      paddingVertical: 8,
      backgroundColor: colors.background.surface,
    },
    scaleLabel: {
      fontSize: 13,
      color: colors.text.tertiary,
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
      backgroundColor: colors.background.elevated,
      borderWidth: 1.5,
      borderColor: colors.accent.green,
      alignItems: 'center',
      justifyContent: 'center',
    },
    scaleBtnDisabled: {
      borderColor: colors.border.subtle,
      backgroundColor: colors.background.surface,
    },
    scaleBtnText: {
      fontSize: 18,
      color: colors.accent.green,
      fontWeight: '600',
      lineHeight: 20,
    },
    scaleBtnTextDisabled: {
      color: colors.border.default,
    },
    scaleValue: {
      fontSize: 15,
      fontWeight: '700',
      color: colors.text.primary,
      minWidth: 40,
      textAlign: 'center',
    },
    divider: {
      height: StyleSheet.hairlineWidth,
      backgroundColor: colors.background.surface,
      marginHorizontal: 16,
    },
    emptyText: {
      fontSize: 13,
      color: colors.text.tertiary,
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
      color: colors.text.primary,
    },
    caloriesLabel: {
      fontSize: 13,
      color: colors.text.tertiary,
      fontWeight: '500',
    },
    macroChips: {
      flexDirection: 'row',
      gap: 6,
    },
  });
}
