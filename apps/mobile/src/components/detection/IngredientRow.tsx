import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import type { ScannedIngredient } from '../../types';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface Props {
  ingredient: ScannedIngredient;
  /** Opens the KG-powered ingredient search sheet. */
  onNameTap: () => void;
  onWeightChange: (amount_g: number) => void;
  onRemove: () => void;
}

export default function IngredientRow({ ingredient, onNameTap, onWeightChange, onRemove }: Props) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const [editingWeight, setEditingWeight] = useState(false);
  const [weightValue, setWeightValue] = useState(String(Math.round(ingredient.amount_g)));

  const scale = ingredient.originalAmount_g > 0
    ? ingredient.amount_g / ingredient.originalAmount_g
    : 1;
  const displayCalories = Math.round(ingredient.calories * scale);
  const displayProtein = Math.round(ingredient.protein * scale);
  const displayCarbs = Math.round(ingredient.carbs * scale);
  const displayFat = Math.round(ingredient.fat * scale);

  function handleWeightSubmit() {
    setEditingWeight(false);
    const num = parseFloat(weightValue);
    if (!isNaN(num) && num > 0) {
      onWeightChange(num);
    } else {
      setWeightValue(String(Math.round(ingredient.amount_g)));
    }
  }

  return (
    <View style={styles.row}>
      {/* Left: name (tap opens search sheet) + kcal */}
      <View style={styles.left}>
        <TouchableOpacity onPress={onNameTap} activeOpacity={0.6}>
          <Text style={styles.name} numberOfLines={1}>{ingredient.name}</Text>
          <Text style={styles.tapHint}>tap to change</Text>
        </TouchableOpacity>
        <Text style={styles.kcal}>{displayCalories} kcal  ·  P {displayProtein}g  C {displayCarbs}g  F {displayFat}g</Text>
      </View>

      {/* Right: weight chip (inline edit) + remove */}
      <View style={styles.right}>
        {editingWeight ? (
          <TextInput
            style={styles.weightInput}
            value={weightValue}
            onChangeText={setWeightValue}
            onBlur={handleWeightSubmit}
            onSubmitEditing={handleWeightSubmit}
            keyboardType="decimal-pad"
            returnKeyType="done"
            autoFocus
            selectTextOnFocus
          />
        ) : (
          <TouchableOpacity onPress={() => { setWeightValue(String(Math.round(ingredient.amount_g))); setEditingWeight(true); }} activeOpacity={0.7}>
            <View style={[styles.weightChip, ingredient.userModified && styles.weightChipModified]}>
              <Text style={styles.weightText}>{Math.round(ingredient.amount_g)}g</Text>
            </View>
          </TouchableOpacity>
        )}
        <TouchableOpacity
          onPress={onRemove}
          style={styles.removeBtn}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Text style={styles.removeBtnText}>✕</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    row: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingVertical: 8,
      paddingHorizontal: 16,
    },
    left: {
      flex: 1,
      marginRight: 12,
    },
    name: {
      fontSize: 14,
      fontWeight: '500',
      color: colors.text.primary,
    },
    tapHint: {
      fontSize: 10,
      color: colors.text.tertiary,
      marginTop: 1,
    },
    kcal: {
      fontSize: 12,
      color: colors.text.tertiary,
      marginTop: 2,
    },
    right: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
    },
    weightChip: {
      backgroundColor: colors.accentTint.green,
      borderRadius: 8,
      paddingHorizontal: 10,
      paddingVertical: 4,
      borderWidth: 1,
      borderColor: colors.accent.green,
    },
    weightChipModified: {
      backgroundColor: colors.accentTint.amber,
      borderColor: colors.accent.amber,
    },
    weightText: {
      fontSize: 13,
      fontWeight: '600',
      color: colors.accent.green,
    },
    weightInput: {
      fontSize: 13,
      fontWeight: '600',
      color: colors.accent.green,
      borderBottomWidth: 1,
      borderBottomColor: colors.accent.green,
      minWidth: 50,
      textAlign: 'right',
      paddingVertical: 0,
    },
    removeBtn: {
      width: 20,
      alignItems: 'center',
    },
    removeBtnText: {
      fontSize: 12,
      color: colors.border.default,
      fontWeight: '600',
    },
  });
}
