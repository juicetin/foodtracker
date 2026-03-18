import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import type { ScannedIngredient } from '../../types';

interface Props {
  ingredient: ScannedIngredient;
  /** Opens the KG-powered ingredient search sheet. */
  onNameTap: () => void;
  onWeightChange: (amount_g: number) => void;
  onRemove: () => void;
}

export default function IngredientRow({ ingredient, onNameTap, onWeightChange, onRemove }: Props) {
  const [editingWeight, setEditingWeight] = useState(false);
  const [weightValue, setWeightValue] = useState(String(Math.round(ingredient.amount_g)));

  const scale = ingredient.originalAmount_g > 0
    ? ingredient.amount_g / ingredient.originalAmount_g
    : 1;
  const displayCalories = Math.round(ingredient.calories * scale);

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
        <Text style={styles.kcal}>{displayCalories} kcal</Text>
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

const styles = StyleSheet.create({
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
    color: '#1A1A1A',
  },
  tapHint: {
    fontSize: 10,
    color: '#BBB',
    marginTop: 1,
  },
  kcal: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  right: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  weightChip: {
    backgroundColor: '#F0FDF4',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: '#BBF7D0',
  },
  weightChipModified: {
    backgroundColor: '#FFF7ED',
    borderColor: '#FED7AA',
  },
  weightText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#16A34A',
  },
  weightInput: {
    fontSize: 13,
    fontWeight: '600',
    color: '#16A34A',
    borderBottomWidth: 1,
    borderBottomColor: '#16A34A',
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
    color: '#CCC',
    fontWeight: '600',
  },
});
