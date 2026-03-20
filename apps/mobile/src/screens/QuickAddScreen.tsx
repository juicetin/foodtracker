/**
 * QuickAddScreen — raw calorie/macro entry form with real-time validation.
 *
 * Users enter Calories + Protein/Carbs/Fat. Macro validation shows a warning
 * when the breakdown doesn't approximately match entered calories, but
 * submission is always allowed (just visually flagged).
 */

import React, { useCallback, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { validateMacros } from '../services/search/historyService';
import { autoDetectMealType } from '../services/detection/types';
import { useFoodLogStore } from '../store/useFoodLogStore';

export default function QuickAddScreen() {
  const navigation = useNavigation();
  const { addEntry, loadTodayEntries } = useFoodLogStore();

  const [foodName, setFoodName] = useState('');
  const [calories, setCalories] = useState('');
  const [protein, setProtein] = useState('');
  const [carbs, setCarbs] = useState('');
  const [fat, setFat] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const cal = parseFloat(calories) || 0;
  const p = parseFloat(protein) || 0;
  const c = parseFloat(carbs) || 0;
  const f = parseFloat(fat) || 0;

  // Only show validation when all four numeric fields have values > 0
  const allFieldsFilled = cal > 0 && p > 0 && c > 0 && f > 0;
  const validation = useMemo(() => validateMacros(cal, p, c, f), [cal, p, c, f]);
  const showWarning = allFieldsFilled && !validation.isValid;

  const canSubmit = cal > 0 && !submitting;

  const handleSubmit = useCallback(async () => {
    if (!canSubmit) return;
    setSubmitting(true);

    try {
      const notes = foodName.trim()
        ? foodName.trim()
        : `Quick Add: ${Math.round(cal)} kcal`;

      await addEntry({
        mealType: autoDetectMealType(),
        totalCalories: Math.round(cal),
        totalProtein: Math.round(p),
        totalCarbs: Math.round(c),
        totalFat: Math.round(f),
        notes,
      });

      await loadTodayEntries();
      navigation.goBack();
    } catch {
      setSubmitting(false);
    }
  }, [canSubmit, foodName, cal, p, c, f, addEntry, loadTodayEntries, navigation]);

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {/* Header */}
        <View style={styles.header}>
          <Pressable onPress={() => navigation.goBack()} hitSlop={12}>
            <Ionicons name="close" size={28} color="#111827" />
          </Pressable>
          <Text style={styles.headerTitle}>Quick Add</Text>
          <View style={{ width: 28 }} />
        </View>

        {/* Food name (optional) */}
        <TextInput
          style={styles.nameInput}
          placeholder="Food name (optional)"
          placeholderTextColor="#9CA3AF"
          value={foodName}
          onChangeText={setFoodName}
          returnKeyType="next"
        />

        {/* Calories — full width, prominent */}
        <View style={styles.caloriesRow}>
          <Text style={styles.fieldLabel}>Calories</Text>
          <TextInput
            style={[
              styles.caloriesInput,
              showWarning && styles.caloriesInputWarning,
            ]}
            placeholder="0"
            placeholderTextColor="#D1D5DB"
            keyboardType="decimal-pad"
            selectTextOnFocus
            value={calories}
            onChangeText={setCalories}
          />
        </View>

        {/* Macros — 3 fields in a row */}
        <View style={styles.macroRow}>
          <View style={styles.macroField}>
            <Text style={styles.fieldLabel}>Protein (g)</Text>
            <TextInput
              style={styles.macroInput}
              placeholder="0"
              placeholderTextColor="#D1D5DB"
              keyboardType="decimal-pad"
              selectTextOnFocus
              value={protein}
              onChangeText={setProtein}
            />
          </View>
          <View style={styles.macroField}>
            <Text style={styles.fieldLabel}>Carbs (g)</Text>
            <TextInput
              style={styles.macroInput}
              placeholder="0"
              placeholderTextColor="#D1D5DB"
              keyboardType="decimal-pad"
              selectTextOnFocus
              value={carbs}
              onChangeText={setCarbs}
            />
          </View>
          <View style={styles.macroField}>
            <Text style={styles.fieldLabel}>Fat (g)</Text>
            <TextInput
              style={styles.macroInput}
              placeholder="0"
              placeholderTextColor="#D1D5DB"
              keyboardType="decimal-pad"
              selectTextOnFocus
              value={fat}
              onChangeText={setFat}
            />
          </View>
        </View>

        {/* Validation warning */}
        {showWarning && (
          <Text style={styles.warningText}>
            Macros suggest ~{Math.round(validation.expected)} kcal (entered{' '}
            {Math.round(cal)})
          </Text>
        )}

        {/* Spacer */}
        <View style={styles.flex} />

        {/* Submit */}
        <Pressable
          style={[styles.submitButton, !canSubmit && styles.submitButtonDisabled]}
          onPress={handleSubmit}
          disabled={!canSubmit}
        >
          <Text style={styles.submitButtonText}>Add to Diary</Text>
        </Pressable>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  flex: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
  },
  nameInput: {
    marginHorizontal: 16,
    marginBottom: 20,
    fontSize: 16,
    color: '#111827',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
    paddingVertical: 10,
  },
  caloriesRow: {
    marginHorizontal: 16,
    marginBottom: 20,
  },
  fieldLabel: {
    fontSize: 13,
    fontWeight: '500',
    color: '#6B7280',
    marginBottom: 6,
  },
  caloriesInput: {
    fontSize: 32,
    fontWeight: '700',
    color: '#111827',
    borderWidth: 1.5,
    borderColor: '#E5E7EB',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    textAlign: 'center',
  },
  caloriesInputWarning: {
    borderColor: '#EF4444',
  },
  macroRow: {
    flexDirection: 'row',
    marginHorizontal: 16,
    gap: 10,
  },
  macroField: {
    flex: 1,
  },
  macroInput: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 12,
    textAlign: 'center',
  },
  warningText: {
    marginHorizontal: 16,
    marginTop: 10,
    fontSize: 13,
    color: '#DC2626',
  },
  submitButton: {
    backgroundColor: '#16A34A',
    borderRadius: 14,
    paddingVertical: 16,
    marginHorizontal: 16,
    marginBottom: 16,
    alignItems: 'center',
  },
  submitButtonDisabled: {
    opacity: 0.5,
  },
  submitButtonText: {
    fontSize: 17,
    fontWeight: '600',
    color: '#FFFFFF',
  },
});
