/**
 * FoodSearchScreen — search the Knowledge Graph for foods and log them.
 *
 * Users search by name, see matching dishes with nutrition info,
 * enter a portion weight, and log to today's diary.
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  FlatList,
  Pressable,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { getKnowledgeGraphService, type DishResult, type MacroResult } from '../services/knowledge-graph';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { autoDetectMealType } from '../services/detection/types';

export default function FoodSearchScreen() {
  const navigation = useNavigation();
  const { addEntry, loadTodayEntries } = useFoodLogStore();

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<DishResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDish, setSelectedDish] = useState<DishResult | null>(null);
  const [portionG, setPortionG] = useState('100');
  const [nutrition, setNutrition] = useState<MacroResult | null>(null);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const doSearch = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 2) {
      setResults([]);
      return;
    }
    setLoading(true);
    try {
      const kg = await getKnowledgeGraphService();
      if (kg) {
        const matches = await kg.searchIngredients(trimmed, 20);
        // Convert ingredient names to DishResult-like objects for display
        const dishResults: DishResult[] = [];
        for (const name of matches.slice(0, 15)) {
          const dish = await kg.searchDish(name);
          if (dish) dishResults.push(dish);
        }
        setResults(dishResults);
      }
    } catch {
      // KG unavailable
    } finally {
      setLoading(false);
    }
  }, []);

  function handleQueryChange(text: string) {
    setQuery(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(text), 300);
  }

  async function handleSelectDish(dish: DishResult) {
    setSelectedDish(dish);
    setPortionG(String(dish.defaultServingGrams ?? 100));
    // Calculate nutrition for default portion
    const grams = dish.defaultServingGrams ?? 100;
    try {
      const kg = await getKnowledgeGraphService();
      if (kg) {
        const result = await kg.calculateDishNutrition(dish.canonicalName, grams);
        setNutrition(result);
      }
    } catch {
      setNutrition(null);
    }
  }

  async function handlePortionChange(text: string) {
    setPortionG(text);
    const grams = parseFloat(text);
    if (isNaN(grams) || grams <= 0 || !selectedDish) {
      setNutrition(null);
      return;
    }
    try {
      const kg = await getKnowledgeGraphService();
      if (kg) {
        const result = await kg.calculateDishNutrition(selectedDish.canonicalName, grams);
        setNutrition(result);
      }
    } catch {
      setNutrition(null);
    }
  }

  async function handleAddFood() {
    if (!selectedDish || !nutrition) return;
    const grams = parseFloat(portionG);
    if (isNaN(grams) || grams <= 0) return;

    await addEntry({
      mealType: autoDetectMealType(),
      totalCalories: Math.round(nutrition.calories),
      totalProtein: Math.round(nutrition.protein),
      totalCarbs: Math.round(nutrition.carbs),
      totalFat: Math.round(nutrition.fat),
      notes: `${selectedDish.canonicalName} (${Math.round(grams)}g)`,
    });
    await loadTodayEntries();

    Alert.alert(
      'Added',
      `${selectedDish.canonicalName} (${Math.round(grams)}g) — ${Math.round(nutrition.calories)} kcal`,
      [{ text: 'OK', onPress: () => { setSelectedDish(null); setQuery(''); setResults([]); } }],
    );
  }

  function handleGoBack() {
    if (selectedDish) {
      setSelectedDish(null);
      setNutrition(null);
    } else if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }

  // ── Detail view (food selected) ──
  if (selectedDish) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.detailContainer}>
          <Pressable onPress={handleGoBack} style={styles.backBtn}>
            <Text style={styles.backBtnText}>← Back to search</Text>
          </Pressable>

          <Text style={styles.detailName}>{selectedDish.canonicalName}</Text>

          <View style={styles.portionRow}>
            <Text style={styles.portionLabel}>Portion:</Text>
            <TextInput
              style={styles.portionInput}
              value={portionG}
              onChangeText={handlePortionChange}
              keyboardType="decimal-pad"
              selectTextOnFocus
            />
            <Text style={styles.portionUnit}>g</Text>
          </View>

          {nutrition && (
            <View style={styles.nutritionCard}>
              <View style={styles.nutritionMain}>
                <Text style={styles.nutritionCalNum}>{Math.round(nutrition.calories)}</Text>
                <Text style={styles.nutritionCalLabel}>kcal</Text>
              </View>
              <View style={styles.nutritionMacros}>
                <NutritionPill value={nutrition.protein} label="Protein" color="#3B82F6" />
                <NutritionPill value={nutrition.carbs} label="Carbs" color="#D97706" />
                <NutritionPill value={nutrition.fat} label="Fat" color="#16A34A" />
              </View>
              <Text style={styles.nutritionSource}>
                Source: {nutrition.source === 'recipe' ? 'Recipe decomposition' : nutrition.source === 'dish_average' ? 'Dish average' : 'Estimate'}
              </Text>
            </View>
          )}

          <Pressable style={styles.addBtn} onPress={handleAddFood}>
            <Text style={styles.addBtnText}>Add to Diary</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  // ── Search view ──
  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}
      >
        <View style={styles.header}>
          <Pressable onPress={handleGoBack} style={styles.headerClose}>
            <Text style={styles.headerCloseText}>✕</Text>
          </Pressable>
          <Text style={styles.headerTitle}>Search Food</Text>
          <View style={{ width: 36 }} />
        </View>

        <View style={styles.searchRow}>
          <TextInput
            style={styles.searchInput}
            value={query}
            onChangeText={handleQueryChange}
            placeholder="Search foods..."
            placeholderTextColor="#9CA3AF"
            autoFocus
            returnKeyType="search"
          />
        </View>

        <FlatList
          data={results}
          keyExtractor={(item) => String(item.id)}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.listContent}
          renderItem={({ item }) => (
            <Pressable style={styles.resultRow} onPress={() => handleSelectDish(item)}>
              <Text style={styles.resultName}>{item.canonicalName}</Text>
              {item.avgCaloriesPerServing != null && (
                <Text style={styles.resultCal}>
                  {Math.round(item.avgCaloriesPerServing)} kcal/serving
                </Text>
              )}
            </Pressable>
          )}
          ListEmptyComponent={
            query.trim().length >= 2 && !loading ? (
              <View style={styles.emptyList}>
                <Text style={styles.emptyText}>No foods found for "{query.trim()}"</Text>
                <Text style={styles.emptySubtext}>Try a different name or use Quick Add for custom entries</Text>
              </View>
            ) : null
          }
        />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function NutritionPill({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <View style={styles.nutPill}>
      <Text style={[styles.nutPillNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={styles.nutPillLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },

  // Header
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingVertical: 14, backgroundColor: '#FFF',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#E5E7EB',
  },
  headerClose: { padding: 4, width: 36 },
  headerCloseText: { fontSize: 18, color: '#9CA3AF', fontWeight: '600' },
  headerTitle: { fontSize: 17, fontWeight: '700', color: '#111827' },

  // Search
  searchRow: { paddingHorizontal: 16, paddingVertical: 12, backgroundColor: '#FFF' },
  searchInput: {
    backgroundColor: '#F3F4F6', borderRadius: 12, paddingHorizontal: 16, paddingVertical: 12,
    fontSize: 16, color: '#111827',
  },

  // Results
  listContent: { paddingTop: 8 },
  resultRow: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginBottom: 4,
    borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14,
  },
  resultName: { fontSize: 15, fontWeight: '600', color: '#111827' },
  resultCal: { fontSize: 13, color: '#6B7280', marginTop: 2 },
  emptyList: { alignItems: 'center', paddingVertical: 40, paddingHorizontal: 20 },
  emptyText: { fontSize: 15, color: '#6B7280', textAlign: 'center', marginBottom: 4 },
  emptySubtext: { fontSize: 13, color: '#9CA3AF', textAlign: 'center' },

  // Detail
  detailContainer: { flex: 1, padding: 20 },
  backBtn: { marginBottom: 16 },
  backBtnText: { fontSize: 15, color: '#3B82F6', fontWeight: '500' },
  detailName: { fontSize: 24, fontWeight: '800', color: '#111827', marginBottom: 20 },
  portionRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 20 },
  portionLabel: { fontSize: 15, color: '#374151', fontWeight: '500' },
  portionInput: {
    backgroundColor: '#FFF', borderRadius: 10, paddingHorizontal: 16, paddingVertical: 10,
    fontSize: 18, fontWeight: '700', color: '#111827', minWidth: 80, textAlign: 'center',
    borderWidth: 1, borderColor: '#E5E7EB',
  },
  portionUnit: { fontSize: 15, color: '#6B7280' },

  nutritionCard: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 24,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  nutritionMain: {
    flexDirection: 'row', alignItems: 'baseline', gap: 4, marginBottom: 12,
  },
  nutritionCalNum: { fontSize: 32, fontWeight: '800', color: '#111827' },
  nutritionCalLabel: { fontSize: 16, color: '#6B7280' },
  nutritionMacros: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  nutritionSource: { fontSize: 11, color: '#9CA3AF', fontStyle: 'italic' },

  nutPill: {
    flex: 1, backgroundColor: '#F9FAFB', borderRadius: 10, paddingVertical: 8, alignItems: 'center',
  },
  nutPillNum: { fontSize: 16, fontWeight: '700' },
  nutPillLabel: { fontSize: 11, color: '#9CA3AF', fontWeight: '500', marginTop: 2 },

  addBtn: {
    backgroundColor: '#16A34A', borderRadius: 14, paddingVertical: 16, alignItems: 'center',
  },
  addBtnText: { color: '#FFF', fontSize: 17, fontWeight: '700' },
});
