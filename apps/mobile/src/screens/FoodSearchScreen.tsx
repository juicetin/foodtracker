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
import { Ionicons } from '@expo/vector-icons';
import { getKnowledgeGraphService, type DishResult, type MacroResult } from '../services/knowledge-graph';
import { searchProducts, type OFFProduct } from '../services/openfoodfacts/openFoodFactsService';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { autoDetectMealType } from '../services/detection/types';

/** Unified search result — either from KG or OFF. */
interface SearchResult {
  id: string;
  name: string;
  brand?: string | null;
  calorieHint?: number;
  source: 'kg' | 'off';
  kgDish?: DishResult;
  offProduct?: OFFProduct;
}

export default function FoodSearchScreen() {
  const navigation = useNavigation();
  const { addEntry, loadTodayEntries } = useFoodLogStore();

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
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
      const unified: SearchResult[] = [];

      // Search KG first (local, fast)
      const kg = await getKnowledgeGraphService();
      if (kg) {
        const matches = await kg.searchIngredients(trimmed, 15);
        for (const name of matches.slice(0, 10)) {
          const dish = await kg.searchDish(name);
          if (dish) {
            unified.push({
              id: `kg-${dish.id}`,
              name: dish.canonicalName,
              calorieHint: dish.avgCaloriesPerServing ?? undefined,
              source: 'kg',
              kgDish: dish,
            });
          }
        }
      }

      // Also search OFF (remote, broader coverage)
      const offResults = await searchProducts(trimmed, 10);
      for (const p of offResults) {
        unified.push({
          id: `off-${p.barcode}`,
          name: p.name,
          brand: p.brand,
          calorieHint: p.nutrimentsPer100g.calories,
          source: 'off',
          offProduct: p,
        });
      }

      setResults(unified);
    } catch {
      // Search failed
    } finally {
      setLoading(false);
    }
  }, []);

  function handleQueryChange(text: string) {
    setQuery(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(text), 300);
  }

  async function handleSelectResult(result: SearchResult) {
    setSelectedResult(result);

    if (result.source === 'kg' && result.kgDish) {
      const grams = result.kgDish.defaultServingGrams ?? 100;
      setPortionG(String(grams));
      try {
        const kg = await getKnowledgeGraphService();
        if (kg) {
          const n = await kg.calculateDishNutrition(result.kgDish.canonicalName, grams);
          setNutrition(n);
        }
      } catch {
        setNutrition(null);
      }
    } else if (result.source === 'off' && result.offProduct) {
      const serving = result.offProduct.servingQuantityG ?? 100;
      setPortionG(String(Math.round(serving)));
      // Build MacroResult from OFF data
      const scale = serving / 100;
      const n = result.offProduct.nutrimentsPer100g;
      setNutrition({
        calories: n.calories * scale,
        protein: n.protein * scale,
        carbs: n.carbs * scale,
        fat: n.fat * scale,
        source: 'off' as any,
      });
    }
  }

  async function handlePortionChange(text: string) {
    setPortionG(text);
    const grams = parseFloat(text);
    if (isNaN(grams) || grams <= 0 || !selectedResult) {
      setNutrition(null);
      return;
    }

    if (selectedResult.source === 'kg' && selectedResult.kgDish) {
      try {
        const kg = await getKnowledgeGraphService();
        if (kg) {
          const n = await kg.calculateDishNutrition(selectedResult.kgDish.canonicalName, grams);
          setNutrition(n);
        }
      } catch {
        setNutrition(null);
      }
    } else if (selectedResult.source === 'off' && selectedResult.offProduct) {
      const scale = grams / 100;
      const n = selectedResult.offProduct.nutrimentsPer100g;
      setNutrition({
        calories: n.calories * scale,
        protein: n.protein * scale,
        carbs: n.carbs * scale,
        fat: n.fat * scale,
        source: 'off' as any,
      });
    }
  }

  async function handleAddFood() {
    if (!selectedResult || !nutrition) return;
    const grams = parseFloat(portionG);
    if (isNaN(grams) || grams <= 0) return;

    const displayName = selectedResult.brand
      ? `${selectedResult.brand} — ${selectedResult.name}`
      : selectedResult.name;

    await addEntry({
      mealType: autoDetectMealType(),
      totalCalories: Math.round(nutrition.calories),
      totalProtein: Math.round(nutrition.protein),
      totalCarbs: Math.round(nutrition.carbs),
      totalFat: Math.round(nutrition.fat),
      notes: `${displayName} (${Math.round(grams)}g)`,
    });
    await loadTodayEntries();

    Alert.alert(
      'Added',
      `${displayName} (${Math.round(grams)}g) — ${Math.round(nutrition.calories)} kcal`,
      [{ text: 'OK', onPress: () => { setSelectedResult(null); setQuery(''); setResults([]); } }],
    );
  }

  function handleGoBack() {
    if (selectedResult) {
      setSelectedResult(null);
      setNutrition(null);
    } else if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }

  // ── Detail view (food selected) ──
  if (selectedResult) {
    const displayName = selectedResult.brand
      ? `${selectedResult.brand} — ${selectedResult.name}`
      : selectedResult.name;
    const sourceLabel =
      selectedResult.source === 'off'
        ? 'Open Food Facts'
        : nutrition?.source === 'recipe'
          ? 'Recipe decomposition'
          : nutrition?.source === 'dish_average'
            ? 'Dish average'
            : 'Estimate';

    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.detailContainer}>
          <Pressable onPress={handleGoBack} style={styles.backBtn}>
            <Text style={styles.backBtnText}>← Back to search</Text>
          </Pressable>

          <Text style={styles.detailName}>{displayName}</Text>

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
                Source: {sourceLabel}
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
            <Ionicons name="close" size={22} color="#9CA3AF" />
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
          keyExtractor={(item) => item.id}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.listContent}
          renderItem={({ item }) => (
            <Pressable style={styles.resultRow} onPress={() => handleSelectResult(item)}>
              <View style={styles.resultLeft}>
                <Text style={styles.resultName} numberOfLines={1}>{item.name}</Text>
                {item.brand && <Text style={styles.resultBrand} numberOfLines={1}>{item.brand}</Text>}
              </View>
              <View style={styles.resultRight}>
                {item.calorieHint != null && (
                  <Text style={styles.resultCal}>
                    {Math.round(item.calorieHint)} kcal
                  </Text>
                )}
                <View style={[styles.sourceBadge, item.source === 'off' ? styles.sourceBadgeOff : styles.sourceBadgeKg]}>
                  <Text style={styles.sourceBadgeText}>{item.source === 'off' ? 'OFF' : 'KG'}</Text>
                </View>
              </View>
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
    flexDirection: 'row', alignItems: 'center',
  },
  resultLeft: { flex: 1, marginRight: 8 },
  resultRight: { alignItems: 'flex-end', gap: 4 },
  resultName: { fontSize: 15, fontWeight: '600', color: '#111827' },
  resultBrand: { fontSize: 12, color: '#9CA3AF', marginTop: 1 },
  resultCal: { fontSize: 13, fontWeight: '600', color: '#6B7280' },
  sourceBadge: { borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
  sourceBadgeKg: { backgroundColor: '#DCFCE7' },
  sourceBadgeOff: { backgroundColor: '#DBEAFE' },
  sourceBadgeText: { fontSize: 10, fontWeight: '700', color: '#374151' },
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
