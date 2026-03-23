/**
 * FoodSearchScreen — search the Knowledge Graph for foods and log them.
 *
 * Users search by name, see matching dishes with nutrition info,
 * enter a portion weight, and log to today's diary.
 *
 * Shows history-first results when query is empty.
 * Quick Add accessible from header and empty state.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { getKnowledgeGraphService, type DishResult, type MacroResult } from '../services/knowledge-graph';
import { searchProducts, type OFFProduct } from '../services/openfoodfacts/openFoodFactsService';
import { searchRecipes, logRecipeAsEntry } from '../services/recipes/recipeService';
import { deduplicateResults } from '../services/search/searchDedup';
import { getRecentHistory, searchHistory, type HistoryItem } from '../services/search/historyService';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { autoDetectMealType } from '../services/detection/types';
import type { RootStackParamList } from '../types';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

/** Unified search result — either from KG or OFF. */
interface SearchResult {
  id: string;
  name: string;
  brand?: string | null;
  calorieHint?: number;
  source: 'kg' | 'off' | 'history' | 'recipe';
  kgDish?: DishResult;
  offProduct?: OFFProduct;
  recipeId?: string;
}

export default function FoodSearchScreen() {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { addEntry, loadTodayEntries } = useFoodLogStore();
  const uxMode = usePreferencesStore((s) => s.uxMode);
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [portionG, setPortionG] = useState('100');
  const [nutrition, setNutrition] = useState<MacroResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load history on mount
  useEffect(() => {
    try {
      const items = getRecentHistory(20);
      setHistory(items);
    } catch {
      // History DB might be empty
    }
    setHistoryLoaded(true);
  }, []);

  const doSearch = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 2) {
      setResults([]);
      return;
    }
    setLoading(true);
    try {
      const unified: SearchResult[] = [];

      // Prepend matching history items
      const historyMatches = searchHistory(trimmed, 5);
      for (const h of historyMatches) {
        unified.push({
          id: `history-${h.name}`,
          name: h.name,
          calorieHint: h.avgCalories,
          source: 'history',
        });
      }

      // Search recipes (local, fast)
      const recipeMatches = searchRecipes(trimmed, 5);
      for (const r of recipeMatches) {
        unified.push({
          id: `recipe-${r.id}`,
          name: r.name,
          calorieHint: Math.round(r.totalCalories / (r.servings || 1)),
          source: 'recipe',
          recipeId: r.id,
        });
      }

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

      setResults(deduplicateResults(unified));
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

  function handleHistoryItemPress(item: HistoryItem) {
    setQuery(item.name);
    doSearch(item.name);
  }

  async function handleSelectResult(result: SearchResult) {
    // If history item, trigger a search for full KG/OFF results
    if (result.source === 'history') {
      setQuery(result.name);
      doSearch(result.name);
      return;
    }

    // If recipe item, log it directly
    if (result.source === 'recipe' && result.recipeId) {
      if (uxMode === 'zero-effort') {
        logRecipeAsEntry(result.recipeId, autoDetectMealType());
        await loadTodayEntries();
        Alert.alert('Logged', `${result.name} added to diary.`);
      } else {
        Alert.alert(
          'Log Recipe',
          `${result.name}\n${result.calorieHint ?? 0} Cal per serving`,
          [
            { text: 'Cancel', style: 'cancel' },
            {
              text: 'Log',
              onPress: async () => {
                logRecipeAsEntry(result.recipeId!, autoDetectMealType());
                await loadTodayEntries();
                Alert.alert('Logged', `${result.name} added to diary.`);
              },
            },
          ],
        );
      }
      return;
    }

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
      `${displayName} (${Math.round(grams)}g) -- ${Math.round(nutrition.calories)} kcal`,
      [
        { text: 'Add Another', onPress: () => { setSelectedResult(null); setQuery(''); setResults([]); } },
        { text: 'Done', onPress: () => navigation.goBack() },
      ],
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
                <NutritionPill value={nutrition.protein} label="Protein" color={colors.accent.blue} colors={colors} />
                <NutritionPill value={nutrition.carbs} label="Carbs" color="#D97706" colors={colors} />
                <NutritionPill value={nutrition.fat} label="Fat" color={colors.accent.green} colors={colors} />
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

  // Show history when query is empty
  const showHistory = query.trim().length < 2 && historyLoaded && history.length > 0;

  // ── Search view ──
  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}
      >
        <View style={styles.header}>
          <Pressable onPress={handleGoBack} style={styles.headerClose}>
            <Ionicons name="close" size={22} color={colors.input.placeholder} />
          </Pressable>
          <Text style={styles.headerTitle}>Search Food</Text>
          <Pressable onPress={() => navigation.navigate('QuickAdd')} style={styles.headerQuickAdd}>
            <Ionicons name="add-circle-outline" size={24} color={colors.accent.green} />
          </Pressable>
        </View>

        <View style={styles.searchRow}>
          <View style={styles.searchInputContainer}>
            <TextInput
              style={styles.searchInput}
              value={query}
              onChangeText={handleQueryChange}
              placeholder="Search foods..."
              placeholderTextColor={colors.input.placeholder}
              autoFocus
              returnKeyType="search"
            />
            <Pressable style={styles.barcodeBtn} onPress={() => {}}>
              <Ionicons name="barcode-outline" size={20} color={colors.input.placeholder} />
            </Pressable>
          </View>
        </View>

        {showHistory ? (
          <FlatList
            data={history}
            keyExtractor={(item) => `history-${item.name}`}
            keyboardShouldPersistTaps="handled"
            contentContainerStyle={styles.listContent}
            ListHeaderComponent={
              <Text style={styles.sectionHeader}>From History</Text>
            }
            renderItem={({ item }) => (
              <Pressable style={styles.resultRow} onPress={() => handleHistoryItemPress(item)}>
                <View style={styles.historyIconWrap}>
                  <Ionicons name="time-outline" size={18} color={colors.input.placeholder} />
                </View>
                <View style={styles.resultLeft}>
                  <Text style={styles.resultName} numberOfLines={1}>{item.name}</Text>
                </View>
                <View style={styles.resultRight}>
                  <View style={styles.countBadge}>
                    <Text style={styles.countBadgeText}>x{item.totalCount}</Text>
                  </View>
                  <Text style={styles.resultCal}>{item.avgCalories} kcal</Text>
                </View>
              </Pressable>
            )}
          />
        ) : (
          <FlatList
            data={results}
            keyExtractor={(item) => item.id}
            keyboardShouldPersistTaps="handled"
            contentContainerStyle={styles.listContent}
            renderItem={({ item }) => (
              <Pressable style={styles.resultRow} onPress={() => handleSelectResult(item)}>
                {item.source === 'history' && (
                  <View style={styles.historyIconWrap}>
                    <Ionicons name="time-outline" size={18} color={colors.input.placeholder} />
                  </View>
                )}
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
                  <View style={[
                    styles.sourceBadge,
                    item.source === 'off' ? styles.sourceBadgeOff
                      : item.source === 'history' ? styles.sourceBadgeHistory
                      : item.source === 'recipe' ? styles.sourceBadgeRecipe
                      : styles.sourceBadgeKg
                  ]}>
                    <Text style={[
                      styles.sourceBadgeText,
                      item.source === 'recipe' && { color: colors.accent.purple },
                    ]}>
                      {item.source === 'off' ? 'OFF' : item.source === 'history' ? 'History' : item.source === 'recipe' ? 'Recipe' : 'KG'}
                    </Text>
                  </View>
                </View>
              </Pressable>
            )}
            ListEmptyComponent={
              query.trim().length >= 2 && !loading ? (
                <View style={styles.emptyList}>
                  <Text style={styles.emptyText}>No foods found for "{query.trim()}"</Text>
                  <Text style={styles.emptySubtext}>Try a different name or use Quick Add for custom entries</Text>
                  <Pressable
                    style={styles.quickAddLink}
                    onPress={() => navigation.navigate('QuickAdd')}
                  >
                    <Ionicons name="add-circle-outline" size={18} color={colors.accent.green} />
                    <Text style={styles.quickAddLinkText}>Quick Add</Text>
                  </Pressable>
                </View>
              ) : null
            }
          />
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function NutritionPill({ value, label, color, colors }: { value: number; label: string; color: string; colors: ThemeColors }) {
  return (
    <View style={{ flex: 1, backgroundColor: colors.background.surface, borderRadius: 10, paddingVertical: 8, alignItems: 'center' }}>
      <Text style={{ fontSize: 16, fontWeight: '700', color }}>{Math.round(value)}g</Text>
      <Text style={{ fontSize: 11, color: colors.input.placeholder, fontWeight: '500', marginTop: 2 }}>{label}</Text>
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.background.primary },

    // Header
    header: {
      flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
      paddingHorizontal: 16, paddingVertical: 14, backgroundColor: colors.background.elevated,
      borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
    },
    headerClose: { padding: 4, width: 36 },
    headerTitle: { fontSize: 17, fontWeight: '700', color: colors.text.primary },
    headerQuickAdd: { padding: 4, width: 36, alignItems: 'flex-end' },

    // Search
    searchRow: { paddingHorizontal: 16, paddingVertical: 12, backgroundColor: colors.background.elevated },
    searchInputContainer: {
      flexDirection: 'row', alignItems: 'center',
      backgroundColor: colors.input.background, borderRadius: 12,
    },
    searchInput: {
      flex: 1, paddingHorizontal: 16, paddingVertical: 12,
      fontSize: 16, color: colors.text.primary,
    },
    barcodeBtn: { paddingHorizontal: 12, paddingVertical: 10 },

    // Section headers
    sectionHeader: {
      fontSize: 13, fontWeight: '700', color: colors.text.tertiary, textTransform: 'uppercase',
      letterSpacing: 0.5, paddingHorizontal: 16, paddingVertical: 8,
    },

    // Results
    listContent: { paddingTop: 8 },
    resultRow: {
      backgroundColor: colors.background.elevated, marginHorizontal: 16, marginBottom: 4,
      borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14,
      flexDirection: 'row', alignItems: 'center',
    },
    historyIconWrap: { marginRight: 10 },
    resultLeft: { flex: 1, marginRight: 8 },
    resultRight: { alignItems: 'flex-end', gap: 4 },
    resultName: { fontSize: 15, fontWeight: '600', color: colors.text.primary },
    resultBrand: { fontSize: 12, color: colors.input.placeholder, marginTop: 1 },
    resultCal: { fontSize: 13, fontWeight: '600', color: colors.text.tertiary },
    sourceBadge: { borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
    sourceBadgeKg: { backgroundColor: colors.accentTint.green },
    sourceBadgeOff: { backgroundColor: '#DBEAFE' },
    sourceBadgeHistory: { backgroundColor: '#F3E8FF' },
    sourceBadgeRecipe: { backgroundColor: colors.accentTint.purple },
    sourceBadgeText: { fontSize: 10, fontWeight: '700', color: colors.text.secondary },
    countBadge: {
      backgroundColor: colors.background.surface, borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2,
    },
    countBadgeText: { fontSize: 11, fontWeight: '700', color: colors.text.tertiary },
    emptyList: { alignItems: 'center', paddingVertical: 40, paddingHorizontal: 20 },
    emptyText: { fontSize: 15, color: colors.text.tertiary, textAlign: 'center', marginBottom: 4 },
    emptySubtext: { fontSize: 13, color: colors.input.placeholder, textAlign: 'center' },
    quickAddLink: {
      flexDirection: 'row', alignItems: 'center', gap: 6,
      marginTop: 16, paddingVertical: 10, paddingHorizontal: 16,
      backgroundColor: colors.accentTint.green, borderRadius: 10,
    },
    quickAddLinkText: { fontSize: 15, fontWeight: '600', color: colors.accent.green },

    // Detail
    detailContainer: { flex: 1, padding: 20 },
    backBtn: { marginBottom: 16 },
    backBtnText: { fontSize: 15, color: colors.accent.blue, fontWeight: '500' },
    detailName: { fontSize: 24, fontWeight: '800', color: colors.text.primary, marginBottom: 20 },
    portionRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 20 },
    portionLabel: { fontSize: 15, color: colors.text.secondary, fontWeight: '500' },
    portionInput: {
      backgroundColor: colors.background.elevated, borderRadius: 10, paddingHorizontal: 16, paddingVertical: 10,
      fontSize: 18, fontWeight: '700', color: colors.text.primary, minWidth: 80, textAlign: 'center',
      borderWidth: 1, borderColor: colors.border.subtle,
    },
    portionUnit: { fontSize: 15, color: colors.text.tertiary },

    nutritionCard: {
      backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16, marginBottom: 24,
      shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
      shadowRadius: 8, elevation: 3,
    },
    nutritionMain: {
      flexDirection: 'row', alignItems: 'baseline', gap: 4, marginBottom: 12,
    },
    nutritionCalNum: { fontSize: 32, fontWeight: '800', color: colors.text.primary },
    nutritionCalLabel: { fontSize: 16, color: colors.text.tertiary },
    nutritionMacros: { flexDirection: 'row', gap: 8, marginBottom: 8 },
    nutritionSource: { fontSize: 11, color: colors.input.placeholder, fontStyle: 'italic' },

    addBtn: {
      backgroundColor: colors.accent.green, borderRadius: 14, paddingVertical: 16, alignItems: 'center',
    },
    addBtnText: { color: colors.text.inverse, fontSize: 17, fontWeight: '700' },
  });
}
