/**
 * IngredientSearchSheet -- bottom sheet for adding ingredients via KG/OFF search.
 *
 * Searches Knowledge Graph and Open Food Facts simultaneously (debounced 300ms).
 * Results show name, cal/100g, P/F/C per 100g, source badge.
 * "Manual Entry" fallback for items not found in either database.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  FlatList,
  Pressable,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import BottomSheet, { BottomSheetView } from '@gorhom/bottom-sheet';
import { Ionicons } from '@expo/vector-icons';
import { getKnowledgeGraphService, type DishResult } from '../../services/knowledge-graph';
import { searchProducts, type OFFProduct } from '../../services/openfoodfacts/openFoodFactsService';
import type { IngredientUpdate } from '../../services/entryEditor/entryEditorService';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface IngredientSearchSheetProps {
  visible: boolean;
  entryId: string;
  dishId: string;
  onAdd: (ingredient: IngredientUpdate) => void;
  onClose: () => void;
}

interface SearchResult {
  id: string;
  name: string;
  calPer100g: number;
  proteinPer100g: number;
  carbsPer100g: number;
  fatPer100g: number;
  fiberPer100g: number;
  source: 'kg' | 'off';
  kgDish?: DishResult;
  offProduct?: OFFProduct;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function IngredientSearchSheet({
  visible,
  entryId,
  dishId,
  onAdd,
  onClose,
}: IngredientSearchSheetProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const bottomSheetRef = useRef<BottomSheet>(null);
  const snapPoints = React.useMemo(() => ['60%', '90%'], []);

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showManual, setShowManual] = useState(false);
  const [manualName, setManualName] = useState('');
  const [manualGrams, setManualGrams] = useState('100');
  const [manualCal, setManualCal] = useState('');
  const [manualProtein, setManualProtein] = useState('');
  const [manualCarbs, setManualCarbs] = useState('');
  const [manualFat, setManualFat] = useState('');

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Open/close sheet
  useEffect(() => {
    if (visible) {
      bottomSheetRef.current?.snapToIndex(0);
      setQuery('');
      setResults([]);
      setShowManual(false);
    } else {
      bottomSheetRef.current?.close();
    }
  }, [visible]);

  const doSearch = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 2) {
      setResults([]);
      return;
    }
    setLoading(true);
    try {
      const unified: SearchResult[] = [];

      const [kgResult, offResult] = await Promise.all([
        (async () => {
          try {
            const kg = await getKnowledgeGraphService();
            if (!kg) return [];
            const matches = await kg.searchIngredients(trimmed, 10);
            const items: SearchResult[] = [];
            for (const name of matches.slice(0, 5)) {
              const dish = await kg.searchDish(name);
              if (dish) {
                const servingG = dish.defaultServingGrams ?? 100;
                const scale = servingG > 0 ? 100 / servingG : 1;
                items.push({
                  id: `kg-${dish.id}`,
                  name: dish.canonicalName,
                  calPer100g: Math.round((dish.avgCaloriesPerServing ?? 0) * scale),
                  proteinPer100g: Math.round((dish.avgProteinPerServing ?? 0) * scale),
                  carbsPer100g: Math.round((dish.avgCarbsPerServing ?? 0) * scale),
                  fatPer100g: Math.round((dish.avgFatPerServing ?? 0) * scale),
                  fiberPer100g: 0,
                  source: 'kg',
                  kgDish: dish,
                });
              }
            }
            return items;
          } catch {
            return [];
          }
        })(),
        (async () => {
          try {
            const products = await searchProducts(trimmed, 5);
            return products.map((p): SearchResult => ({
              id: `off-${p.barcode}`,
              name: p.name,
              calPer100g: Math.round(p.nutrimentsPer100g.calories),
              proteinPer100g: Math.round(p.nutrimentsPer100g.protein),
              carbsPer100g: Math.round(p.nutrimentsPer100g.carbs),
              fatPer100g: Math.round(p.nutrimentsPer100g.fat),
              fiberPer100g: Math.round(p.nutrimentsPer100g.fiber),
              source: 'off',
              offProduct: p,
            }));
          } catch {
            return [];
          }
        })(),
      ]);

      unified.push(...kgResult, ...offResult);
      setResults(unified);
    } catch {
      // Search failed silently
    } finally {
      setLoading(false);
    }
  }, []);

  const handleQueryChange = useCallback((text: string) => {
    setQuery(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(text), 300);
  }, [doSearch]);

  const handleSelectResult = useCallback(async (result: SearchResult) => {
    const portionG = 100;

    let cal = result.calPer100g;
    let protein = result.proteinPer100g;
    let carbs = result.carbsPer100g;
    let fat = result.fatPer100g;
    let fiber = result.fiberPer100g;

    if (result.source === 'kg' && result.kgDish) {
      try {
        const kg = await getKnowledgeGraphService();
        if (kg) {
          const n = await kg.calculateDishNutrition(result.kgDish.canonicalName, portionG);
          if (n) {
            cal = Math.round(n.calories);
            protein = Math.round(n.protein);
            carbs = Math.round(n.carbs);
            fat = Math.round(n.fat);
          }
        }
      } catch {
        // Use per-100g estimates
      }
    }

    onAdd({
      entryId,
      dishId,
      name: result.name,
      amountG: portionG,
      calories: cal,
      protein,
      carbs,
      fat,
      fiber,
    });
    onClose();
  }, [entryId, dishId, onAdd, onClose]);

  const handleManualAdd = useCallback(() => {
    const grams = parseFloat(manualGrams) || 100;
    onAdd({
      entryId,
      dishId,
      name: manualName.trim() || 'Custom ingredient',
      amountG: grams,
      calories: parseFloat(manualCal) || 0,
      protein: parseFloat(manualProtein) || 0,
      carbs: parseFloat(manualCarbs) || 0,
      fat: parseFloat(manualFat) || 0,
      fiber: 0,
    });
    onClose();
  }, [entryId, dishId, manualName, manualGrams, manualCal, manualProtein, manualCarbs, manualFat, onAdd, onClose]);

  const handleSheetChange = useCallback((index: number) => {
    if (index === -1) onClose();
  }, [onClose]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return (
    <BottomSheet
      ref={bottomSheetRef}
      index={-1}
      snapPoints={snapPoints}
      enablePanDownToClose
      onChange={handleSheetChange}
      backgroundStyle={{ backgroundColor: colors.background.elevated }}
      handleIndicatorStyle={{ backgroundColor: colors.border.default }}
    >
      <BottomSheetView style={styles.content}>
        <Text style={styles.title}>Add Ingredient</Text>

        {!showManual ? (
          <>
            <View style={styles.searchRow}>
              <Ionicons name="search" size={18} color={colors.text.tertiary} />
              <TextInput
                style={styles.searchInput}
                value={query}
                onChangeText={handleQueryChange}
                placeholder="Search foods..."
                placeholderTextColor={colors.input.placeholder}
                autoFocus
                returnKeyType="search"
              />
            </View>

            <FlatList
              data={results}
              keyExtractor={(item) => item.id}
              keyboardShouldPersistTaps="handled"
              style={styles.resultsList}
              renderItem={({ item }) => (
                <Pressable style={styles.resultRow} onPress={() => handleSelectResult(item)}>
                  <View style={styles.resultLeft}>
                    <Text style={styles.resultName} numberOfLines={1}>{item.name}</Text>
                    <Text style={styles.resultMacros}>
                      {item.calPer100g} kcal · P{item.proteinPer100g} C{item.carbsPer100g} F{item.fatPer100g} /100g
                    </Text>
                  </View>
                  <View style={[
                    styles.sourceBadge,
                    item.source === 'off' ? styles.sourceBadgeOff : styles.sourceBadgeKg,
                  ]}>
                    <Text style={styles.sourceBadgeText}>
                      {item.source === 'off' ? 'OFF' : 'KG'}
                    </Text>
                  </View>
                </Pressable>
              )}
              ListEmptyComponent={
                query.trim().length >= 2 && !loading ? (
                  <Text style={styles.emptyText}>No results for "{query.trim()}"</Text>
                ) : null
              }
            />

            <Pressable
              style={styles.manualBtn}
              onPress={() => {
                setShowManual(true);
                setManualName(query.trim());
              }}
            >
              <Ionicons name="create-outline" size={18} color={colors.accent.blue} />
              <Text style={styles.manualBtnText}>Manual Entry</Text>
            </Pressable>
          </>
        ) : (
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          >
            <View style={styles.manualForm}>
              <ManualField label="Name" value={manualName} onChange={setManualName} colors={colors} />
              <ManualField label="Grams" value={manualGrams} onChange={setManualGrams} numeric colors={colors} />
              <ManualField label="Calories" value={manualCal} onChange={setManualCal} numeric colors={colors} />
              <ManualField label="Protein (g)" value={manualProtein} onChange={setManualProtein} numeric colors={colors} />
              <ManualField label="Carbs (g)" value={manualCarbs} onChange={setManualCarbs} numeric colors={colors} />
              <ManualField label="Fat (g)" value={manualFat} onChange={setManualFat} numeric colors={colors} />

              <View style={styles.manualActions}>
                <Pressable style={styles.manualCancelBtn} onPress={() => setShowManual(false)}>
                  <Text style={styles.manualCancelText}>Back</Text>
                </Pressable>
                <Pressable style={styles.manualAddBtn} onPress={handleManualAdd}>
                  <Text style={styles.manualAddText}>Add</Text>
                </Pressable>
              </View>
            </View>
          </KeyboardAvoidingView>
        )}
      </BottomSheetView>
    </BottomSheet>
  );
}

// ---------------------------------------------------------------------------
// Manual entry field helper
// ---------------------------------------------------------------------------

function ManualField({
  label,
  value,
  onChange,
  numeric,
  colors,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  numeric?: boolean;
  colors: ThemeColors;
}) {
  return (
    <View style={fieldStyles.fieldRow}>
      <Text style={[fieldStyles.fieldLabel, { color: colors.text.secondary }]}>{label}</Text>
      <TextInput
        style={[fieldStyles.fieldInput, {
          color: colors.text.primary,
          backgroundColor: colors.input.background,
          borderColor: colors.input.border,
        }]}
        value={value}
        onChangeText={onChange}
        keyboardType={numeric ? 'decimal-pad' : 'default'}
        returnKeyType="next"
        placeholderTextColor={colors.input.placeholder}
      />
    </View>
  );
}

const fieldStyles = StyleSheet.create({
  fieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  fieldLabel: {
    width: 90,
    fontSize: 14,
    fontWeight: '500',
  },
  fieldInput: {
    flex: 1,
    fontSize: 16,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderWidth: 1,
  },
});

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    content: { paddingHorizontal: 20, paddingBottom: 24, flex: 1 },
    title: { fontSize: 18, fontWeight: '700', color: colors.text.primary, marginBottom: 12 },

    searchRow: {
      flexDirection: 'row', alignItems: 'center', gap: 8,
      backgroundColor: colors.input.background, borderRadius: 12, paddingHorizontal: 12, paddingVertical: 10,
      marginBottom: 8,
    },
    searchInput: { flex: 1, fontSize: 16, color: colors.text.primary },

    resultsList: { flex: 1, maxHeight: 300 },
    resultRow: {
      flexDirection: 'row', alignItems: 'center',
      paddingVertical: 12, paddingHorizontal: 8,
      borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
    },
    resultLeft: { flex: 1, marginRight: 8 },
    resultName: { fontSize: 15, fontWeight: '600', color: colors.text.primary },
    resultMacros: { fontSize: 12, color: colors.text.tertiary, marginTop: 2 },
    sourceBadge: { borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
    sourceBadgeKg: { backgroundColor: colors.accentTint.green },
    sourceBadgeOff: { backgroundColor: colors.accentTint.blue },
    sourceBadgeText: { fontSize: 10, fontWeight: '700', color: colors.text.secondary },

    emptyText: { textAlign: 'center', color: colors.text.tertiary, paddingVertical: 20, fontSize: 14 },

    manualBtn: {
      flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
      paddingVertical: 14, marginTop: 8,
      borderRadius: 12, borderWidth: 1, borderColor: colors.accent.blue, backgroundColor: colors.accentTint.blue,
    },
    manualBtnText: { fontSize: 15, fontWeight: '600', color: colors.accent.blue },

    manualForm: { paddingTop: 8 },

    manualActions: {
      flexDirection: 'row', gap: 12, marginTop: 16,
    },
    manualCancelBtn: {
      flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center',
      backgroundColor: colors.background.surface,
    },
    manualCancelText: { fontSize: 15, fontWeight: '600', color: colors.text.tertiary },
    manualAddBtn: {
      flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center',
      backgroundColor: colors.accent.green,
    },
    manualAddText: { fontSize: 15, fontWeight: '700', color: colors.text.inverse },
  });
}
