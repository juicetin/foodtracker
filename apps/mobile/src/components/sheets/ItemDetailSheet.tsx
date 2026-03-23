/**
 * ItemDetailSheet -- read-only bottom sheet showing full entry detail.
 *
 * Opened on tap of a food item in DiaryHomeScreen. Shows dish name,
 * total macros, ingredient list with per-ingredient macros, and
 * expandable sections for micronutrients, nutrition source, and photo.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Image } from 'react-native';
import BottomSheet, { BottomSheetScrollView, BottomSheetBackdrop } from '@gorhom/bottom-sheet';
import { Ionicons } from '@expo/vector-icons';
import { opsqlite } from '../../../db/client';
import { isFavourited, addFavourite, removeFavourite } from '../../services/favourites';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DetailIngredient {
  id: string;
  name: string;
  amountG: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sugar: number;
  databaseSource: string | null;
}

interface DetailDish {
  id: string;
  name: string;
  cuisine: string | null;
  ingredients: DetailIngredient[];
}

interface EntryDetail {
  id: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  createdAt: string;
  photoUri: string | null;
  dishes: DetailDish[];
}

interface ItemDetailSheetProps {
  entryId: string | null;
  onDismiss: () => void;
  onEdit: (entryId: string) => void;
  onDelete: (entryId: string) => void;
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

function loadEntryDetail(entryId: string): EntryDetail | null {
  const entryRows = opsqlite.executeSync(
    `SELECT id, total_calories, total_protein, total_carbs, total_fat, created_at
     FROM food_entries WHERE id = ? AND is_deleted = 0`,
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  if (entryRows.length === 0) return null;
  const row = entryRows[0];

  const photoRows = opsqlite.executeSync(
    'SELECT uri FROM photos WHERE entry_id = ? LIMIT 1',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const dishRows = opsqlite.executeSync(
    'SELECT id, name, cuisine FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const dishes: DetailDish[] = dishRows.map((d) => {
    const dishId = d.id as string;
    const ingRows = opsqlite.executeSync(
      `SELECT id, name, amount_g, calories, protein, carbs, fat, fiber, sugar, database_source
       FROM ingredients WHERE dish_id = ? ORDER BY created_at`,
      [dishId],
    ).rows as Array<Record<string, unknown>>;

    return {
      id: dishId,
      name: d.name as string,
      cuisine: (d.cuisine as string) ?? null,
      ingredients: ingRows.map((i) => ({
        id: i.id as string,
        name: i.name as string,
        amountG: (i.amount_g as number) ?? 0,
        calories: (i.calories as number) ?? 0,
        protein: (i.protein as number) ?? 0,
        carbs: (i.carbs as number) ?? 0,
        fat: (i.fat as number) ?? 0,
        fiber: (i.fiber as number) ?? 0,
        sugar: (i.sugar as number) ?? 0,
        databaseSource: (i.database_source as string) ?? null,
      })),
    };
  });

  return {
    id: row.id as string,
    totalCalories: (row.total_calories as number) ?? 0,
    totalProtein: (row.total_protein as number) ?? 0,
    totalCarbs: (row.total_carbs as number) ?? 0,
    totalFat: (row.total_fat as number) ?? 0,
    createdAt: row.created_at as string,
    photoUri: photoRows.length > 0 ? (photoRows[0].uri as string) : null,
    dishes,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ItemDetailSheet({ entryId, onDismiss, onEdit, onDelete }: ItemDetailSheetProps) {
  const bottomSheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ['50%', '90%'], []);

  const [entry, setEntry] = useState<EntryDetail | null>(null);
  const [isFaved, setIsFaved] = useState(false);
  const [microExpanded, setMicroExpanded] = useState(false);
  const [sourceExpanded, setSourceExpanded] = useState(false);
  const [photoExpanded, setPhotoExpanded] = useState(false);

  // Load entry when entryId changes
  useEffect(() => {
    if (entryId) {
      const loaded = loadEntryDetail(entryId);
      setEntry(loaded);
      if (loaded && loaded.dishes.length > 0) {
        setIsFaved(isFavourited(loaded.dishes.map((d) => d.name).join(', ')));
      }
      bottomSheetRef.current?.snapToIndex(0);
    } else {
      bottomSheetRef.current?.close();
    }
  }, [entryId]);

  const handleChange = useCallback(
    (index: number) => {
      if (index === -1) {
        onDismiss();
      }
    },
    [onDismiss],
  );

  const renderBackdrop = useCallback(
    (props: React.ComponentProps<typeof BottomSheetBackdrop>) => (
      <BottomSheetBackdrop {...props} appearsOnIndex={0} disappearsOnIndex={-1} pressBehavior="close" />
    ),
    [],
  );

  const handleToggleFavourite = useCallback(() => {
    if (!entry) return;
    const name = entry.dishes.map((d) => d.name).join(', ');
    if (isFaved) {
      // Find and remove the favourite by name
      const rows = opsqlite.executeSync(
        'SELECT id FROM favourite_meals WHERE name = ? LIMIT 1',
        [name],
      ).rows as Array<Record<string, unknown>>;
      if (rows.length > 0) {
        removeFavourite(rows[0].id as string);
      }
      setIsFaved(false);
    } else {
      addFavourite({
        name,
        totalCalories: entry.totalCalories,
        totalProtein: entry.totalProtein,
        totalCarbs: entry.totalCarbs,
        totalFat: entry.totalFat,
      });
      setIsFaved(true);
    }
  }, [entry, isFaved]);

  // Compute micronutrients
  const allIngredients = entry?.dishes.flatMap((d) => d.ingredients) ?? [];
  const totalFiber = allIngredients.reduce((s, i) => s + i.fiber, 0);
  const totalSugar = allIngredients.reduce((s, i) => s + i.sugar, 0);
  const hasMicros = totalFiber > 0 || totalSugar > 0;

  // Collect unique nutrition sources
  const sources = [...new Set(allIngredients.map((i) => i.databaseSource).filter(Boolean))];
  const hasSources = sources.length > 0;

  const dishName = entry?.dishes.map((d) => d.name).join(', ') ?? 'Food Item';
  const timeStr = entry ? new Date(entry.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

  return (
    <BottomSheet
      ref={bottomSheetRef}
      index={-1}
      snapPoints={snapPoints}
      enablePanDownToClose
      onChange={handleChange}
      backdropComponent={renderBackdrop}
      backgroundStyle={styles.sheetBackground}
      handleIndicatorStyle={styles.handleIndicator}
    >
      <BottomSheetScrollView contentContainerStyle={styles.content}>
        {entry && (
          <>
            {/* Header row */}
            <View style={styles.headerRow}>
              <Text style={styles.dishName} numberOfLines={2}>
                {dishName}
              </Text>
              <View style={styles.headerIcons}>
                <Pressable onPress={handleToggleFavourite} hitSlop={8}>
                  <Ionicons
                    name={isFaved ? 'heart' : 'heart-outline'}
                    size={22}
                    color={isFaved ? '#EF4444' : '#6B7280'}
                  />
                </Pressable>
                <Pressable onPress={() => onEdit(entry.id)} hitSlop={8}>
                  <Ionicons name="create-outline" size={22} color="#3B82F6" />
                </Pressable>
                <Pressable onPress={() => onDelete(entry.id)} hitSlop={8}>
                  <Ionicons name="trash-outline" size={22} color="#EF4444" />
                </Pressable>
              </View>
            </View>

            {/* Time logged */}
            <Text style={styles.timeText}>{timeStr}</Text>

            {/* Total macros row */}
            <View style={styles.macroRow}>
              <Text style={styles.totalCal}>{Math.round(entry.totalCalories)}</Text>
              <Text style={styles.totalCalUnit}>kcal</Text>
              <View style={styles.macroPills}>
                <View style={[styles.pill, { backgroundColor: '#EFF6FF' }]}>
                  <Text style={[styles.pillText, { color: '#3B82F6' }]}>P {Math.round(entry.totalProtein)}g</Text>
                </View>
                <View style={[styles.pill, { backgroundColor: '#FFFBEB' }]}>
                  <Text style={[styles.pillText, { color: '#D97706' }]}>C {Math.round(entry.totalCarbs)}g</Text>
                </View>
                <View style={[styles.pill, { backgroundColor: '#F0FDF4' }]}>
                  <Text style={[styles.pillText, { color: '#059669' }]}>F {Math.round(entry.totalFat)}g</Text>
                </View>
              </View>
            </View>

            {/* Ingredient list */}
            {entry.dishes.map((dish) => (
              <View key={dish.id} style={styles.dishSection}>
                {entry.dishes.length > 1 && (
                  <Text style={styles.dishSectionName}>{dish.name}</Text>
                )}
                {dish.ingredients.map((ing, idx) => (
                  <View key={ing.id}>
                    <View style={styles.ingRow}>
                      <View style={styles.ingLeft}>
                        <Text style={styles.ingName}>{ing.name}</Text>
                        <Text style={styles.ingAmount}>{Math.round(ing.amountG)}g</Text>
                      </View>
                      <Text style={styles.ingMacros}>
                        {Math.round(ing.calories)} cal | P{Math.round(ing.protein)} C{Math.round(ing.carbs)} F{Math.round(ing.fat)}
                      </Text>
                    </View>
                    {idx < dish.ingredients.length - 1 && <View style={styles.divider} />}
                  </View>
                ))}
              </View>
            ))}

            {/* + Add Ingredient button */}
            <Pressable style={styles.addIngredientBtn} onPress={() => onEdit(entry.id)}>
              <Ionicons name="add-circle-outline" size={18} color="#16A34A" />
              <Text style={styles.addIngredientText}>+ Add Ingredient</Text>
            </Pressable>

            {/* Expandable sections */}
            {hasMicros && (
              <View style={styles.expandableSection}>
                <Pressable style={styles.sectionHeader} onPress={() => setMicroExpanded(!microExpanded)}>
                  <Text style={styles.sectionHeaderText}>Micronutrients</Text>
                  <Ionicons
                    name={microExpanded ? 'chevron-up' : 'chevron-down'}
                    size={18}
                    color="#374151"
                  />
                </Pressable>
                {microExpanded && (
                  <View style={styles.sectionContent}>
                    {totalFiber > 0 && (
                      <View style={styles.microRow}>
                        <Text style={styles.microLabel}>Fiber</Text>
                        <Text style={styles.microValue}>{totalFiber.toFixed(1)}g</Text>
                      </View>
                    )}
                    {totalSugar > 0 && (
                      <View style={styles.microRow}>
                        <Text style={styles.microLabel}>Sugar</Text>
                        <Text style={styles.microValue}>{totalSugar.toFixed(1)}g</Text>
                      </View>
                    )}
                  </View>
                )}
              </View>
            )}

            {hasSources && (
              <View style={styles.expandableSection}>
                <Pressable style={styles.sectionHeader} onPress={() => setSourceExpanded(!sourceExpanded)}>
                  <Text style={styles.sectionHeaderText}>Nutrition Source</Text>
                  <Ionicons
                    name={sourceExpanded ? 'chevron-up' : 'chevron-down'}
                    size={18}
                    color="#374151"
                  />
                </Pressable>
                {sourceExpanded && (
                  <View style={styles.sectionContent}>
                    {sources.map((src) => (
                      <Text key={src} style={styles.sourceText}>{src}</Text>
                    ))}
                  </View>
                )}
              </View>
            )}

            {entry.photoUri && (
              <View style={styles.expandableSection}>
                <Pressable style={styles.sectionHeader} onPress={() => setPhotoExpanded(!photoExpanded)}>
                  <Text style={styles.sectionHeaderText}>View Photo</Text>
                  <Ionicons
                    name={photoExpanded ? 'chevron-up' : 'chevron-down'}
                    size={18}
                    color="#374151"
                  />
                </Pressable>
                {photoExpanded && (
                  <View style={styles.sectionContent}>
                    <Image
                      source={{ uri: entry.photoUri }}
                      style={styles.entryPhoto}
                      resizeMode="cover"
                    />
                  </View>
                )}
              </View>
            )}

            <View style={{ height: 40 }} />
          </>
        )}
      </BottomSheetScrollView>
    </BottomSheet>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  sheetBackground: {
    backgroundColor: '#FFFFFF',
  },
  handleIndicator: {
    backgroundColor: '#D1D5DB',
  },
  content: {
    paddingHorizontal: 16,
    paddingTop: 8,
  },

  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  dishName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
    flex: 1,
    marginRight: 12,
  },
  headerIcons: {
    flexDirection: 'row',
    gap: 16,
    alignItems: 'center',
  },

  timeText: {
    fontSize: 14,
    color: '#9CA3AF',
    marginBottom: 12,
  },

  macroRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 16,
    gap: 4,
  },
  totalCal: {
    fontSize: 32,
    fontWeight: '600',
    color: '#111827',
  },
  totalCalUnit: {
    fontSize: 14,
    color: '#6B7280',
    marginRight: 12,
  },
  macroPills: {
    flexDirection: 'row',
    gap: 6,
  },
  pill: {
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  pillText: {
    fontSize: 13,
    fontWeight: '600',
  },

  dishSection: {
    marginBottom: 8,
  },
  dishSectionName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },

  ingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  ingLeft: {
    flex: 1,
  },
  ingName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
  },
  ingAmount: {
    fontSize: 14,
    color: '#374151',
    marginTop: 2,
  },
  ingMacros: {
    fontSize: 14,
    color: '#9CA3AF',
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: '#E5E7EB',
  },

  addIngredientBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 14,
    marginTop: 4,
  },
  addIngredientText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#16A34A',
  },

  expandableSection: {
    marginTop: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#E5E7EB',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  sectionHeaderText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  sectionContent: {
    paddingBottom: 8,
  },

  microRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  microLabel: {
    fontSize: 14,
    color: '#6B7280',
  },
  microValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#111827',
  },

  sourceText: {
    fontSize: 14,
    color: '#6B7280',
    paddingVertical: 2,
  },

  entryPhoto: {
    width: '100%',
    height: 200,
    borderRadius: 12,
  },
});
