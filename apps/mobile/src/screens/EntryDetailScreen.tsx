/**
 * EntryDetailScreen -- view and edit a logged food entry's dishes and ingredients.
 *
 * Read mode: photo, dish names, ingredients with weights and nutrition, macro totals.
 * Edit mode: ServingSizeSelector for portions, IngredientSearchSheet for adding,
 *   undo/redo via command pattern, photo viewer with pinch-to-zoom, re-scan button.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Pressable,
  TextInput,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { opsqlite } from '../../db/client';
import type { RootStackParamList } from '../types';
import { addFavourite, isFavourited } from '../services/favourites';
import { useFoodLogStore } from '../store/useFoodLogStore';
import {
  recalculateEntryTotals,
  type IngredientUpdate,
} from '../services/entryEditor/entryEditorService';
import {
  ChangeWeightCommand,
  AddIngredientCommand,
  RemoveIngredientCommand,
  RenameIngredientCommand,
  RenameDishCommand,
  type EntrySnapshot,
} from '../services/entryEditor/editSessionManager';
import { useEditSession } from '../hooks/useEditSession';
import { ServingSizeSelector, IngredientSearchSheet, PhotoViewer } from '../components/edit';
import { geminiNanoService } from '../services/vlm/geminiNanoService';

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
}

interface DetailDish {
  id: string;
  name: string;
  cuisine: string | null;
  portionScale: number;
  ingredients: DetailIngredient[];
}

interface EntryDetail {
  id: string;
  mealType: string;
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  notes: string | null;
  createdAt: string;
  photoUri: string | null;
  photos: Array<{ uri: string }>;
  dishes: DetailDish[];
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

function loadEntry(entryId: string): EntryDetail | null {
  const entryRows = opsqlite.executeSync(
    `SELECT id, meal_type, total_calories, total_protein, total_carbs, total_fat, notes, created_at
     FROM food_entries WHERE id = ?`,
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  if (entryRows.length === 0) return null;
  const row = entryRows[0];

  // Load ALL photos (no LIMIT 1)
  const photoRows = opsqlite.executeSync(
    'SELECT uri FROM photos WHERE entry_id = ?',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const photos = photoRows.map((p) => ({ uri: p.uri as string }));

  const dishRows = opsqlite.executeSync(
    'SELECT id, name, cuisine, portion_scale FROM scanned_dishes WHERE entry_id = ? ORDER BY created_at',
    [entryId],
  ).rows as Array<Record<string, unknown>>;

  const dishes: DetailDish[] = dishRows.map((d) => {
    const dishId = d.id as string;
    const ingRows = opsqlite.executeSync(
      `SELECT id, name, amount_g, calories, protein, carbs, fat, fiber, sugar
       FROM ingredients WHERE dish_id = ? ORDER BY created_at`,
      [dishId],
    ).rows as Array<Record<string, unknown>>;

    return {
      id: dishId,
      name: d.name as string,
      cuisine: (d.cuisine as string) ?? null,
      portionScale: (d.portion_scale as number) ?? 1,
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
      })),
    };
  });

  return {
    id: row.id as string,
    mealType: row.meal_type as string,
    totalCalories: (row.total_calories as number) ?? 0,
    totalProtein: (row.total_protein as number) ?? 0,
    totalCarbs: (row.total_carbs as number) ?? 0,
    totalFat: (row.total_fat as number) ?? 0,
    notes: (row.notes as string) ?? null,
    createdAt: row.created_at as string,
    photoUri: photos.length > 0 ? photos[0].uri : null,
    photos,
    dishes,
  };
}

/** Convert EntryDetail to EntrySnapshot for undo/redo. */
function toSnapshot(entry: EntryDetail): EntrySnapshot {
  return {
    id: entry.id,
    mealType: entry.mealType,
    totalCalories: entry.totalCalories,
    totalProtein: entry.totalProtein,
    totalCarbs: entry.totalCarbs,
    totalFat: entry.totalFat,
    notes: entry.notes,
    createdAt: entry.createdAt,
    photoUri: entry.photoUri,
    photos: entry.photos,
    dishes: entry.dishes.map((d) => ({
      id: d.id,
      name: d.name,
      cuisine: d.cuisine,
      portionScale: d.portionScale,
      ingredients: d.ingredients.map((i) => ({
        id: i.id,
        name: i.name,
        amountG: i.amountG,
        calories: i.calories,
        protein: i.protein,
        carbs: i.carbs,
        fat: i.fat,
        fiber: i.fiber,
        sugar: i.sugar,
      })),
    })),
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function EntryDetailScreen() {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const route = useRoute<RouteProp<RootStackParamList, 'EntryDetail'>>();
  const { deleteEntry, loadTodayEntries } = useFoodLogStore();
  const [entry, setEntry] = useState<EntryDetail | null>(null);
  const [alreadyFaved, setAlreadyFaved] = useState(false);
  const [editing, setEditing] = useState(false);
  const [photoViewerVisible, setPhotoViewerVisible] = useState(false);
  const [ingredientSearchVisible, setIngredientSearchVisible] = useState(false);
  const [ingredientSearchDishId, setIngredientSearchDishId] = useState('');
  const [geminiAvailable, setGeminiAvailable] = useState(false);

  const editSession = useEditSession();

  const reload = useCallback(() => {
    const loaded = loadEntry(route.params.entryId);
    setEntry(loaded);
    if (loaded && loaded.dishes.length > 0) {
      setAlreadyFaved(isFavourited(loaded.dishes.map((d) => d.name).join(', ')));
    }
  }, [route.params.entryId]);

  useEffect(() => {
    reload();
  }, [reload]);

  // Check Gemini Nano availability
  useEffect(() => {
    geminiNanoService.isAvailable().then(setGeminiAvailable).catch(() => setGeminiAvailable(false));
  }, []);

  // -- Enter edit mode --
  const handleStartEditing = useCallback(() => {
    if (!entry) return;
    editSession.initSession(toSnapshot(entry));
    setEditing(true);
  }, [entry, editSession]);

  // -- Save --
  const handleSave = useCallback(() => {
    if (!entry) return;
    recalculateEntryTotals(entry.id);
    loadTodayEntries();
    editSession.clearSession();
    reload();
    setEditing(false);
  }, [entry, reload, loadTodayEntries, editSession]);

  // -- Cancel --
  const handleCancel = useCallback(() => {
    editSession.reset();
    editSession.clearSession();
    reload();
    setEditing(false);
  }, [editSession, reload]);

  // -- Weight change via command --
  const handleWeightChange = useCallback((ingId: string, oldAmountG: number, newAmountG: number) => {
    if (!entry) return;
    editSession.executeCommand(
      new ChangeWeightCommand(ingId, entry.id, oldAmountG, newAmountG),
    );
    reload();
  }, [entry, editSession, reload]);

  // -- Remove ingredient via command --
  const handleRemoveIngredient = useCallback((ing: DetailIngredient, dishId: string) => {
    if (!entry) return;
    Alert.alert('Remove Ingredient', 'Remove this ingredient?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Remove',
        style: 'destructive',
        onPress: () => {
          const savedData: IngredientUpdate = {
            entryId: entry.id,
            dishId,
            name: ing.name,
            amountG: ing.amountG,
            calories: ing.calories,
            protein: ing.protein,
            carbs: ing.carbs,
            fat: ing.fat,
            fiber: ing.fiber,
          };
          editSession.executeCommand(
            new RemoveIngredientCommand(ing.id, savedData),
          );
          reload();
        },
      },
    ]);
  }, [entry, editSession, reload]);

  // -- Add ingredient via search sheet --
  const handleOpenIngredientSearch = useCallback((dishId: string) => {
    setIngredientSearchDishId(dishId);
    setIngredientSearchVisible(true);
  }, []);

  const handleAddIngredient = useCallback((data: IngredientUpdate) => {
    editSession.executeCommand(new AddIngredientCommand(data));
    reload();
  }, [editSession, reload]);

  // -- Rename ingredient via command --
  const handleNameChange = useCallback((ingId: string, oldName: string, newName: string) => {
    if (newName.trim() && newName.trim() !== oldName) {
      editSession.executeCommand(
        new RenameIngredientCommand(ingId, oldName, newName.trim()),
      );
      reload();
    }
  }, [editSession, reload]);

  // -- Rename dish via command --
  const handleDishNameChange = useCallback((dishId: string, oldName: string, newName: string) => {
    if (newName.trim() && newName.trim() !== oldName) {
      editSession.executeCommand(
        new RenameDishCommand(dishId, oldName, newName.trim()),
      );
      reload();
    }
  }, [editSession, reload]);

  // -- Undo/Redo handlers --
  const handleUndo = useCallback(() => {
    editSession.undo();
    reload();
  }, [editSession, reload]);

  const handleRedo = useCallback(() => {
    editSession.redo();
    reload();
  }, [editSession, reload]);

  const handleReset = useCallback(() => {
    editSession.reset();
    reload();
  }, [editSession, reload]);

  if (!entry) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  const time = new Date(entry.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const mealLabel = entry.mealType.charAt(0).toUpperCase() + entry.mealType.slice(1);
  const hasPhotos = entry.photos.length > 0;

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Photo -- tappable to open PhotoViewer */}
        {entry.photoUri && (
          <Pressable onPress={() => setPhotoViewerVisible(true)}>
            <Image source={{ uri: entry.photoUri }} style={styles.photo} resizeMode="cover" />
            {entry.photos.length > 1 && (
              <View style={styles.photoCountBadge}>
                <Ionicons name="images-outline" size={14} color="#FFF" />
                <Text style={styles.photoCountText}>{entry.photos.length}</Text>
              </View>
            )}
          </Pressable>
        )}

        {/* Header */}
        <View style={styles.header}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10, flex: 1 }}>
            <View style={styles.mealBadge}>
              <Text style={styles.mealBadgeText}>{mealLabel}</Text>
            </View>
            <Text style={styles.timeText}>{time}</Text>
          </View>

          {/* Edit / Save + Cancel */}
          {entry.dishes.length > 0 && (
            <View style={{ flexDirection: 'row', gap: 8 }}>
              {editing && (
                <Pressable style={styles.cancelBtn} onPress={handleCancel}>
                  <Text style={styles.cancelBtnText}>Cancel</Text>
                </Pressable>
              )}
              <Pressable
                style={editing ? styles.saveBtn : styles.editBtn}
                onPress={editing ? handleSave : handleStartEditing}
              >
                <Ionicons
                  name={editing ? 'checkmark' : 'pencil'}
                  size={16}
                  color={editing ? '#FFF' : '#3B82F6'}
                />
                <Text style={editing ? styles.saveBtnText : styles.editBtnText}>
                  {editing ? 'Save' : 'Edit'}
                </Text>
              </Pressable>
            </View>
          )}
        </View>

        {/* Totals */}
        <View style={styles.totalsCard}>
          <View style={styles.totalMain}>
            <Text style={styles.totalCalNum}>{Math.round(entry.totalCalories)}</Text>
            <Text style={styles.totalCalLabel}>kcal</Text>
          </View>
          <View style={styles.totalMacros}>
            <MacroPill value={entry.totalProtein} label="P" color="#3B82F6" />
            <MacroPill value={entry.totalCarbs} label="C" color="#D97706" />
            <MacroPill value={entry.totalFat} label="F" color="#16A34A" />
          </View>
        </View>

        {/* Micronutrient panel */}
        {!editing && (() => {
          const allIngs = entry.dishes.flatMap((d) => d.ingredients);
          const totalFiber = allIngs.reduce((s, i) => s + i.fiber, 0);
          const totalSugar = allIngs.reduce((s, i) => s + i.sugar, 0);
          const hasMicros = totalFiber > 0 || totalSugar > 0;
          if (!hasMicros) return null;
          return (
            <View style={styles.microCard}>
              <Text style={styles.microTitle}>Additional Nutrients</Text>
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
          );
        })()}

        {/* Save to favourites */}
        {!editing && entry.dishes.length > 0 && !alreadyFaved && (
          <Pressable
            style={styles.favBtn}
            onPress={() => {
              const name = entry.dishes.map((d) => d.name).join(', ');
              addFavourite({
                name,
                totalCalories: entry.totalCalories,
                totalProtein: entry.totalProtein,
                totalCarbs: entry.totalCarbs,
                totalFat: entry.totalFat,
              });
              setAlreadyFaved(true);
              Alert.alert('Saved', `"${name}" added to favourites.`);
            }}
          >
            <Text style={styles.favBtnText}>Save to Favourites</Text>
          </Pressable>
        )}
        {!editing && alreadyFaved && (
          <View style={styles.favedBadge}>
            <Text style={styles.favedBadgeText}>In your favourites</Text>
          </View>
        )}

        {/* Dishes */}
        {entry.dishes.map((dish) => (
          <View key={dish.id} style={styles.dishCard}>
            <View style={styles.dishHeader}>
              {editing ? (
                <EditableText
                  value={dish.name}
                  onSubmit={(v) => handleDishNameChange(dish.id, dish.name, v)}
                  style={styles.dishName}
                />
              ) : (
                <Text style={styles.dishName}>{dish.name}</Text>
              )}
              {dish.cuisine && (
                <View style={styles.cuisinePill}>
                  <Text style={styles.cuisineText}>{dish.cuisine}</Text>
                </View>
              )}
              {!editing && dish.portionScale !== 1 && (
                <Text style={styles.scaleText}>{dish.portionScale}x</Text>
              )}
            </View>

            {dish.ingredients.map((ing) => (
              <View key={ing.id} style={styles.ingRow}>
                {editing ? (
                  <>
                    <View style={styles.ingLeft}>
                      <EditableText
                        value={ing.name}
                        onSubmit={(v) => handleNameChange(ing.id, ing.name, v)}
                        style={styles.ingName}
                      />
                      <Text style={styles.ingCal}>{Math.round(ing.calories)} kcal</Text>
                    </View>
                    <ServingSizeSelector
                      ingredientId={ing.id}
                      ingredientName={ing.name}
                      currentAmountG={ing.amountG}
                      onWeightChange={(grams) => handleWeightChange(ing.id, ing.amountG, grams)}
                    />
                    <Pressable
                      style={styles.removeBtn}
                      onPress={() => handleRemoveIngredient(ing, dish.id)}
                    >
                      <Ionicons name="close-circle" size={20} color="#EF4444" />
                    </Pressable>
                  </>
                ) : (
                  <>
                    <View style={styles.ingLeft}>
                      <Text style={styles.ingName}>{ing.name}</Text>
                      <Text style={styles.ingCal}>
                        {Math.round(ing.calories)} kcal
                        {ing.protein > 0 || ing.carbs > 0 || ing.fat > 0
                          ? `  ·  P${Math.round(ing.protein)} C${Math.round(ing.carbs)} F${Math.round(ing.fat)}`
                          : ''}
                      </Text>
                    </View>
                    <View style={styles.ingWeightChip}>
                      <Text style={styles.ingWeightText}>{Math.round(ing.amountG)}g</Text>
                    </View>
                  </>
                )}
              </View>
            ))}

            {editing && (
              <Pressable
                style={styles.addIngBtn}
                onPress={() => handleOpenIngredientSearch(dish.id)}
              >
                <Ionicons name="add-circle-outline" size={18} color="#16A34A" />
                <Text style={styles.addIngText}>Add Ingredient</Text>
              </Pressable>
            )}

            {!editing && dish.ingredients.length === 0 && (
              <Text style={styles.noIngText}>No ingredients recorded</Text>
            )}
          </View>
        ))}

        {/* Re-scan with Gemini Nano button */}
        {editing && hasPhotos && geminiAvailable && (
          <Pressable
            style={styles.rescanBtn}
            onPress={() => navigation.navigate('ReidentifyMerge', { entryId: entry.id })}
          >
            <Ionicons name="sparkles-outline" size={18} color="#7C3AED" />
            <Text style={styles.rescanBtnText}>Re-scan with Gemini Nano</Text>
          </Pressable>
        )}

        {entry.dishes.length === 0 && entry.notes && (
          <View style={styles.notesCard}>
            <Text style={styles.notesText}>{entry.notes}</Text>
          </View>
        )}

        {/* Delete */}
        {!editing && (
          <Pressable
            style={styles.deleteBtn}
            onPress={() => {
              Alert.alert('Delete Meal', 'Are you sure? This cannot be undone.', [
                { text: 'Cancel', style: 'cancel' },
                {
                  text: 'Delete',
                  style: 'destructive',
                  onPress: async () => {
                    await deleteEntry(entry.id);
                    if (navigation.canGoBack()) navigation.goBack();
                  },
                },
              ]);
            }}
          >
            <Text style={styles.deleteBtnText}>Delete Meal</Text>
          </Pressable>
        )}

        <View style={{ height: editing ? 80 : 40 }} />
      </ScrollView>

      {/* Undo/Redo/Reset bar -- floating at bottom in edit mode */}
      {editing && (
        <View style={styles.undoRedoBar}>
          <Pressable
            style={[styles.undoRedoBtn, !editSession.canUndo && styles.undoRedoBtnDisabled]}
            onPress={handleUndo}
            disabled={!editSession.canUndo}
          >
            <Ionicons name="arrow-undo" size={18} color={editSession.canUndo ? '#3B82F6' : '#D1D5DB'} />
            <Text style={[styles.undoRedoText, !editSession.canUndo && styles.undoRedoTextDisabled]}>Undo</Text>
          </Pressable>

          <Pressable
            style={[styles.undoRedoBtn, !editSession.canRedo && styles.undoRedoBtnDisabled]}
            onPress={handleRedo}
            disabled={!editSession.canRedo}
          >
            <Ionicons name="arrow-redo" size={18} color={editSession.canRedo ? '#3B82F6' : '#D1D5DB'} />
            <Text style={[styles.undoRedoText, !editSession.canRedo && styles.undoRedoTextDisabled]}>Redo</Text>
          </Pressable>

          <Pressable
            style={[styles.undoRedoBtn, !editSession.canUndo && styles.undoRedoBtnDisabled]}
            onPress={handleReset}
            disabled={!editSession.canUndo}
          >
            <Ionicons name="refresh" size={18} color={editSession.canUndo ? '#EF4444' : '#D1D5DB'} />
            <Text style={[styles.undoRedoText, !editSession.canUndo && styles.undoRedoTextDisabled, editSession.canUndo && { color: '#EF4444' }]}>Reset</Text>
          </Pressable>
        </View>
      )}

      {/* PhotoViewer modal */}
      <PhotoViewer
        photos={entry.photos}
        initialIndex={0}
        visible={photoViewerVisible}
        onClose={() => setPhotoViewerVisible(false)}
      />

      {/* Ingredient search bottom sheet */}
      <IngredientSearchSheet
        visible={ingredientSearchVisible}
        entryId={entry.id}
        dishId={ingredientSearchDishId}
        onAdd={handleAddIngredient}
        onClose={() => setIngredientSearchVisible(false)}
      />
    </View>
  );
}

// ---------------------------------------------------------------------------
// Edit subcomponents
// ---------------------------------------------------------------------------

function EditableText({
  value,
  onSubmit,
  style,
}: {
  value: string;
  onSubmit: (v: string) => void;
  style: object;
}) {
  const [text, setText] = useState(value);
  return (
    <TextInput
      style={[style, styles.editableText]}
      value={text}
      onChangeText={setText}
      onBlur={() => onSubmit(text)}
      onSubmitEditing={() => onSubmit(text)}
      returnKeyType="done"
      selectTextOnFocus
    />
  );
}

function MacroPill({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <View style={styles.macroPill}>
      <Text style={[styles.macroPillNum, { color }]}>{Math.round(value)}g</Text>
      <Text style={[styles.macroPillLabel, { color }]}> {label}</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  scrollContent: { paddingBottom: 20 },
  loadingText: { marginTop: 100, textAlign: 'center', color: '#6B7280', fontSize: 16 },

  photo: { width: '100%', height: 240 },
  photoCountBadge: {
    position: 'absolute', bottom: 12, right: 12, flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 12, paddingHorizontal: 8, paddingVertical: 4,
  },
  photoCountText: { fontSize: 12, fontWeight: '600', color: '#FFF' },

  header: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 14, backgroundColor: '#FFF',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  mealBadge: {
    backgroundColor: '#16A34A', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 4,
  },
  mealBadgeText: { fontSize: 13, fontWeight: '600', color: '#FFF' },
  timeText: { fontSize: 14, color: '#6B7280' },

  // Edit/Save/Cancel buttons
  editBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: '#EFF6FF', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
    borderWidth: 1, borderColor: '#BFDBFE',
  },
  editBtnText: { fontSize: 13, fontWeight: '600', color: '#3B82F6' },
  saveBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: '#16A34A', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
  },
  saveBtnText: { fontSize: 13, fontWeight: '600', color: '#FFF' },
  cancelBtn: {
    borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
    backgroundColor: '#F3F4F6', borderWidth: 1, borderColor: '#E5E7EB',
  },
  cancelBtnText: { fontSize: 13, fontWeight: '600', color: '#6B7280' },

  totalsCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  totalMain: { flexDirection: 'row', alignItems: 'baseline', gap: 4 },
  totalCalNum: { fontSize: 28, fontWeight: '800', color: '#111827' },
  totalCalLabel: { fontSize: 14, color: '#6B7280' },
  totalMacros: { flexDirection: 'row', gap: 6 },
  macroPill: {
    flexDirection: 'row', backgroundColor: '#F9FAFB', borderRadius: 8,
    paddingHorizontal: 8, paddingVertical: 4, alignItems: 'center',
  },
  macroPillNum: { fontSize: 13, fontWeight: '700' },
  macroPillLabel: { fontSize: 11, fontWeight: '600' },

  microCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  microTitle: { fontSize: 14, fontWeight: '700', color: '#374151', marginBottom: 8 },
  microRow: {
    flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  microLabel: { fontSize: 14, color: '#6B7280' },
  microValue: { fontSize: 14, fontWeight: '600', color: '#111827' },

  favBtn: {
    backgroundColor: '#FEF3C7', borderRadius: 12, paddingVertical: 12, marginHorizontal: 16,
    marginTop: 12, alignItems: 'center', borderWidth: 1, borderColor: '#FDE68A',
  },
  favBtnText: { fontSize: 15, fontWeight: '600', color: '#92400E' },
  favedBadge: {
    backgroundColor: '#F0FDF4', borderRadius: 12, paddingVertical: 10, marginHorizontal: 16,
    marginTop: 12, alignItems: 'center', borderWidth: 1, borderColor: '#BBF7D0',
  },
  favedBadgeText: { fontSize: 14, fontWeight: '500', color: '#16A34A' },

  dishCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  dishHeader: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: 16, paddingVertical: 12, backgroundColor: '#F9FAFB',
  },
  dishName: { fontSize: 16, fontWeight: '700', color: '#111827', flex: 1 },
  cuisinePill: { backgroundColor: '#DCFCE7', borderRadius: 20, paddingHorizontal: 8, paddingVertical: 2 },
  cuisineText: { fontSize: 11, fontWeight: '500', color: '#16A34A' },
  scaleText: { fontSize: 13, fontWeight: '600', color: '#6B7280' },

  ingRow: {
    flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F9FAFB',
  },
  ingLeft: { flex: 1 },
  ingName: { fontSize: 14, fontWeight: '500', color: '#111827' },
  ingCal: { fontSize: 12, color: '#9CA3AF', marginTop: 1 },
  ingWeightChip: {
    backgroundColor: '#F0FDF4', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4,
    borderWidth: 1, borderColor: '#BBF7D0',
  },
  ingWeightText: { fontSize: 13, fontWeight: '600', color: '#16A34A' },
  noIngText: { textAlign: 'center', padding: 16, color: '#9CA3AF', fontSize: 13 },

  // Edit mode styles
  editableText: {
    backgroundColor: '#F3F4F6', borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4,
    borderWidth: 1, borderColor: '#E5E7EB',
  },
  removeBtn: { marginLeft: 8, padding: 4 },
  addIngBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    paddingVertical: 12, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: '#F3F4F6',
  },
  addIngText: { fontSize: 13, fontWeight: '600', color: '#16A34A' },

  // Re-scan button
  rescanBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    marginHorizontal: 16, marginTop: 12, paddingVertical: 14, borderRadius: 12,
    backgroundColor: '#F5F3FF', borderWidth: 1, borderColor: '#DDD6FE',
  },
  rescanBtnText: { fontSize: 15, fontWeight: '600', color: '#7C3AED' },

  // Undo/Redo bar
  undoRedoBar: {
    position: 'absolute', bottom: 20, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'space-around',
    backgroundColor: '#FFF', borderRadius: 16, paddingVertical: 12, paddingHorizontal: 8,
    shadowColor: '#000', shadowOffset: { width: 0, height: -2 }, shadowOpacity: 0.1,
    shadowRadius: 12, elevation: 8,
  },
  undoRedoBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: 16, paddingVertical: 8, borderRadius: 8,
  },
  undoRedoBtnDisabled: { opacity: 0.4 },
  undoRedoText: { fontSize: 13, fontWeight: '600', color: '#3B82F6' },
  undoRedoTextDisabled: { color: '#D1D5DB' },

  notesCard: {
    backgroundColor: '#FFF', marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
  },
  notesText: { fontSize: 14, color: '#374151', lineHeight: 20 },
  deleteBtn: {
    marginHorizontal: 16, marginTop: 24, paddingVertical: 14, borderRadius: 12,
    backgroundColor: '#FEF2F2', alignItems: 'center', borderWidth: 1, borderColor: '#FECACA',
  },
  deleteBtnText: { fontSize: 15, fontWeight: '600', color: '#DC2626' },
});
