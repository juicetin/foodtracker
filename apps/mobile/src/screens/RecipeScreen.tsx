/**
 * RecipeScreen — create, import, and manage recipes.
 *
 * Two modes:
 * - List: shows saved recipes with photo, per-serving macros, search, one-tap re-log
 * - Builder: create/edit a recipe (manually or from URL import) with versioning
 */

import React, {useCallback, useState, useMemo} from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import {
  createRecipe,
  loadRecipes,
  loadRecipe,
  addRecipeIngredient,
  removeRecipeIngredient,
  updateRecipeIngredient,
  updateRecipeName,
  deleteRecipe,
  logRecipeAsEntry,
  searchRecipes,
  updateRecipeWithVersioning,
  type RecipeSummary,
  type RecipeDetail,
  type RecipeIngredientInput,
} from '../services/recipes/recipeService';
import { parseRecipeFromHtml } from '../services/recipes/recipeUrlParser';
import { autoDetectMealType } from '../services/detection/types';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { usePreferencesStore } from '../store/usePreferencesStore';
import type { UxMode } from '../types';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function RecipeScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const navigation = useNavigation();
  const { loadTodayEntries } = useFoodLogStore();
  const uxMode = usePreferencesStore((s) => s.uxMode);

  const [recipes, setRecipes] = useState<RecipeSummary[]>([]);
  const [activeRecipe, setActiveRecipe] = useState<RecipeDetail | null>(null);
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [importUrl, setImportUrl] = useState('');
  const [importing, setImporting] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // New ingredient form
  const [newIngName, setNewIngName] = useState('');
  const [newIngQty, setNewIngQty] = useState('');
  const [newIngCal, setNewIngCal] = useState('');

  // Builder servings
  const [builderServings, setBuilderServings] = useState('1');

  const refreshList = useCallback(() => {
    setRecipes(loadRecipes());
  }, []);

  const refreshActiveRecipe = useCallback(() => {
    if (activeRecipe) {
      const updated = loadRecipe(activeRecipe.id);
      setActiveRecipe(updated);
    }
  }, [activeRecipe]);

  useFocusEffect(useCallback(() => { refreshList(); }, [refreshList]));

  // Handle search filtering
  function getDisplayedRecipes(): RecipeSummary[] {
    const trimmed = searchQuery.trim();
    if (trimmed.length >= 2) {
      return searchRecipes(trimmed, 20);
    }
    return recipes;
  }

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  function handleCreateRecipe() {
    createRecipeWithPrompt();
  }

  function createRecipeWithPrompt() {
    const id = createRecipe({ name: 'New Recipe' });
    const recipe = loadRecipe(id);
    setActiveRecipe(recipe);
    if (recipe) setBuilderServings(String(recipe.servings || 1));
    refreshList();
  }

  function handleLogRecipe(recipe: RecipeSummary) {
    if (uxMode === 'zero-effort') {
      logRecipeAsEntry(recipe.id, autoDetectMealType());
      loadTodayEntries();
      refreshList();
      Alert.alert('Logged', `${recipe.name} added to diary.`);
    } else if (uxMode === 'confirm-only') {
      const servings = recipe.servings || 1;
      const calPerServing = Math.round(recipe.totalCalories / servings);
      const pPerServing = Math.round(recipe.totalProtein / servings);
      const cPerServing = Math.round(recipe.totalCarbs / servings);
      const fPerServing = Math.round(recipe.totalFat / servings);
      Alert.alert(
        'Log Recipe',
        `${recipe.name}\n${calPerServing} Cal · P${pPerServing} C${cPerServing} F${fPerServing} per serving\n\nLog 1 serving?`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Log',
            onPress: () => {
              logRecipeAsEntry(recipe.id, autoDetectMealType());
              loadTodayEntries();
              refreshList();
              Alert.alert('Logged', `${recipe.name} added to diary.`);
            },
          },
        ],
      );
    } else {
      // guided-edit: open builder
      const full = loadRecipe(recipe.id);
      setActiveRecipe(full);
      if (full) setBuilderServings(String(full.servings || 1));
    }
  }

  function handleDeleteRecipe(id: string) {
    Alert.alert('Delete Recipe', 'Are you sure?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          deleteRecipe(id);
          if (activeRecipe?.id === id) setActiveRecipe(null);
          refreshList();
        },
      },
    ]);
  }

  function handleAddIngredient() {
    if (!activeRecipe || !newIngName.trim()) return;
    const qty = parseFloat(newIngQty) || 100;
    const cal = parseFloat(newIngCal) || 0;

    addRecipeIngredient({
      recipeId: activeRecipe.id,
      name: newIngName.trim(),
      quantity: qty,
      unit: 'g',
      calories: cal,
      protein: 0,
      carbs: 0,
      fat: 0,
    });

    setNewIngName('');
    setNewIngQty('');
    setNewIngCal('');
    refreshActiveRecipe();
    refreshList();
  }

  function handleRemoveIngredient(ingId: string) {
    if (!activeRecipe) return;
    removeRecipeIngredient(ingId, activeRecipe.id);
    refreshActiveRecipe();
    refreshList();
  }

  function handleSaveWithVersioning() {
    if (!activeRecipe) return;

    const ingredients: RecipeIngredientInput[] = activeRecipe.ingredients.map((ing) => ({
      recipeId: activeRecipe.id,
      name: ing.name,
      quantity: ing.quantity,
      unit: ing.unit,
      calories: ing.calories,
      protein: ing.protein,
      carbs: ing.carbs,
      fat: ing.fat,
    }));

    Alert.alert(
      'Save Changes',
      'How would you like to save?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Update All',
          onPress: () => {
            updateRecipeWithVersioning(activeRecipe.id, ingredients, 'update-all');
            refreshActiveRecipe();
            refreshList();
            Alert.alert('Updated', 'Recipe and linked diary entries updated.');
          },
        },
        {
          text: 'Save as New',
          onPress: () => {
            const newId = updateRecipeWithVersioning(activeRecipe.id, ingredients, 'save-as-new');
            const newRecipe = loadRecipe(newId);
            setActiveRecipe(newRecipe);
            refreshList();
            Alert.alert('Saved', 'New recipe created from your edits.');
          },
        },
      ],
    );
  }

  async function handleImportUrl() {
    const url = importUrl.trim();
    if (!url) return;

    setImporting(true);
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'Tastimate/1.0 RecipeImport' },
      });
      if (!response.ok) {
        Alert.alert('Error', `Failed to fetch URL (${response.status}).`);
        setImporting(false);
        return;
      }

      const html = await response.text();
      const parsed = parseRecipeFromHtml(html);

      if (!parsed) {
        Alert.alert('No Recipe Found', 'Could not find a recipe on this page. The site may not use standard recipe markup.');
        setImporting(false);
        return;
      }

      // Create recipe and add ingredients
      const id = createRecipe({
        name: parsed.name,
        description: parsed.description,
      });

      for (const ingText of parsed.ingredients) {
        addRecipeIngredient({
          recipeId: id,
          name: ingText,
          quantity: 0,
          unit: 'serving',
          calories: 0,
          protein: 0,
          carbs: 0,
          fat: 0,
        });
      }

      const recipe = loadRecipe(id);
      setActiveRecipe(recipe);
      if (recipe) setBuilderServings(String(recipe.servings || 1));

      refreshList();
      setImportModalVisible(false);
      setImportUrl('');
      Alert.alert('Imported', `"${parsed.name}" with ${parsed.ingredients.length} ingredients.`);
    } catch (err) {
      Alert.alert('Error', 'Failed to import recipe. Check the URL and try again.');
    } finally {
      setImporting(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Builder view (editing a recipe)
  // ---------------------------------------------------------------------------

  if (activeRecipe) {
    const servings = parseInt(builderServings) || 1;
    const calPerServing = Math.round(activeRecipe.totalCalories / servings);
    const pPerServing = Math.round(activeRecipe.totalProtein / servings);
    const cPerServing = Math.round(activeRecipe.totalCarbs / servings);
    const fPerServing = Math.round(activeRecipe.totalFat / servings);

    return (
      <View style={styles.container}>
        <View style={styles.builderHeader}>
          <Pressable onPress={() => { setActiveRecipe(null); refreshList(); }}>
            <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
          </Pressable>
          <TextInput
            style={styles.builderTitleInput}
            defaultValue={activeRecipe.name}
            onEndEditing={(e) => {
              const name = e.nativeEvent.text.trim();
              if (name && name !== activeRecipe.name) {
                updateRecipeName(activeRecipe.id, name);
                refreshActiveRecipe();
                refreshList();
              }
            }}
            returnKeyType="done"
            selectTextOnFocus
          />
          <Pressable onPress={() => handleDeleteRecipe(activeRecipe.id)}>
            <Ionicons name="trash-outline" size={22} color={colors.accent.red} />
          </Pressable>
        </View>

        <ScrollView contentContainerStyle={styles.builderContent}>
          {/* Totals -- total and per-serving */}
          <View style={styles.totalsRow}>
            <Text style={styles.totalsBig}>{Math.round(activeRecipe.totalCalories)} kcal</Text>
            <Text style={styles.totalsSub}>
              P: {Math.round(activeRecipe.totalProtein)}g  C: {Math.round(activeRecipe.totalCarbs)}g  F: {Math.round(activeRecipe.totalFat)}g
            </Text>
            {servings > 1 && (
              <Text style={styles.perServingSub}>
                Per serving: {calPerServing} Cal · P{pPerServing} C{cPerServing} F{fPerServing}
              </Text>
            )}
          </View>

          {/* Servings input */}
          <View style={styles.servingsRow}>
            <Text style={styles.servingsLabel}>Servings:</Text>
            <TextInput
              style={styles.servingsInput}
              value={builderServings}
              onChangeText={setBuilderServings}
              keyboardType="numeric"
              selectTextOnFocus
            />
          </View>

          {/* Ingredients */}
          <Text style={styles.sectionLabel}>Ingredients ({activeRecipe.ingredients.length})</Text>

          {activeRecipe.ingredients.map((ing) => (
            <View key={ing.id} style={styles.ingRow}>
              <View style={styles.ingInfo}>
                <Text style={styles.ingName} numberOfLines={2}>{ing.name}</Text>
                {ing.quantity > 0 && (
                  <Text style={styles.ingMeta}>{ing.quantity}{ing.unit} · {Math.round(ing.calories)} kcal</Text>
                )}
              </View>
              <Pressable onPress={() => handleRemoveIngredient(ing.id)} style={styles.ingRemove}>
                <Ionicons name="close-circle" size={20} color={colors.accent.red} />
              </Pressable>
            </View>
          ))}

          {/* Add ingredient form */}
          <View style={styles.addIngForm}>
            <TextInput
              style={styles.addIngInput}
              value={newIngName}
              onChangeText={setNewIngName}
              placeholder="Ingredient name"
              placeholderTextColor="#9CA3AF"
            />
            <View style={styles.addIngRow}>
              <TextInput
                style={[styles.addIngInput, { flex: 1 }]}
                value={newIngQty}
                onChangeText={setNewIngQty}
                placeholder="Grams"
                placeholderTextColor="#9CA3AF"
                keyboardType="numeric"
              />
              <TextInput
                style={[styles.addIngInput, { flex: 1 }]}
                value={newIngCal}
                onChangeText={setNewIngCal}
                placeholder="Calories"
                placeholderTextColor="#9CA3AF"
                keyboardType="numeric"
              />
              <Pressable style={styles.addIngBtn} onPress={handleAddIngredient}>
                <Ionicons name="add" size={20} color={colors.text.inverse} />
              </Pressable>
            </View>
          </View>

          {/* Save with versioning */}
          <Pressable style={styles.versionBtn} onPress={handleSaveWithVersioning}>
            <Ionicons name="save-outline" size={18} color={colors.accent.purple} />
            <Text style={styles.versionBtnText}>Save Changes</Text>
          </Pressable>

          {/* Log recipe */}
          <Pressable
            style={styles.logBtn}
            onPress={() => {
              handleLogRecipe(activeRecipe);
              if (uxMode !== 'guided-edit') setActiveRecipe(null);
            }}
          >
            <Ionicons name="add-circle" size={20} color={colors.text.inverse} />
            <Text style={styles.logBtnText}>Log to Diary</Text>
          </Pressable>
        </ScrollView>
      </View>
    );
  }

  // ---------------------------------------------------------------------------
  // List view
  // ---------------------------------------------------------------------------

  const displayedRecipes = getDisplayedRecipes();

  return (
    <View style={styles.container}>
      <View style={styles.listHeader}>
        <Pressable onPress={() => navigation.canGoBack() && navigation.goBack()}>
          <Ionicons name="close" size={24} color={colors.text.primary} />
        </Pressable>
        <Text style={styles.listTitle}>Recipes</Text>
        <View style={{ width: 24 }} />
      </View>

      {/* Search bar */}
      <View style={styles.searchRow}>
        <View style={styles.searchInputContainer}>
          <Ionicons name="search" size={18} color={colors.text.tertiary} style={{ marginLeft: 12 }} />
          <TextInput
            style={styles.searchInput}
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholder="Search recipes..."
            placeholderTextColor="#9CA3AF"
            returnKeyType="search"
          />
        </View>
      </View>

      {/* Action buttons */}
      <View style={styles.actionsRow}>
        <Pressable style={styles.actionBtn} onPress={handleCreateRecipe}>
          <Ionicons name="add-circle-outline" size={20} color={colors.accent.green} />
          <Text style={styles.actionText}>Create</Text>
        </Pressable>
        <Pressable style={styles.actionBtn} onPress={() => setImportModalVisible(true)}>
          <Ionicons name="link-outline" size={20} color={colors.accent.blue} />
          <Text style={[styles.actionText, { color: colors.accent.blue }]}>Import URL</Text>
        </Pressable>
      </View>

      <FlatList
        data={displayedRecipes}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => {
          const servings = item.servings || 1;
          const calPerServing = Math.round(item.totalCalories / servings);
          const pPerServing = Math.round(item.totalProtein / servings);
          const cPerServing = Math.round(item.totalCarbs / servings);
          const fPerServing = Math.round(item.totalFat / servings);

          return (
            <Pressable style={styles.recipeCard} onPress={() => handleLogRecipe(item)}>
              {/* Photo thumbnail */}
              {item.photoUri ? (
                <Image source={{ uri: item.photoUri }} style={styles.recipeThumb} />
              ) : (
                <View style={styles.recipeThumbPlaceholder}>
                  <Ionicons name="restaurant-outline" size={20} color={colors.border.default} />
                </View>
              )}
              <View style={styles.recipeInfo}>
                <Text style={styles.recipeName} numberOfLines={1}>{item.name}</Text>
                <Text style={styles.recipeMacros}>
                  {calPerServing} Cal · P{pPerServing} C{cPerServing} F{fPerServing} /serving
                </Text>
                <Text style={styles.recipeUsage}>
                  {item.timesUsed > 0 ? `Used ${item.timesUsed} time${item.timesUsed !== 1 ? 's' : ''}` : 'Never used'}
                </Text>
              </View>
              <Pressable
                style={styles.recipeEditBtn}
                onPress={() => {
                  const full = loadRecipe(item.id);
                  setActiveRecipe(full);
                  if (full) setBuilderServings(String(full.servings || 1));
                }}
              >
                <Ionicons name="create-outline" size={20} color={colors.text.tertiary} />
              </Pressable>
            </Pressable>
          );
        }}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons name="restaurant-outline" size={48} color={colors.border.default} />
            <Text style={styles.emptyTitle}>No recipes yet</Text>
            <Text style={styles.emptySub}>Create one or import from a URL.</Text>
          </View>
        }
      />

      {/* Import URL Modal */}
      <Modal visible={importModalVisible} transparent animationType="fade" onRequestClose={() => setImportModalVisible(false)}>
        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.modalOverlay}>
          <Pressable style={styles.modalBackdrop} onPress={() => setImportModalVisible(false)} />
          <View style={styles.importSheet}>
            <Text style={styles.importTitle}>Import from URL</Text>
            <Text style={styles.importHint}>Paste a recipe URL from any cooking site</Text>
            <TextInput
              style={styles.importInput}
              value={importUrl}
              onChangeText={setImportUrl}
              placeholder="https://allrecipes.com/recipe/..."
              placeholderTextColor="#9CA3AF"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              autoFocus
            />
            {importing ? (
              <ActivityIndicator size="small" color={colors.accent.green} style={{ marginTop: 16 }} />
            ) : (
              <View style={styles.importActions}>
                <Pressable style={styles.importCancel} onPress={() => setImportModalVisible(false)}>
                  <Text style={styles.importCancelText}>Cancel</Text>
                </Pressable>
                <Pressable style={styles.importConfirm} onPress={handleImportUrl}>
                  <Text style={styles.importConfirmText}>Import</Text>
                </Pressable>
              </View>
            )}
          </View>
        </KeyboardAvoidingView>
      </Modal>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background.primary },

  // List header
  listHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 50, paddingBottom: 12,
    backgroundColor: colors.background.elevated, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
  },
  listTitle: { fontSize: 17, fontWeight: '700', color: colors.text.primary },

  // Search
  searchRow: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 4, backgroundColor: colors.background.elevated },
  searchInputContainer: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: colors.background.surface, borderRadius: 12,
  },
  searchInput: {
    flex: 1, paddingHorizontal: 10, paddingVertical: 10,
    fontSize: 15, color: colors.text.primary,
  },

  // Actions
  actionsRow: {
    flexDirection: 'row', gap: 12, paddingHorizontal: 16, paddingVertical: 12,
  },
  actionBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    backgroundColor: colors.background.elevated, borderRadius: 12, paddingVertical: 12,
    borderWidth: 1, borderColor: colors.border.subtle,
  },
  actionText: { fontSize: 14, fontWeight: '600', color: colors.accent.green },

  // Recipe list
  listContent: { paddingBottom: 40 },
  recipeCard: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: colors.background.elevated,
    marginHorizontal: 16, marginBottom: 8, borderRadius: 12, padding: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.03, shadowRadius: 4, elevation: 1,
  },
  recipeThumb: {
    width: 44, height: 44, borderRadius: 10, marginRight: 12,
  },
  recipeThumbPlaceholder: {
    width: 44, height: 44, borderRadius: 10, marginRight: 12,
    backgroundColor: colors.background.surface, justifyContent: 'center', alignItems: 'center',
  },
  recipeInfo: { flex: 1 },
  recipeName: { fontSize: 15, fontWeight: '600', color: colors.text.primary },
  recipeMacros: { fontSize: 12, color: colors.text.secondary, marginTop: 2 },
  recipeUsage: { fontSize: 11, color: colors.text.tertiary, marginTop: 1 },
  recipeEditBtn: { padding: 8 },

  emptyState: { alignItems: 'center', paddingVertical: 60 },
  emptyTitle: { fontSize: 18, fontWeight: '600', color: colors.text.tertiary, marginTop: 12 },
  emptySub: { fontSize: 14, color: colors.border.default, marginTop: 4 },

  // Builder
  builderHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 50, paddingBottom: 12,
    backgroundColor: colors.background.elevated, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
  },
  builderTitleInput: {
    fontSize: 17, fontWeight: '700', color: colors.text.primary, flex: 1, textAlign: 'center',
    marginHorizontal: 12, paddingVertical: 4, borderBottomWidth: 1, borderBottomColor: colors.border.subtle,
  },
  builderContent: { padding: 16, paddingBottom: 40 },

  totalsRow: {
    backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16, marginBottom: 12,
    alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  totalsBig: { fontSize: 24, fontWeight: '800', color: colors.text.primary },
  totalsSub: { fontSize: 13, color: colors.text.tertiary, marginTop: 4 },
  perServingSub: { fontSize: 12, color: colors.accent.purple, marginTop: 4, fontWeight: '600' },

  servingsRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    marginBottom: 16, paddingHorizontal: 4,
  },
  servingsLabel: { fontSize: 15, fontWeight: '500', color: colors.text.secondary },
  servingsInput: {
    backgroundColor: colors.background.elevated, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
    fontSize: 15, fontWeight: '700', color: colors.text.primary, minWidth: 50, textAlign: 'center',
    borderWidth: 1, borderColor: colors.border.subtle,
  },

  sectionLabel: { fontSize: 15, fontWeight: '700', color: colors.text.secondary, marginBottom: 8 },

  ingRow: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: colors.background.elevated,
    borderRadius: 10, padding: 12, marginBottom: 6,
  },
  ingInfo: { flex: 1 },
  ingName: { fontSize: 14, fontWeight: '500', color: colors.text.primary },
  ingMeta: { fontSize: 12, color: colors.text.tertiary, marginTop: 2 },
  ingRemove: { padding: 4, marginLeft: 8 },

  addIngForm: {
    backgroundColor: colors.background.elevated, borderRadius: 12, padding: 12, marginTop: 8, marginBottom: 12,
    borderWidth: 1, borderColor: colors.border.subtle,
  },
  addIngInput: {
    backgroundColor: colors.background.surface, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10,
    fontSize: 14, color: colors.text.primary, marginBottom: 8,
  },
  addIngRow: { flexDirection: 'row', gap: 8 },
  addIngBtn: {
    backgroundColor: colors.accent.green, borderRadius: 8, width: 40, height: 40,
    justifyContent: 'center', alignItems: 'center',
  },

  versionBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    backgroundColor: colors.accentTint.purple, borderRadius: 14, paddingVertical: 14, marginBottom: 12,
    borderWidth: 1, borderColor: '#DDD6FE',
  },
  versionBtnText: { color: colors.accent.purple, fontSize: 15, fontWeight: '600' },

  logBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    backgroundColor: colors.accent.green, borderRadius: 14, paddingVertical: 16,
  },
  logBtnText: { color: colors.text.inverse, fontSize: 16, fontWeight: '700' },

  // Import modal
  modalOverlay: { flex: 1, justifyContent: 'flex-end' },
  modalBackdrop: { flex: 1 },
  importSheet: {
    backgroundColor: colors.background.elevated, borderTopLeftRadius: 20, borderTopRightRadius: 20,
    padding: 24, paddingBottom: 40,
    shadowColor: '#000', shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 10,
  },
  importTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary, textAlign: 'center', marginBottom: 4 },
  importHint: { fontSize: 13, color: colors.text.tertiary, textAlign: 'center', marginBottom: 16 },
  importInput: {
    backgroundColor: colors.background.surface, borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14,
    fontSize: 15, color: colors.text.primary,
  },
  importActions: { flexDirection: 'row', gap: 12, marginTop: 16 },
  importCancel: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: colors.background.surface, alignItems: 'center' },
  importCancelText: { fontSize: 16, fontWeight: '600', color: colors.text.tertiary },
  importConfirm: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: colors.accent.green, alignItems: 'center' },
  importConfirmText: { fontSize: 16, fontWeight: '700', color: colors.text.inverse },
});
}
