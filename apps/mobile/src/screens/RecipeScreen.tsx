/**
 * RecipeScreen — create, import, and manage recipes.
 *
 * Two modes:
 * - List: shows saved recipes with one-tap logging
 * - Builder: create/edit a recipe (manually or from URL import)
 */

import React, { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
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
  type RecipeSummary,
  type RecipeDetail,
} from '../services/recipes/recipeService';
import { parseRecipeFromHtml } from '../services/recipes/recipeUrlParser';
import { autoDetectMealType } from '../services/detection/types';
import { useFoodLogStore } from '../store/useFoodLogStore';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function RecipeScreen() {
  const navigation = useNavigation();
  const { loadTodayEntries } = useFoodLogStore();

  const [recipes, setRecipes] = useState<RecipeSummary[]>([]);
  const [activeRecipe, setActiveRecipe] = useState<RecipeDetail | null>(null);
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [importUrl, setImportUrl] = useState('');
  const [importing, setImporting] = useState(false);

  // New ingredient form
  const [newIngName, setNewIngName] = useState('');
  const [newIngQty, setNewIngQty] = useState('');
  const [newIngCal, setNewIngCal] = useState('');

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

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  function handleCreateRecipe() {
    // Android doesn't have Alert.prompt, use default name
    createRecipeWithPrompt();
  }

  function createRecipeWithPrompt() {
    // Android doesn't have Alert.prompt, use a simple default name
    const id = createRecipe({ name: 'New Recipe' });
    const recipe = loadRecipe(id);
    setActiveRecipe(recipe);
    refreshList();
  }

  function handleLogRecipe(recipe: RecipeSummary) {
    logRecipeAsEntry(recipe.id, autoDetectMealType());
    loadTodayEntries();
    refreshList();
    Alert.alert('Logged', `${recipe.name} added to diary.`);
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
          quantity: 0, // Raw ingredient text, no parsed quantity yet
          unit: 'serving',
          calories: 0,
          protein: 0,
          carbs: 0,
          fat: 0,
        });
      }

      // If nutrition is provided, update totals
      if (parsed.nutrition) {
        // We'll estimate per-ingredient later, for now just set totals
        const recipe = loadRecipe(id);
        setActiveRecipe(recipe);
      } else {
        const recipe = loadRecipe(id);
        setActiveRecipe(recipe);
      }

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
    return (
      <View style={styles.container}>
        <View style={styles.builderHeader}>
          <Pressable onPress={() => { setActiveRecipe(null); refreshList(); }}>
            <Ionicons name="arrow-back" size={24} color="#111827" />
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
            <Ionicons name="trash-outline" size={22} color="#DC2626" />
          </Pressable>
        </View>

        <ScrollView contentContainerStyle={styles.builderContent}>
          {/* Totals */}
          <View style={styles.totalsRow}>
            <Text style={styles.totalsBig}>{Math.round(activeRecipe.totalCalories)} kcal</Text>
            <Text style={styles.totalsSub}>
              P: {Math.round(activeRecipe.totalProtein)}g  C: {Math.round(activeRecipe.totalCarbs)}g  F: {Math.round(activeRecipe.totalFat)}g
            </Text>
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
                <Ionicons name="close-circle" size={20} color="#EF4444" />
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
                <Ionicons name="add" size={20} color="#FFF" />
              </Pressable>
            </View>
          </View>

          {/* Log recipe */}
          <Pressable
            style={styles.logBtn}
            onPress={() => {
              handleLogRecipe(activeRecipe);
              setActiveRecipe(null);
            }}
          >
            <Ionicons name="add-circle" size={20} color="#FFF" />
            <Text style={styles.logBtnText}>Log to Diary</Text>
          </Pressable>
        </ScrollView>
      </View>
    );
  }

  // ---------------------------------------------------------------------------
  // List view
  // ---------------------------------------------------------------------------

  return (
    <View style={styles.container}>
      <View style={styles.listHeader}>
        <Pressable onPress={() => navigation.canGoBack() && navigation.goBack()}>
          <Ionicons name="close" size={24} color="#111827" />
        </Pressable>
        <Text style={styles.listTitle}>Recipes</Text>
        <View style={{ width: 24 }} />
      </View>

      {/* Action buttons */}
      <View style={styles.actionsRow}>
        <Pressable style={styles.actionBtn} onPress={handleCreateRecipe}>
          <Ionicons name="add-circle-outline" size={20} color="#16A34A" />
          <Text style={styles.actionText}>Create</Text>
        </Pressable>
        <Pressable style={styles.actionBtn} onPress={() => setImportModalVisible(true)}>
          <Ionicons name="link-outline" size={20} color="#3B82F6" />
          <Text style={[styles.actionText, { color: '#3B82F6' }]}>Import URL</Text>
        </Pressable>
      </View>

      <FlatList
        data={recipes}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => (
          <Pressable style={styles.recipeCard} onPress={() => {
            const full = loadRecipe(item.id);
            setActiveRecipe(full);
          }}>
            <View style={styles.recipeInfo}>
              <Text style={styles.recipeName}>{item.name}</Text>
              <Text style={styles.recipeMeta}>
                {Math.round(item.totalCalories)} kcal
                {item.timesUsed > 0 ? ` · Used ${item.timesUsed}×` : ''}
              </Text>
            </View>
            <Pressable
              style={styles.recipeLogBtn}
              onPress={() => handleLogRecipe(item)}
            >
              <Ionicons name="add-circle" size={28} color="#16A34A" />
            </Pressable>
          </Pressable>
        )}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons name="restaurant-outline" size={48} color="#D1D5DB" />
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
              <ActivityIndicator size="small" color="#16A34A" style={{ marginTop: 16 }} />
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

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },

  // List header
  listHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 50, paddingBottom: 12,
    backgroundColor: '#FFF', borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#E5E7EB',
  },
  listTitle: { fontSize: 17, fontWeight: '700', color: '#111827' },

  // Actions
  actionsRow: {
    flexDirection: 'row', gap: 12, paddingHorizontal: 16, paddingVertical: 12,
  },
  actionBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    backgroundColor: '#FFF', borderRadius: 12, paddingVertical: 12,
    borderWidth: 1, borderColor: '#E5E7EB',
  },
  actionText: { fontSize: 14, fontWeight: '600', color: '#16A34A' },

  // Recipe list
  listContent: { paddingBottom: 40 },
  recipeCard: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#FFF',
    marginHorizontal: 16, marginBottom: 8, borderRadius: 12, padding: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.03, shadowRadius: 4, elevation: 1,
  },
  recipeInfo: { flex: 1 },
  recipeName: { fontSize: 15, fontWeight: '600', color: '#111827' },
  recipeMeta: { fontSize: 13, color: '#6B7280', marginTop: 2 },
  recipeLogBtn: { padding: 4 },

  emptyState: { alignItems: 'center', paddingVertical: 60 },
  emptyTitle: { fontSize: 18, fontWeight: '600', color: '#9CA3AF', marginTop: 12 },
  emptySub: { fontSize: 14, color: '#D1D5DB', marginTop: 4 },

  // Builder
  builderHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 50, paddingBottom: 12,
    backgroundColor: '#FFF', borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#E5E7EB',
  },
  builderTitleInput: {
    fontSize: 17, fontWeight: '700', color: '#111827', flex: 1, textAlign: 'center',
    marginHorizontal: 12, paddingVertical: 4, borderBottomWidth: 1, borderBottomColor: '#E5E7EB',
  },
  builderContent: { padding: 16, paddingBottom: 40 },

  totalsRow: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 16,
    alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  totalsBig: { fontSize: 24, fontWeight: '800', color: '#111827' },
  totalsSub: { fontSize: 13, color: '#6B7280', marginTop: 4 },

  sectionLabel: { fontSize: 15, fontWeight: '700', color: '#374151', marginBottom: 8 },

  ingRow: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#FFF',
    borderRadius: 10, padding: 12, marginBottom: 6,
  },
  ingInfo: { flex: 1 },
  ingName: { fontSize: 14, fontWeight: '500', color: '#111827' },
  ingMeta: { fontSize: 12, color: '#9CA3AF', marginTop: 2 },
  ingRemove: { padding: 4, marginLeft: 8 },

  addIngForm: {
    backgroundColor: '#FFF', borderRadius: 12, padding: 12, marginTop: 8, marginBottom: 20,
    borderWidth: 1, borderColor: '#E5E7EB',
  },
  addIngInput: {
    backgroundColor: '#F3F4F6', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10,
    fontSize: 14, color: '#111827', marginBottom: 8,
  },
  addIngRow: { flexDirection: 'row', gap: 8 },
  addIngBtn: {
    backgroundColor: '#16A34A', borderRadius: 8, width: 40, height: 40,
    justifyContent: 'center', alignItems: 'center',
  },

  logBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    backgroundColor: '#16A34A', borderRadius: 14, paddingVertical: 16,
  },
  logBtnText: { color: '#FFF', fontSize: 16, fontWeight: '700' },

  // Import modal
  modalOverlay: { flex: 1, justifyContent: 'flex-end' },
  modalBackdrop: { flex: 1 },
  importSheet: {
    backgroundColor: '#FFF', borderTopLeftRadius: 20, borderTopRightRadius: 20,
    padding: 24, paddingBottom: 40,
    shadowColor: '#000', shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 10,
  },
  importTitle: { fontSize: 18, fontWeight: '700', color: '#111827', textAlign: 'center', marginBottom: 4 },
  importHint: { fontSize: 13, color: '#9CA3AF', textAlign: 'center', marginBottom: 16 },
  importInput: {
    backgroundColor: '#F3F4F6', borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14,
    fontSize: 15, color: '#111827',
  },
  importActions: { flexDirection: 'row', gap: 12, marginTop: 16 },
  importCancel: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: '#F3F4F6', alignItems: 'center' },
  importCancelText: { fontSize: 16, fontWeight: '600', color: '#6B7280' },
  importConfirm: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: '#16A34A', alignItems: 'center' },
  importConfirmText: { fontSize: 16, fontWeight: '700', color: '#FFF' },
});
