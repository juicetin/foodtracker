/**
 * AddFoodScreen — unified hub for all food entry methods.
 *
 * Consolidates search, camera, barcode, voice, quick-add, and gallery
 * entry paths into a single screen. Shows quick access tabs for
 * Recent/Frequent/Favorites/Recipes and entry method cards.
 *
 * Fixes QA-06: barcode is always visible on the add food screen
 * (both in search bar icons and as an entry method card).
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  Pressable,
  ScrollView,
  StyleSheet,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';

import { AddFoodSearchBar } from '../components/add-food/AddFoodSearchBar';
import {
  QuickAccessTabs,
  type QuickAccessItem,
  type QuickAccessTabKey,
} from '../components/add-food/QuickAccessTabs';
import { EntryMethodCards } from '../components/add-food/EntryMethodCards';
import { getRecentHistory } from '../services/search/historyService';
import { loadFavourites } from '../services/favourites';
import { autoDetectMealType } from '../services/detection/types';
import type { RootStackParamList } from '../types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MEAL_TYPE_LABELS: Record<string, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  snack: 'Snack',
  dinner: 'Dinner',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AddFoodScreen() {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const route = useRoute<RouteProp<RootStackParamList, 'AddFood'>>();

  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<QuickAccessTabKey>('recent');
  const [tabItems, setTabItems] = useState<QuickAccessItem[]>([]);
  const [mealType, setMealType] = useState(
    route.params?.mealType ?? autoDetectMealType(),
  );

  // ── Data loading ──
  const loadTabData = useCallback((tab: QuickAccessTabKey) => {
    try {
      switch (tab) {
        case 'recent': {
          const history = getRecentHistory(20);
          setTabItems(
            history.map((h) => ({
              id: `recent-${h.name}`,
              name: h.name,
              calories: h.avgCalories,
              protein: h.avgProtein,
              carbs: h.avgCarbs,
              fat: h.avgFat,
              source: 'history' as const,
            })),
          );
          break;
        }
        case 'frequent': {
          // getRecentHistory already sorts by totalCount DESC (frequency)
          const frequent = getRecentHistory(20);
          setTabItems(
            frequent.map((h) => ({
              id: `frequent-${h.name}`,
              name: h.name,
              calories: h.avgCalories,
              protein: h.avgProtein,
              carbs: h.avgCarbs,
              fat: h.avgFat,
              source: 'frequent' as const,
            })),
          );
          break;
        }
        case 'favorites': {
          const favs = loadFavourites(20);
          setTabItems(
            favs.map((f) => ({
              id: `fav-${f.id}`,
              name: f.name,
              calories: f.totalCalories,
              protein: f.totalProtein,
              carbs: f.totalCarbs,
              fat: f.totalFat,
              source: 'favourite' as const,
            })),
          );
          break;
        }
        case 'recipes': {
          // Recipes are loaded from the recipe service but we keep it
          // simple and use an empty list for now — the recipe service
          // searchRecipes requires a query string. Saved recipes will
          // be populated when the recipes feature is enhanced.
          setTabItems([]);
          break;
        }
      }
    } catch {
      setTabItems([]);
    }
  }, []);

  useEffect(() => {
    loadTabData(activeTab);
  }, [activeTab, loadTabData]);

  // ── Search ──
  function handleSearch() {
    if (searchQuery.trim().length >= 2) {
      navigation.navigate('FoodSearch');
    }
  }

  function handleSearchChange(text: string) {
    setSearchQuery(text);
    if (text.trim().length >= 2) {
      // Navigate to FoodSearchScreen for full search experience
      navigation.navigate('FoodSearch');
    }
  }

  // ── Action handlers ──
  function handleCameraPress() {
    navigation.navigate('Detection');
  }

  function handleBarcodePress() {
    navigation.navigate('BarcodeScan');
  }

  function handleVoicePress() {
    Alert.alert(
      'Voice Input',
      "Use your keyboard's voice input button to dictate food names.",
      [{ text: 'OK' }],
    );
  }

  function handleQuickAdd() {
    navigation.navigate('QuickAdd');
  }

  async function handleFromGallery() {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.8,
      });

      if (!result.canceled && result.assets.length > 0) {
        // Navigate to Detection with the selected image
        // The Detection screen will pick up the image from params or state
        navigation.navigate('Detection');
      }
    } catch {
      Alert.alert('Error', 'Could not open image gallery.');
    }
  }

  function handleItemPress(item: QuickAccessItem) {
    // Navigate to food search to allow user to review and log
    navigation.navigate('FoodSearch');
  }

  function handleTabChange(tab: QuickAccessTabKey) {
    setActiveTab(tab);
  }

  function handleGoBack() {
    if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }

  // ── Render ──
  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={handleGoBack} style={styles.headerClose}>
          <Ionicons name="close" size={24} color="#6B7280" />
        </Pressable>
        <Text style={styles.headerTitle}>Add Food</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Meal type pill */}
        <View style={styles.mealTypePillContainer}>
          <View style={styles.mealTypePill}>
            <Text style={styles.mealTypePillText}>
              {MEAL_TYPE_LABELS[mealType] ?? mealType}
            </Text>
          </View>
        </View>

        {/* Search bar */}
        <View style={styles.searchBarContainer}>
          <AddFoodSearchBar
            value={searchQuery}
            onChangeText={handleSearchChange}
            onCameraPress={handleCameraPress}
            onVoicePress={handleVoicePress}
            onBarcodePress={handleBarcodePress}
            onSubmit={handleSearch}
          />
        </View>

        {/* Entry method cards */}
        <View style={styles.cardsContainer}>
          <EntryMethodCards
            onScanPhoto={handleCameraPress}
            onScanBarcode={handleBarcodePress}
            onQuickAdd={handleQuickAdd}
            onFromGallery={handleFromGallery}
          />
        </View>

        {/* Quick access tabs */}
        <View style={styles.tabsContainer}>
          <QuickAccessTabs
            activeTab={activeTab}
            onTabChange={handleTabChange}
            items={tabItems}
            onItemPress={handleItemPress}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E5E7EB',
  },
  headerClose: {
    padding: 4,
    width: 36,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
  },
  headerSpacer: {
    width: 36,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingBottom: 40,
  },
  mealTypePillContainer: {
    flexDirection: 'row',
    marginTop: 16,
    marginBottom: 12,
  },
  mealTypePill: {
    backgroundColor: '#DCFCE7',
    borderRadius: 16,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  mealTypePillText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#16A34A',
  },
  searchBarContainer: {
    marginBottom: 16,
  },
  cardsContainer: {
    marginBottom: 20,
  },
  tabsContainer: {
    flex: 1,
    minHeight: 200,
  },
});
