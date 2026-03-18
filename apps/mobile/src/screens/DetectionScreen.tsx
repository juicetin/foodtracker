/**
 * DetectionScreen — Gemini Nano food scan pipeline.
 *
 * Flow: idle → analyzing → results → (saved)
 * Supports multi-photo selection from gallery (processes sequentially,
 * first photo shown immediately, rest processed in background).
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';

import { scanFood } from '../services/vlm/vlmPipeline';
import { useDetectionStore } from '../store/useDetectionStore';
import { useFoodLogStore } from '../store/useFoodLogStore';
import DishCard from '../components/detection/DishCard';
import IngredientSearchSheet from '../components/detection/IngredientSearchSheet';
import type { MealType } from '../services/detection/types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type FlowState = 'idle' | 'analyzing' | 'results' | 'saving';

const MEAL_TYPES: MealType[] = ['breakfast', 'lunch', 'snack', 'dinner'];
const MEAL_LABELS: Record<MealType, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  snack: 'Snack',
  dinner: 'Dinner',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DetectionScreen() {
  const navigation = useNavigation();
  const [flowState, setFlowState] = useState<FlowState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [analyzingPhotoUri, setAnalyzingPhotoUri] = useState<string | null>(null);

  // Ingredient search sheet state
  const [searchTarget, setSearchTarget] = useState<{
    dishId: string;
    ingId: string;
    currentName: string;
  } | null>(null);

  const backgroundProcessing = useRef(false);

  const {
    photoUri,
    dishes,
    isMock,
    mealType,
    pendingPhotos,
    totalPhotos,
    setAnalyzing,
    setScanResult,
    addScanResult,
    setPendingPhotos,
    setMealType,
    updateDishScale,
    updateDishName,
    updateIngredient,
    removeIngredient,
    removeDish,
    getTotalNutrition,
    reset,
  } = useDetectionStore();

  const { logScanResult } = useFoodLogStore();

  // -------------------------------------------------------------------------
  // Photo selection + scanning
  // -------------------------------------------------------------------------

  async function pickAndScan(source: 'camera' | 'gallery') {
    setError(null);
    try {
      let result;
      if (source === 'camera') {
        const perm = await ImagePicker.requestCameraPermissionsAsync();
        if (!perm.granted) {
          Alert.alert('Permission required', 'Camera access is needed to take photos.');
          return;
        }
        result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.9 });
      } else {
        const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!perm.granted) {
          Alert.alert('Permission required', 'Photo library access is needed.');
          return;
        }
        result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ['images'],
          quality: 0.9,
          allowsMultipleSelection: true,
          selectionLimit: 10,
        });
      }

      if (result.canceled || !result.assets?.length) return;

      const assets = result.assets;
      const firstUri = assets[0].uri;

      // Show analyzing state with first photo
      setAnalyzingPhotoUri(firstUri);
      setFlowState('analyzing');
      setAnalyzing(true);

      // Set pending photo count (excluding first)
      if (assets.length > 1) {
        setPendingPhotos(assets.length - 1, assets.length);
      }

      try {
        // Process first photo
        const scanResult = await scanFood(firstUri);
        setScanResult(scanResult);
        setFlowState('results');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
        setFlowState('idle');
        setAnalyzingPhotoUri(null);
        setAnalyzing(false);
        return;
      } finally {
        setAnalyzing(false);
      }

      // Process remaining photos in background
      if (assets.length > 1) {
        backgroundProcessing.current = true;
        processRemainingPhotos(assets.slice(1).map((a) => a.uri));
      }
    } catch {
      setError('Failed to open photo source. Please try again.');
      setFlowState('idle');
    }
  }

  async function processRemainingPhotos(uris: string[]) {
    for (const uri of uris) {
      try {
        const result = await scanFood(uri);
        addScanResult(result);
      } catch (err) {
        console.warn('[Detection] Background photo failed:', err);
        // Decrement pending count even on failure
        setPendingPhotos(
          useDetectionStore.getState().pendingPhotos - 1,
          useDetectionStore.getState().totalPhotos,
        );
      }
    }
    backgroundProcessing.current = false;
  }

  // -------------------------------------------------------------------------
  // Ingredient search sheet handlers
  // -------------------------------------------------------------------------

  function handleIngredientNameTap(dishId: string, ingId: string, currentName: string) {
    setSearchTarget({ dishId, ingId, currentName });
  }

  function handleIngredientSearchSelect(name: string) {
    if (searchTarget) {
      updateIngredient(searchTarget.dishId, searchTarget.ingId, { name });
    }
    setSearchTarget(null);
  }

  function handleIngredientWeightChange(dishId: string, ingId: string, amount_g: number) {
    updateIngredient(dishId, ingId, { amount_g });
  }

  // -------------------------------------------------------------------------
  // Log meal
  // -------------------------------------------------------------------------

  const handleLogMeal = useCallback(async () => {
    if (dishes.length === 0 || !photoUri) return;
    setFlowState('saving');
    try {
      await logScanResult({ photoUri, dishes, isMock }, mealType);
      reset();
      setFlowState('idle');
      setAnalyzingPhotoUri(null);
      if (navigation.canGoBack()) navigation.goBack();
    } catch (err) {
      setFlowState('results');
      Alert.alert('Error', err instanceof Error ? err.message : 'Failed to save meal.');
    }
  }, [dishes, photoUri, isMock, mealType, logScanResult, reset, navigation]);

  // -------------------------------------------------------------------------
  // Go back
  // -------------------------------------------------------------------------

  function handleGoBack() {
    reset();
    setFlowState('idle');
    setAnalyzingPhotoUri(null);
    if (navigation.canGoBack()) navigation.goBack();
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const totals = getTotalNutrition();

  // ── IDLE ──────────────────────────────────────────────────────────────────
  if (flowState === 'idle') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.idleContainer}>
          <Pressable onPress={handleGoBack} style={styles.closeBtn}>
            <Text style={styles.closeBtnText}>✕</Text>
          </Pressable>

          <View style={styles.idleIcon}>
            <Text style={styles.idleIconText}>🍽️</Text>
          </View>
          <Text style={styles.idleTitle}>Log Your Meal</Text>
          <Text style={styles.idleSubtitle}>
            Take a photo or choose from your gallery.{'\n'}
            Select multiple photos to scan all at once.
          </Text>

          {error && (
            <View style={styles.errorBanner}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          <Pressable style={styles.primaryBtn} onPress={() => pickAndScan('camera')}>
            <Text style={styles.primaryBtnText}>📷  Take a Photo</Text>
          </Pressable>
          <Pressable style={styles.secondaryBtn} onPress={() => pickAndScan('gallery')}>
            <Text style={styles.secondaryBtnText}>🖼️  Choose from Gallery</Text>
          </Pressable>
          <Pressable
            style={[styles.secondaryBtn, { marginTop: 12, backgroundColor: '#EFF6FF' }]}
            onPress={() => (navigation as any).navigate('FoodSearch')}
          >
            <Text style={[styles.secondaryBtnText, { color: '#3B82F6' }]}>🔍  Search Food Database</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  // ── ANALYZING ─────────────────────────────────────────────────────────────
  if (flowState === 'analyzing') {
    return (
      <SafeAreaView style={styles.container}>
        {analyzingPhotoUri && (
          <Image source={{ uri: analyzingPhotoUri }} style={styles.analyzingPhoto} resizeMode="cover" />
        )}
        <View style={styles.analyzingOverlay}>
          <ActivityIndicator size="large" color="#16A34A" />
          <Text style={styles.analyzingTitle}>Analysing your meal…</Text>
          <Text style={styles.analyzingSubtitle}>
            Gemini Nano is identifying your food
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // ── SAVING ────────────────────────────────────────────────────────────────
  if (flowState === 'saving') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.savingContainer}>
          <ActivityIndicator size="large" color="#16A34A" />
          <Text style={styles.savingText}>Saving meal…</Text>
        </View>
      </SafeAreaView>
    );
  }

  // ── RESULTS ───────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.resultsHeader}>
        <Pressable onPress={handleGoBack} style={styles.headerBack}>
          <Text style={styles.headerBackText}>✕</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Detected Food</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Photo */}
        {photoUri && (
          <Image source={{ uri: photoUri }} style={styles.photoHeader} resizeMode="cover" />
        )}

        {/* Mock banner */}
        {isMock && (
          <View style={styles.mockBanner}>
            <Text style={styles.mockBannerText}>
              ⚠️  Demo mode — Gemini Nano not available on this device
            </Text>
          </View>
        )}

        {/* Background processing banner */}
        {pendingPhotos > 0 && (
          <View style={styles.processingBanner}>
            <ActivityIndicator size="small" color="#3B82F6" />
            <Text style={styles.processingText}>
              Processing {totalPhotos - pendingPhotos + 1}/{totalPhotos} photos…
            </Text>
          </View>
        )}

        {/* Meal type selector */}
        <View style={styles.mealTypeRow}>
          {MEAL_TYPES.map((type) => (
            <Pressable
              key={type}
              style={[styles.mealTypePill, mealType === type && styles.mealTypePillActive]}
              onPress={() => setMealType(type)}
            >
              <Text style={[styles.mealTypePillText, mealType === type && styles.mealTypePillTextActive]}>
                {MEAL_LABELS[type]}
              </Text>
            </Pressable>
          ))}
        </View>

        {/* Dish cards */}
        {dishes.length === 0 ? (
          <View style={styles.emptyDishesContainer}>
            <Text style={styles.emptyDishesText}>No dishes detected.</Text>
            <Pressable style={styles.secondaryBtn} onPress={handleGoBack}>
              <Text style={styles.secondaryBtnText}>Try again</Text>
            </Pressable>
          </View>
        ) : (
          dishes.map((dish) => (
            <DishCard
              key={dish.id}
              dish={dish}
              onScaleChange={updateDishScale}
              onNameChange={updateDishName}
              onIngredientNameTap={handleIngredientNameTap}
              onIngredientWeightChange={handleIngredientWeightChange}
              onRemoveIngredient={removeIngredient}
              onRemove={removeDish}
            />
          ))
        )}

        <View style={{ height: 140 }} />
      </ScrollView>

      {/* Sticky footer */}
      <View style={styles.footer}>
        <View style={styles.footerTotals}>
          <Text style={styles.footerCalories}>{Math.round(totals.calories)} kcal</Text>
          <View style={styles.footerMacros}>
            <Text style={styles.footerMacro}>
              <Text style={[styles.footerMacroNum, { color: '#3B82F6' }]}>{Math.round(totals.protein)}g</Text>
              <Text style={styles.footerMacroLabel}> P</Text>
            </Text>
            <Text style={styles.footerMacro}>
              <Text style={[styles.footerMacroNum, { color: '#D97706' }]}>{Math.round(totals.carbs)}g</Text>
              <Text style={styles.footerMacroLabel}> C</Text>
            </Text>
            <Text style={styles.footerMacro}>
              <Text style={[styles.footerMacroNum, { color: '#16A34A' }]}>{Math.round(totals.fat)}g</Text>
              <Text style={styles.footerMacroLabel}> F</Text>
            </Text>
          </View>
        </View>
        <Pressable
          style={[styles.logBtn, dishes.length === 0 && styles.logBtnDisabled]}
          onPress={handleLogMeal}
          disabled={dishes.length === 0}
        >
          <Text style={styles.logBtnText}>Log Meal</Text>
        </Pressable>
      </View>

      {/* Ingredient search sheet (single instance, state-driven) */}
      <IngredientSearchSheet
        visible={searchTarget !== null}
        initialQuery={searchTarget?.currentName ?? ''}
        onSelect={handleIngredientSearchSelect}
        onDismiss={() => setSearchTarget(null)}
      />
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },

  // ── Idle ──
  idleContainer: {
    flex: 1, justifyContent: 'center', alignItems: 'center',
    paddingHorizontal: 32, backgroundColor: '#FFFFFF',
  },
  closeBtn: { position: 'absolute', top: 16, right: 20, padding: 8 },
  closeBtnText: { fontSize: 18, color: '#9CA3AF', fontWeight: '600' },
  idleIcon: {
    width: 80, height: 80, borderRadius: 40, backgroundColor: '#F0FDF4',
    alignItems: 'center', justifyContent: 'center', marginBottom: 20,
  },
  idleIconText: { fontSize: 36 },
  idleTitle: { fontSize: 26, fontWeight: '800', color: '#111827', marginBottom: 10 },
  idleSubtitle: {
    fontSize: 15, color: '#6B7280', textAlign: 'center', lineHeight: 22, marginBottom: 36,
  },
  errorBanner: {
    width: '100%', backgroundColor: '#FEF2F2', borderRadius: 10, padding: 14, marginBottom: 20,
  },
  errorText: { color: '#DC2626', fontSize: 14, textAlign: 'center' },
  primaryBtn: {
    width: '100%', paddingVertical: 16, backgroundColor: '#16A34A',
    borderRadius: 14, alignItems: 'center', marginBottom: 12,
  },
  primaryBtnText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700', letterSpacing: 0.3 },
  secondaryBtn: {
    width: '100%', paddingVertical: 16, backgroundColor: '#F3F4F6',
    borderRadius: 14, alignItems: 'center',
  },
  secondaryBtnText: { color: '#374151', fontSize: 17, fontWeight: '600' },

  // ── Analyzing ──
  analyzingPhoto: { width: '100%', height: 240 },
  analyzingOverlay: {
    flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, paddingHorizontal: 32,
  },
  analyzingTitle: { fontSize: 20, fontWeight: '700', color: '#111827', marginTop: 8 },
  analyzingSubtitle: { fontSize: 14, color: '#6B7280', textAlign: 'center' },

  // ── Saving ──
  savingContainer: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16 },
  savingText: { fontSize: 16, color: '#6B7280', fontWeight: '500' },

  // ── Results ──
  resultsHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: '#FFFFFF', paddingHorizontal: 16, paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#E5E7EB',
  },
  headerBack: { padding: 4, width: 36 },
  headerBackText: { fontSize: 18, color: '#9CA3AF', fontWeight: '600' },
  headerTitle: { fontSize: 17, fontWeight: '700', color: '#111827' },
  headerRight: { width: 36 },
  scrollView: { flex: 1 },
  scrollContent: { paddingTop: 0 },
  photoHeader: { width: '100%', height: 220 },
  mockBanner: {
    backgroundColor: '#FEF3C7', paddingHorizontal: 16, paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#FDE68A',
  },
  mockBannerText: { fontSize: 13, color: '#92400E', textAlign: 'center', fontWeight: '500' },
  processingBanner: {
    backgroundColor: '#EFF6FF', paddingHorizontal: 16, paddingVertical: 10,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#DBEAFE',
  },
  processingText: { fontSize: 13, color: '#1E40AF', fontWeight: '500' },
  mealTypeRow: {
    flexDirection: 'row', paddingHorizontal: 16, paddingVertical: 14, gap: 8,
    backgroundColor: '#FFFFFF', marginBottom: 12,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  mealTypePill: {
    flex: 1, paddingVertical: 8, borderRadius: 20, backgroundColor: '#F3F4F6', alignItems: 'center',
  },
  mealTypePillActive: { backgroundColor: '#16A34A' },
  mealTypePillText: { fontSize: 13, fontWeight: '600', color: '#6B7280' },
  mealTypePillTextActive: { color: '#FFFFFF' },
  emptyDishesContainer: { alignItems: 'center', paddingVertical: 40, paddingHorizontal: 32, gap: 16 },
  emptyDishesText: { fontSize: 16, color: '#6B7280', textAlign: 'center' },

  // ── Footer ──
  footer: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    backgroundColor: '#FFFFFF', borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#E5E7EB', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 28,
    shadowColor: '#000', shadowOffset: { width: 0, height: -2 }, shadowOpacity: 0.06,
    shadowRadius: 8, elevation: 8,
  },
  footerTotals: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12,
  },
  footerCalories: { fontSize: 22, fontWeight: '800', color: '#111827' },
  footerMacros: { flexDirection: 'row', gap: 12 },
  footerMacro: { fontSize: 13 },
  footerMacroNum: { fontWeight: '700', fontSize: 14 },
  footerMacroLabel: { color: '#9CA3AF', fontSize: 12 },
  logBtn: {
    backgroundColor: '#16A34A', borderRadius: 14, paddingVertical: 16, alignItems: 'center',
  },
  logBtnDisabled: { backgroundColor: '#D1D5DB' },
  logBtnText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700', letterSpacing: 0.3 },
});
