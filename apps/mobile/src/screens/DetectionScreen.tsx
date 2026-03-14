import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';

import { useDetectionStore } from '../store/useDetectionStore';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { runDetectionPipeline } from '../services/detection/inferenceRouter';
import { loadModelSet } from '../services/detection/modelLoader';
import { preprocessImageForModel } from '../services/detection/imagePreprocess';
import {
  DETECT_CLASS_NAMES,
  CLASSIFY_INPUT_SIZE,
  DETECT_INPUT_SIZE,
} from '../services/detection/constants';
import {
  estimatePortion,
  type ImageSize,
} from '../services/detection/portionBridge';
import type { DetectedItem } from '../services/detection/types';
import {
  getKnowledgeGraphService,
  type MacroResult,
} from '../services/knowledge-graph';

import { runVlmRefinement } from '../services/vlm/vlmPipeline';
import { vlmService } from '../services/vlm/vlmService';
import { detectVlmTier, getVlmTierConfig } from '../services/vlm/ramDetector';
import { PackManager } from '../services/packs/packManager';

import {
  AnnotatedPhoto,
  BoundingBoxOverlay,
  SummaryBar,
  DetectionList,
  ItemDetailSheet,
  LogMealFAB,
  UndoToast,
  MealTextInput,
  RefiningBadge,
} from '../components/detection';

// ---------------------------------------------------------------------------
// Flow states
// ---------------------------------------------------------------------------

type FlowState = 'idle' | 'picking' | 'detecting' | 'results' | 'logging';

// ---------------------------------------------------------------------------
// Constants -- flat-rate proxy (Tier 3 fallback when KG has no data)
// ---------------------------------------------------------------------------

/** Flat-rate kcal-per-gram proxy (used when KG has no recipe/dish data). */
const PROXY_KCAL_PER_GRAM = 1.5;
/** Flat-rate protein-per-gram proxy. */
const PROXY_PROTEIN_PER_GRAM = 0.1;
/** Flat-rate carbs-per-gram proxy. */
const PROXY_CARB_PER_GRAM = 0.2;
/** Flat-rate fat-per-gram proxy. */
const PROXY_FAT_PER_GRAM = 0.08;

// Input sizes imported from constants.ts:
// DETECT_INPUT_SIZE (640), CLASSIFY_INPUT_SIZE (224).
// DETECT_CLASS_NAMES (241 GGCD food classes) for YOLO decoding.
// Per-item YOLO labelling and EfficientNet-Lite0 classification handled in inferenceRouter.

// ---------------------------------------------------------------------------
// KG-powered nutrition helper
// ---------------------------------------------------------------------------

/**
 * Get nutrition for a detected item using the three-tier fallback chain:
 *   Tier 1: KG recipe decomposition (source='recipe')
 *   Tier 2: KG dish averages (source='dish_average')
 *   Tier 3: Flat-rate proxy (source='proxy')
 *
 * If the KG service is unavailable (null), goes straight to Tier 3.
 * Local SQLite queries take <5ms each, so this is fast enough for inline use.
 */
async function getNutritionForItem(item: DetectedItem): Promise<MacroResult> {
  const weightG = item.portionEstimate.weightG * item.portionMultiplier;

  try {
    const kgService = await getKnowledgeGraphService();

    if (kgService) {
      // Tier 1: Recipe decomposition (returns MacroResult with source='recipe')
      // calculateDishNutrition internally tries recipe first, then dish averages
      const kgResult = await kgService.calculateDishNutrition(
        item.className,
        weightG
      );

      if (kgResult) {
        if (__DEV__) {
          console.log(
            `[KG] ${item.className}: ${kgResult.source} -- ` +
              `${Math.round(kgResult.calories)} kcal, ` +
              `${Math.round(kgResult.protein)}g protein, ` +
              `${Math.round(kgResult.carbs)}g carbs, ` +
              `${Math.round(kgResult.fat)}g fat`
          );
        }
        return kgResult;
      }
    }
  } catch (err) {
    if (__DEV__) {
      console.warn(
        `[KG] Error querying nutrition for ${item.className}:`,
        err instanceof Error ? err.message : err
      );
    }
  }

  // Tier 3: Flat-rate proxy (KG not available or dish not found)
  const proxyResult: MacroResult = {
    calories: weightG * PROXY_KCAL_PER_GRAM,
    protein: weightG * PROXY_PROTEIN_PER_GRAM,
    carbs: weightG * PROXY_CARB_PER_GRAM,
    fat: weightG * PROXY_FAT_PER_GRAM,
    weightGrams: weightG,
    source: 'proxy',
  };

  if (__DEV__) {
    console.log(
      `[KG] ${item.className}: proxy -- ` +
        `${Math.round(proxyResult.calories)} kcal, ` +
        `${Math.round(proxyResult.protein)}g protein, ` +
        `${Math.round(proxyResult.carbs)}g carbs, ` +
        `${Math.round(proxyResult.fat)}g fat`
    );
  }

  return proxyResult;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Main detection flow screen.
 *
 * Orchestrates: photo selection -> spinner -> inference -> results -> log meal.
 *
 * Per locked decisions:
 * - Simple spinner with "Detecting foods..." text during inference
 * - Results persist until "Log Meal" or dismiss
 * - Tapping bounding box or list item opens same detail sheet
 * - Cross-highlight between bbox and list
 * - Meal type auto-detected from time of day
 */
export function DetectionScreen() {
  const navigation = useNavigation();
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();

  // -- Store hooks -----------------------------------------------------------
  const {
    photoUri,
    photoWidth,
    photoHeight,
    items,
    isDetecting,
    isRefining,
    userMealText,
    mealType,
    selectedItemId,
    setPhoto,
    setItems,
    setDetecting,
    setRefining,
    refineItem,
    setUserText,
    displayLabel,
    removeItem,
    restoreItem,
    updatePortion,
    correctItem,
    setMealType,
    selectItem,
    reset,
    activeItems,
  } = useDetectionStore();

  const { addEntry } = useFoodLogStore();

  // -- Local state -----------------------------------------------------------
  const [flowState, setFlowState] = useState<FlowState>('idle');
  const [undoVisible, setUndoVisible] = useState(false);
  const [lastRemovedItem, setLastRemovedItem] = useState<DetectedItem | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Ref to track if component is still mounted during async ops
  const mountedRef = useRef(true);

  // VLM state: track init attempt and debounce text input
  const vlmInitAttempted = useRef(false);
  const vlmRefinementDone = useRef(false);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastVlmTextRef = useRef<string>('');

  // -- VLM availability check ------------------------------------------------
  // VLM is required for usable detection. Check on mount and gate the flow.

  const [vlmAvailable, setVlmAvailable] = useState<boolean | null>(null); // null = checking

  useEffect(() => {
    const checkVlm = async () => {
      const tier = detectVlmTier();
      if (tier === 'none') {
        setVlmAvailable(false);
        return;
      }
      const tierConfig = getVlmTierConfig();
      if (!tierConfig) {
        setVlmAvailable(false);
        return;
      }
      const pack = await PackManager.getInstalledPack(tierConfig.modelId);
      setVlmAvailable(!!(pack && pack.mmprojFilePath));
    };
    checkVlm();
  }, []);

  // -- VLM lazy init and refinement (after YOLO results) --------------------

  useEffect(() => {
    if (flowState !== 'results') return;
    if (items.length === 0) return;

    const initAndRefine = async () => {
      if (!vlmInitAttempted.current) {
        vlmInitAttempted.current = true;

        const tierConfig = getVlmTierConfig();
        if (!tierConfig) return;

        try {
          const pack = await PackManager.getInstalledPack(tierConfig.modelId);
          if (pack && pack.mmprojFilePath) {
            await vlmService.init(pack.filePath, pack.mmprojFilePath);
          }
        } catch (err) {
          if (__DEV__) {
            console.warn('[VLM] Failed to init VLM:', err instanceof Error ? err.message : err);
          }
          return;
        }
      }

      // Run VLM refinement if ready and not already done
      if (vlmService.isReady && !vlmRefinementDone.current) {
        vlmRefinementDone.current = true;
        lastVlmTextRef.current = userMealText;

        try {
          setRefining(true);
          const refined = await runVlmRefinement(
            photoUri!,
            items,
            userMealText || undefined,
          );

          for (const item of refined) {
            if (item.vlmLabel) {
              refineItem(item.id, {
                vlmLabel: item.vlmLabel,
                vlmCuisine: item.vlmCuisine,
                vlmIngredients: item.vlmIngredients,
                vlmConfidence: item.vlmConfidence,
              });
            }
          }
        } catch (err) {
          if (__DEV__) {
            console.warn('[VLM] Refinement failed:', err instanceof Error ? err.message : err);
          }
        } finally {
          setRefining(false);
        }
      }
    };

    initAndRefine();
  }, [flowState, items.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Debounced re-refinement on user text change --------------------------

  const handleUserTextChange = useCallback((text: string) => {
    setUserText(text);

    // Debounce: re-trigger VLM refinement 500ms after last keystroke
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(async () => {
      if (!vlmService.isReady) return;
      if (text === lastVlmTextRef.current) return;

      lastVlmTextRef.current = text;

      try {
        setRefining(true);
        const refined = await runVlmRefinement(
          photoUri!,
          items,
          text || undefined,
        );

        for (const item of refined) {
          if (item.vlmLabel) {
            refineItem(item.id, {
              vlmLabel: item.vlmLabel,
              vlmCuisine: item.vlmCuisine,
              vlmIngredients: item.vlmIngredients,
              vlmConfidence: item.vlmConfidence,
            });
          }
        }
      } catch {
        // Graceful fallback
      } finally {
        setRefining(false);
      }
    }, 500);
  }, [photoUri, items, setUserText, setRefining, refineItem]);

  // -- Photo dimensions for display -----------------------------------------
  const aspectRatio = photoWidth > 0 ? photoHeight / photoWidth : 1;
  const displayWidth = screenWidth;
  // Cap photo at 35% of screen height so results are visible below
  const displayHeight = Math.min(screenWidth * aspectRatio, screenHeight * 0.35);

  // -- Active items ----------------------------------------------------------
  const active = activeItems();

  // -- Photo selection -------------------------------------------------------

  const pickFromCamera = useCallback(async () => {
    setFlowState('picking');
    setError(null);
    try {
      const permission = await ImagePicker.requestCameraPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission required', 'Camera access is needed to take photos.');
        setFlowState('idle');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ['images'],
        quality: 0.9,
      });
      if (result.canceled || !result.assets?.length) {
        setFlowState('idle');
        return;
      }
      const asset = result.assets[0];
      setPhoto(asset.uri, asset.width, asset.height);
      await runInference(asset.uri, asset.width, asset.height);
    } catch (err) {
      setFlowState('idle');
      setError('Failed to open camera. Please try again.');
    }
  }, []);

  const pickFromGallery = useCallback(async () => {
    setFlowState('picking');
    setError(null);
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission required', 'Photo library access is needed.');
        setFlowState('idle');
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.9,
      });
      if (result.canceled || !result.assets?.length) {
        setFlowState('idle');
        return;
      }
      const asset = result.assets[0];
      setPhoto(asset.uri, asset.width, asset.height);
      await runInference(asset.uri, asset.width, asset.height);
    } catch (err) {
      setFlowState('idle');
      setError('Failed to open photo library. Please try again.');
    }
  }, []);

  // -- Inference pipeline ----------------------------------------------------

  const runInference = useCallback(async (
    uri: string,
    imgWidth: number,
    imgHeight: number,
  ) => {
    setFlowState('detecting');
    setDetecting(true);
    setError(null);

    try {
      // Load models if not already loaded
      await loadModelSet();

      // Preprocess at two sizes: 640x640 for detection, 224x224 for classification
      const [detectPixels, classifyPixels] = await Promise.all([
        preprocessImageForModel(uri, DETECT_INPUT_SIZE),
        preprocessImageForModel(uri, CLASSIFY_INPUT_SIZE, 'imagenet'),
      ]);

      // Pass Float32Array directly (not .buffer) -- react-native-fast-tflite expects TypedArray
      // DETECT_CLASS_NAMES (241 GGCD food names) is passed for YOLO decoding.
      // All classes are food-specific -- no filtering needed.
      const result = await runDetectionPipeline(
        detectPixels,
        classifyPixels,
        DETECT_INPUT_SIZE,
        DETECT_INPUT_SIZE,
        DETECT_CLASS_NAMES,
      );

      // Enrich each detected item with portion estimates.
      // Items are food-only (241 GGCD classes) with per-box YOLO food labels.
      const imageSize: ImageSize = { width: imgWidth, height: imgHeight };
      const enrichedItems = result.items.map((item) => ({
        ...item,
        portionEstimate: estimatePortion(
          item.bbox,
          imageSize,
          item.className,
        ),
      }));

      setItems(enrichedItems);
      setFlowState('results');
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Detection failed unexpectedly.';

      // Friendly message for model-not-installed scenario
      if (message.includes('not installed') || message.includes('not loaded')) {
        setError(
          'Detection models are not installed yet. Download the model pack first.',
        );
      } else {
        setError(message);
      }
      setFlowState('idle');
    } finally {
      setDetecting(false);
    }
  }, [setDetecting, setItems]);

  // -- Item actions ----------------------------------------------------------

  const handleSelectItem = useCallback(
    (id: string) => {
      selectItem(id);
    },
    [selectItem],
  );

  const handleRemoveItem = useCallback(
    (id: string) => {
      const item = items.find((i) => i.id === id);
      if (item) {
        setLastRemovedItem(item);
        setUndoVisible(true);
      }
      removeItem(id);
    },
    [items, removeItem],
  );

  const handleUndo = useCallback(() => {
    if (lastRemovedItem) {
      restoreItem(lastRemovedItem.id);
    }
    setUndoVisible(false);
    setLastRemovedItem(null);
  }, [lastRemovedItem, restoreItem]);

  const handleUndoDismiss = useCallback(() => {
    setUndoVisible(false);
    setLastRemovedItem(null);
  }, []);

  const handleUpdatePortion = useCallback(
    (id: string, multiplier: number) => {
      updatePortion(id, multiplier);
    },
    [updatePortion],
  );

  const handleCorrectItem = useCallback(
    (id: string, newClassName: string) => {
      correctItem(id, newClassName);
    },
    [correctItem],
  );

  const handleDismissSheet = useCallback(() => {
    selectItem(null);
  }, [selectItem]);

  const handleChangeMealType = useCallback(
    (type: typeof mealType) => {
      setMealType(type);
    },
    [setMealType],
  );

  // -- Log Meal flow ---------------------------------------------------------

  const handleLogMeal = useCallback(async () => {
    const currentActive = activeItems();
    if (currentActive.length === 0) return;

    setFlowState('logging');

    try {
      // Calculate totals from active items using KG nutrition (three-tier fallback)
      let totalCal = 0;
      let totalProtein = 0;
      let totalCarbs = 0;
      let totalFat = 0;

      for (const item of currentActive) {
        const macros = await getNutritionForItem(item);
        totalCal += macros.calories;
        totalProtein += macros.protein;
        totalCarbs += macros.carbs;
        totalFat += macros.fat;
      }

      await addEntry({
        mealType,
        totalCalories: Math.round(totalCal),
        totalProtein: Math.round(totalProtein),
        totalCarbs: Math.round(totalCarbs),
        totalFat: Math.round(totalFat),
        notes: `AI detected: ${currentActive.map((i) => displayLabel(i)).join(', ')}`,
      });

      // Reset detection store and navigate back
      reset();
      setFlowState('idle');

      if (navigation.canGoBack()) {
        navigation.goBack();
      }
    } catch (err) {
      setFlowState('results');
      Alert.alert('Error', 'Failed to log meal. Please try again.');
    }
  }, [activeItems, mealType, addEntry, reset, navigation]);

  // -- Dismiss / go back ----------------------------------------------------

  const handleGoBack = useCallback(() => {
    reset();
    setFlowState('idle');
    if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }, [reset, navigation]);

  // -- Selected item for detail sheet ----------------------------------------
  const selectedItem =
    selectedItemId ? items.find((i) => i.id === selectedItemId) ?? null : null;

  // =========================================================================
  // Render
  // =========================================================================

  // -- VLM not available: redirect to download --------------------------------
  if (flowState === 'idle' && !photoUri && vlmAvailable === false) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.idleContainer}>
          {navigation.canGoBack() && (
            <Pressable onPress={handleGoBack} style={styles.backButton}>
              <Text style={styles.backButtonText}>Back</Text>
            </Pressable>
          )}

          <Text style={styles.idleTitle}>AI Model Required</Text>
          <Text style={styles.idleSubtitle}>
            Download the AI model to identify foods from photos. Detection requires the VLM model to be installed.
          </Text>

          <Pressable
            onPress={() => (navigation as any).navigate('VlmDownload')}
            style={styles.primaryButton}
          >
            <Text style={styles.primaryButtonText}>Download AI Model</Text>
          </Pressable>

          {navigation.canGoBack() && (
            <Pressable onPress={handleGoBack} style={styles.cancelButton}>
              <Text style={styles.cancelButtonText}>Go Back</Text>
            </Pressable>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // -- VLM check in progress: spinner ----------------------------------------
  if (vlmAvailable === null) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.spinnerContainer}>
          <ActivityIndicator size="large" color="#22C55E" />
        </View>
      </SafeAreaView>
    );
  }

  // -- Idle state: photo selection buttons -----------------------------------
  if (flowState === 'idle' && !photoUri) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.idleContainer}>
          {/* Back button if navigable */}
          {navigation.canGoBack() && (
            <Pressable onPress={handleGoBack} style={styles.backButton}>
              <Text style={styles.backButtonText}>Back</Text>
            </Pressable>
          )}

          <Text style={styles.idleTitle}>Detect Food</Text>
          <Text style={styles.idleSubtitle}>
            Take a photo or choose from your gallery to detect food items
          </Text>

          {error && (
            <View style={styles.errorBanner}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          <Pressable onPress={pickFromCamera} style={styles.primaryButton}>
            <Text style={styles.primaryButtonText}>Take a Photo</Text>
          </Pressable>

          <Pressable onPress={pickFromGallery} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Choose from Gallery</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  // -- Detecting state: spinner ----------------------------------------------
  if (flowState === 'detecting' || isDetecting) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.spinnerContainer}>
          <ActivityIndicator size="large" color="#22C55E" />
          <Text style={styles.spinnerText}>Detecting foods...</Text>
        </View>
      </SafeAreaView>
    );
  }

  // -- Error after detection attempt (with photo shown) ----------------------
  if (flowState === 'idle' && error && photoUri) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.idleContainer}>
          <View style={styles.errorBanner}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
          <Pressable onPress={pickFromCamera} style={styles.primaryButton}>
            <Text style={styles.primaryButtonText}>Try Again (Camera)</Text>
          </Pressable>
          <Pressable onPress={pickFromGallery} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Try Again (Gallery)</Text>
          </Pressable>
          <Pressable onPress={handleGoBack} style={styles.cancelButton}>
            <Text style={styles.cancelButtonText}>Go Back</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  // -- Results state: full detection UI --------------------------------------
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.resultsContainer}>
        {/* Header with dismiss button and refining badge */}
        <View style={styles.resultsHeader}>
          <Pressable onPress={handleGoBack} style={styles.dismissButton}>
            <Text style={styles.dismissButtonText}>Cancel</Text>
          </Pressable>
          <View style={styles.titleRow}>
            <Text style={styles.resultsTitle}>Detection Results</Text>
            <RefiningBadge visible={isRefining} />
          </View>
          <View style={styles.dismissButtonPlaceholder} />
        </View>

        {/* Meal text input for VLM disambiguation */}
        <MealTextInput
          value={userMealText}
          onChangeText={handleUserTextChange}
          disabled={flowState !== 'results'}
        />

        {/* Annotated photo with bounding boxes */}
        {photoUri && (
          <View style={{ height: displayHeight, overflow: 'hidden' }}>
            <AnnotatedPhoto
              photoUri={photoUri}
              photoWidth={photoWidth}
              photoHeight={photoHeight}
            >
              <BoundingBoxOverlay
                items={items}
                photoWidth={photoWidth}
                photoHeight={photoHeight}
                displayWidth={displayWidth}
                displayHeight={displayHeight}
                selectedItemId={selectedItemId}
                onSelectItem={handleSelectItem}
                onRemoveItem={handleRemoveItem}
              />
            </AnnotatedPhoto>
          </View>
        )}

        {/* Summary bar */}
        <SummaryBar
          items={active}
          mealType={mealType}
          onChangeMealType={handleChangeMealType}
        />

        {/* Detection list (scrollable, fills remaining space) */}
        <DetectionList
          items={active}
          selectedItemId={selectedItemId}
          onSelectItem={handleSelectItem}
          onRemoveItem={handleRemoveItem}
        />

        {/* Log Meal FAB */}
        <LogMealFAB
          itemCount={active.length}
          onPress={handleLogMeal}
        />

        {/* Undo toast */}
        <UndoToast
          itemName={lastRemovedItem?.className ?? ''}
          onUndo={handleUndo}
          visible={undoVisible}
          onDismiss={handleUndoDismiss}
        />

        {/* Item detail bottom sheet */}
        <ItemDetailSheet
          item={selectedItem}
          visible={selectedItemId !== null}
          onDismiss={handleDismissSheet}
          onUpdatePortion={handleUpdatePortion}
          onCorrectItem={handleCorrectItem}
        />
      </View>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  // -- Idle state --
  idleContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  backButton: {
    position: 'absolute',
    top: 16,
    left: 16,
  },
  backButtonText: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '500',
  },
  idleTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1A1A1A',
    marginBottom: 8,
  },
  idleSubtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 22,
  },
  primaryButton: {
    width: '100%',
    paddingVertical: 16,
    backgroundColor: '#22C55E',
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 12,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '700',
  },
  secondaryButton: {
    width: '100%',
    paddingVertical: 16,
    backgroundColor: '#F5F5F5',
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 12,
  },
  secondaryButtonText: {
    color: '#333',
    fontSize: 17,
    fontWeight: '600',
  },
  cancelButton: {
    marginTop: 12,
    paddingVertical: 8,
  },
  cancelButtonText: {
    color: '#999',
    fontSize: 15,
  },
  // -- Spinner --
  spinnerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  spinnerText: {
    marginTop: 16,
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  // -- Error --
  errorBanner: {
    width: '100%',
    backgroundColor: '#FFF3E0',
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginBottom: 20,
  },
  errorText: {
    color: '#E65100',
    fontSize: 14,
    textAlign: 'center',
  },
  // -- Results --
  resultsContainer: {
    flex: 1,
  },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E0E0E0',
    backgroundColor: '#FAFAFA',
  },
  dismissButton: {
    paddingVertical: 4,
    paddingHorizontal: 8,
  },
  dismissButtonText: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '500',
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1A1A1A',
  },
  dismissButtonPlaceholder: {
    width: 60,
  },
});
