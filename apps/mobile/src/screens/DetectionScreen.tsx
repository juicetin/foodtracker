import React, { useCallback, useRef, useState } from 'react';
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
import {
  estimatePortion,
  type ImageSize,
} from '../services/detection/portionBridge';
import type { DetectedItem } from '../services/detection/types';

import {
  AnnotatedPhoto,
  BoundingBoxOverlay,
  SummaryBar,
  DetectionList,
  ItemDetailSheet,
  LogMealFAB,
  UndoToast,
} from '../components/detection';

// ---------------------------------------------------------------------------
// Flow states
// ---------------------------------------------------------------------------

type FlowState = 'idle' | 'picking' | 'detecting' | 'results' | 'logging';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Rough kcal-per-gram (Phase 2 proxy). */
const KCAL_PER_GRAM = 1.5;
/** Rough protein-per-gram (Phase 2 proxy). */
const PROTEIN_PER_GRAM = 0.1;
/** Rough carbs-per-gram (Phase 2 proxy). */
const CARB_PER_GRAM = 0.2;
/** Rough fat-per-gram (Phase 2 proxy). */
const FAT_PER_GRAM = 0.08;

/**
 * Placeholder class names for YOLO detection output decoding.
 * In production, these come from the model pack metadata.
 * For Phase 2, we use a compact list covering common food categories.
 */
const DEFAULT_CLASS_NAMES = [
  'rice', 'noodle', 'bread', 'meat', 'chicken', 'fish', 'egg',
  'vegetable', 'salad', 'fruit', 'soup', 'curry', 'pizza', 'burger',
  'sandwich', 'sushi', 'cake', 'pasta', 'steak', 'tofu',
];

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
  const { width: screenWidth } = useWindowDimensions();

  // -- Store hooks -----------------------------------------------------------
  const {
    photoUri,
    photoWidth,
    photoHeight,
    items,
    isDetecting,
    mealType,
    selectedItemId,
    setPhoto,
    setItems,
    setDetecting,
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

  // -- Photo dimensions for display -----------------------------------------
  const aspectRatio = photoWidth > 0 ? photoHeight / photoWidth : 1;
  const displayWidth = screenWidth;
  const displayHeight = Math.min(screenWidth * aspectRatio, screenWidth * 0.65);

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
    _uri: string,
    imgWidth: number,
    imgHeight: number,
  ) => {
    setFlowState('detecting');
    setDetecting(true);
    setError(null);

    try {
      // Load models if not already loaded
      await loadModelSet();

      // Phase 2 limitation: Image preprocessing (resize + pixel buffer conversion)
      // requires native module support. For now, create a placeholder buffer.
      // Real preprocessing will use expo-image-manipulator or a custom native module
      // to resize to 640x640 and convert to Float32Array pixel data.
      //
      // Known limitation: Without proper preprocessing, inference will not produce
      // accurate results until Phase 2.5 adds the image-to-tensor bridge.
      const modelInputSize = 640;
      const placeholderBuffer = new Float32Array(
        modelInputSize * modelInputSize * 3,
      ).buffer;

      const result = await runDetectionPipeline(
        placeholderBuffer,
        modelInputSize,
        modelInputSize,
        DEFAULT_CLASS_NAMES,
      );

      // Enrich each detected item with portion estimates
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
      // Calculate totals from active items
      let totalCal = 0;
      let totalProtein = 0;
      let totalCarbs = 0;
      let totalFat = 0;

      for (const item of currentActive) {
        const weightG = item.portionEstimate.weightG * item.portionMultiplier;
        totalCal += weightG * KCAL_PER_GRAM;
        totalProtein += weightG * PROTEIN_PER_GRAM;
        totalCarbs += weightG * CARB_PER_GRAM;
        totalFat += weightG * FAT_PER_GRAM;
      }

      await addEntry({
        mealType,
        totalCalories: Math.round(totalCal),
        totalProtein: Math.round(totalProtein),
        totalCarbs: Math.round(totalCarbs),
        totalFat: Math.round(totalFat),
        notes: `AI detected: ${currentActive.map((i) => i.className).join(', ')}`,
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
        {/* Annotated photo with bounding boxes (~50% screen height) */}
        {photoUri && (
          <View style={{ height: displayHeight }}>
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
});
