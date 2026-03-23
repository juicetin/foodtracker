/**
 * BarcodeScanScreen — scan a barcode, look up the product on Open Food Facts,
 * show nutrition, adjust portion, and add to diary.
 */

import React, {useCallback, useEffect, useRef, useState, useMemo} from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../types';
import { lookupBarcode, type OFFProduct } from '../services/openfoodfacts/openFoodFactsService';
import { useFoodLogStore } from '../store/useFoodLogStore';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Phase = 'scanning' | 'loading' | 'result' | 'not_found';

function detectMealType(): string {
  const h = new Date().getHours();
  if (h < 10) return 'breakfast';
  if (h < 14) return 'lunch';
  if (h < 17) return 'snack';
  return 'dinner';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function BarcodeScanScreen() {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [permission, requestPermission] = useCameraPermissions();
  const [phase, setPhase] = useState<Phase>('scanning');
  const [product, setProduct] = useState<OFFProduct | null>(null);
  const [portionG, setPortionG] = useState('100');
  const [scannedCode, setScannedCode] = useState<string | null>(null);
  const scanLock = useRef(false);
  const { logBarcodeProduct } = useFoodLogStore();

  // Request permission on mount
  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission, requestPermission]);

  const handleBarCodeScanned = useCallback(
    async ({ data }: { data: string }) => {
      // Debounce — only process one scan
      if (scanLock.current) return;
      scanLock.current = true;
      setScannedCode(data);
      setPhase('loading');

      const result = await lookupBarcode(data);

      if (result) {
        setProduct(result);
        // Default portion to serving size if known, else 100g
        if (result.servingQuantityG) {
          setPortionG(String(Math.round(result.servingQuantityG)));
        }
        setPhase('result');
      } else {
        setPhase('not_found');
      }
    },
    [],
  );

  const portionNum = parseFloat(portionG) || 0;
  const scale = portionNum / 100;

  const handleAddToDiary = useCallback(() => {
    if (!product) return;

    const n = product.nutrimentsPer100g;
    logBarcodeProduct({
      name: [product.brand, product.name].filter(Boolean).join(' — '),
      barcode: product.barcode,
      totalCalories: Math.round(n.calories * scale),
      totalProtein: Math.round(n.protein * scale * 10) / 10,
      totalCarbs: Math.round(n.carbs * scale * 10) / 10,
      totalFat: Math.round(n.fat * scale * 10) / 10,
      portionG: portionNum,
      mealType: detectMealType(),
    });

    Alert.alert('Added', `${product.name} added to your diary.`, [
      { text: 'OK', onPress: () => navigation.goBack() },
    ]);
  }, [product, scale, portionNum, logBarcodeProduct, navigation]);

  const handleScanAgain = useCallback(() => {
    scanLock.current = false;
    setScannedCode(null);
    setProduct(null);
    setPhase('scanning');
  }, []);

  // ---------------------------------------------------------------------------
  // Permission not granted
  // ---------------------------------------------------------------------------

  if (!permission?.granted) {
    return (
      <View style={styles.center}>
        <Ionicons name="camera-outline" size={48} color={colors.text.tertiary} />
        <Text style={styles.permText}>Camera permission is required to scan barcodes.</Text>
        <Pressable style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Permission</Text>
        </Pressable>
        <Pressable style={styles.closeBtn} onPress={() => navigation.goBack()}>
          <Text style={styles.closeBtnText}>Go Back</Text>
        </Pressable>
      </View>
    );
  }

  // ---------------------------------------------------------------------------
  // Scanning phase
  // ---------------------------------------------------------------------------

  if (phase === 'scanning') {
    return (
      <View style={styles.scanContainer}>
        <CameraView
          style={StyleSheet.absoluteFill}
          facing="back"
          barcodeScannerSettings={{
            barcodeTypes: ['ean13', 'ean8', 'upc_a', 'upc_e', 'code128'],
          }}
          onBarcodeScanned={handleBarCodeScanned}
        />

        {/* Overlay */}
        <View style={styles.overlay}>
          {/* Close button */}
          <Pressable style={styles.overlayClose} onPress={() => navigation.goBack()}>
            <Ionicons name="close" size={28} color={colors.text.inverse} />
          </Pressable>

          {/* Scan frame */}
          <View style={styles.scanFrame}>
            <View style={[styles.corner, styles.cornerTL]} />
            <View style={[styles.corner, styles.cornerTR]} />
            <View style={[styles.corner, styles.cornerBL]} />
            <View style={[styles.corner, styles.cornerBR]} />
          </View>

          <Text style={styles.scanHint}>Point at a barcode</Text>
        </View>
      </View>
    );
  }

  // ---------------------------------------------------------------------------
  // Loading phase
  // ---------------------------------------------------------------------------

  if (phase === 'loading') {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={colors.accent.green} />
        <Text style={styles.loadingText}>Looking up barcode…</Text>
        <Text style={styles.barcodeText}>{scannedCode}</Text>
      </View>
    );
  }

  // ---------------------------------------------------------------------------
  // Not found
  // ---------------------------------------------------------------------------

  if (phase === 'not_found') {
    return (
      <View style={styles.center}>
        <Ionicons name="alert-circle-outline" size={56} color={colors.accent.red} />
        <Text style={styles.notFoundTitle}>Product not found</Text>
        <Text style={styles.notFoundSub}>
          Barcode {scannedCode} isn't in the Open Food Facts database.
        </Text>
        <Pressable style={styles.scanAgainBtn} onPress={handleScanAgain}>
          <Text style={styles.scanAgainText}>Scan Another</Text>
        </Pressable>
        <Pressable style={styles.closeBtn} onPress={() => navigation.goBack()}>
          <Text style={styles.closeBtnText}>Go Back</Text>
        </Pressable>
      </View>
    );
  }

  // ---------------------------------------------------------------------------
  // Result phase
  // ---------------------------------------------------------------------------

  if (!product) return null;
  const n = product.nutrimentsPer100g;

  return (
    <View style={styles.resultContainer}>
      {/* Header */}
      <View style={styles.resultHeader}>
        <Pressable onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={colors.text.primary} />
        </Pressable>
        <Text style={styles.resultTitle} numberOfLines={1}>
          Scanned Product
        </Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.resultScroll}>
        {/* Product info */}
        <View style={styles.productCard}>
          <Text style={styles.productName}>{product.name}</Text>
          {product.brand && <Text style={styles.productBrand}>{product.brand}</Text>}
          {product.quantity && <Text style={styles.productQty}>{product.quantity}</Text>}
          {product.nutritionGrade && (
            <View style={[styles.gradeBadge, gradeColor(product.nutritionGrade, colors)]}>
              <Text style={styles.gradeText}>Nutri-Score {product.nutritionGrade.toUpperCase()}</Text>
            </View>
          )}
        </View>

        {/* Portion input */}
        <View style={styles.portionCard}>
          <Text style={styles.portionLabel}>Portion size</Text>
          <View style={styles.portionRow}>
            <TextInput
              style={styles.portionInput}
              value={portionG}
              onChangeText={setPortionG}
              keyboardType="numeric"
              selectTextOnFocus
            />
            <Text style={styles.portionUnit}>g</Text>
          </View>
          {product.servingSize && (
            <Text style={styles.servingHint}>
              1 serving = {product.servingSize}
              {product.servingQuantityG ? ` (${product.servingQuantityG}g)` : ''}
            </Text>
          )}
        </View>

        {/* Nutrition for selected portion */}
        <View style={styles.nutritionCard}>
          <Text style={styles.nutritionTitle}>Nutrition for {portionG}g</Text>
          <NutritionRow label="Calories" value={`${Math.round(n.calories * scale)} kcal`} bold />
          <NutritionRow label="Protein" value={`${(n.protein * scale).toFixed(1)}g`} color={colors.accent.blue} />
          <NutritionRow label="Carbs" value={`${(n.carbs * scale).toFixed(1)}g`} color="#D97706" />
          <NutritionRow label="  Sugar" value={`${(n.sugar * scale).toFixed(1)}g`} indent />
          <NutritionRow label="Fat" value={`${(n.fat * scale).toFixed(1)}g`} color={colors.accent.green} />
          <NutritionRow label="  Saturated" value={`${(n.saturatedFat * scale).toFixed(1)}g`} indent />
          <NutritionRow label="Fiber" value={`${(n.fiber * scale).toFixed(1)}g`} />
          <NutritionRow label="Sodium" value={`${(n.sodium * scale * 1000).toFixed(0)}mg`} />
        </View>

        {/* Scan another */}
        <Pressable style={styles.scanAgainBtn} onPress={handleScanAgain}>
          <Ionicons name="barcode-outline" size={18} color={colors.accent.green} style={{ marginRight: 6 }} />
          <Text style={styles.scanAgainText}>Scan Another</Text>
        </Pressable>
      </ScrollView>

      {/* Sticky footer */}
      <View style={styles.footer}>
        <Pressable style={styles.addBtn} onPress={handleAddToDiary}>
          <Ionicons name="add-circle" size={20} color={colors.text.inverse} style={{ marginRight: 6 }} />
          <Text style={styles.addBtnText}>Add to Diary</Text>
        </Pressable>
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function NutritionRow({
  label,
  value,
  color,
  bold,
  indent,
}: {
  label: string;
  value: string;
  color?: string;
  bold?: boolean;
  indent?: boolean;
}) {
  const { colors } = useTheme();
  const rowStyles = useMemo(() => createStyles(colors), [colors]);
  return (
    <View style={rowStyles.nutRow}>
      <Text
        style={[
          rowStyles.nutLabel,
          bold && { fontWeight: '700' },
          indent && { color: colors.text.tertiary, fontSize: 13 },
        ]}
      >
        {label}
      </Text>
      <Text
        style={[
          rowStyles.nutValue,
          color ? { color } : undefined,
          bold && { fontWeight: '700', fontSize: 16 },
        ]}
      >
        {value}
      </Text>
    </View>
  );
}

function gradeColor(grade: string, colors: ThemeColors): { backgroundColor: string } {
  switch (grade.toLowerCase()) {
    case 'a': return { backgroundColor: colors.accent.green };
    case 'b': return { backgroundColor: '#65A30D' };
    case 'c': return { backgroundColor: '#EAB308' };
    case 'd': return { backgroundColor: '#EA580C' };
    case 'e': return { backgroundColor: colors.accent.red };
    default: return { backgroundColor: colors.text.tertiary };
  }
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
  // Scanning
  scanContainer: { flex: 1, backgroundColor: '#000' },
  overlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center', alignItems: 'center' },
  overlayClose: {
    position: 'absolute', top: 50, left: 20,
    width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center', alignItems: 'center',
  },
  scanFrame: {
    width: 260, height: 160, borderRadius: 12,
    borderWidth: 0, // corners only
  },
  corner: {
    position: 'absolute', width: 30, height: 30,
    borderColor: colors.accent.green, borderWidth: 3,
  },
  cornerTL: { top: 0, left: 0, borderBottomWidth: 0, borderRightWidth: 0, borderTopLeftRadius: 12 },
  cornerTR: { top: 0, right: 0, borderBottomWidth: 0, borderLeftWidth: 0, borderTopRightRadius: 12 },
  cornerBL: { bottom: 0, left: 0, borderTopWidth: 0, borderRightWidth: 0, borderBottomLeftRadius: 12 },
  cornerBR: { bottom: 0, right: 0, borderTopWidth: 0, borderLeftWidth: 0, borderBottomRightRadius: 12 },
  scanHint: { color: colors.text.inverse, fontSize: 16, fontWeight: '500', marginTop: 24 },

  // Center (loading, permission, not found)
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24, backgroundColor: colors.background.primary },
  permText: { fontSize: 16, color: colors.text.tertiary, textAlign: 'center', marginTop: 16 },
  permBtn: {
    marginTop: 20, backgroundColor: colors.accent.green, borderRadius: 12,
    paddingHorizontal: 24, paddingVertical: 12,
  },
  permBtnText: { color: colors.text.inverse, fontWeight: '600', fontSize: 15 },
  closeBtn: { marginTop: 12, padding: 12 },
  closeBtnText: { color: colors.text.tertiary, fontSize: 14 },
  loadingText: { marginTop: 16, fontSize: 16, color: colors.text.secondary },
  barcodeText: { marginTop: 4, fontSize: 13, color: colors.text.tertiary, fontFamily: 'monospace' },
  notFoundTitle: { fontSize: 20, fontWeight: '700', color: colors.text.primary, marginTop: 16 },
  notFoundSub: { fontSize: 14, color: colors.text.tertiary, textAlign: 'center', marginTop: 8, marginHorizontal: 32 },

  // Result
  resultContainer: { flex: 1, backgroundColor: colors.background.primary },
  resultHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 50, paddingBottom: 12,
    backgroundColor: colors.background.elevated, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
  },
  resultTitle: { fontSize: 17, fontWeight: '600', color: colors.text.primary, flex: 1, textAlign: 'center' },
  resultScroll: { paddingBottom: 100 },

  productCard: {
    backgroundColor: colors.background.elevated, marginHorizontal: 16, marginTop: 16, borderRadius: 16, padding: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  productName: { fontSize: 20, fontWeight: '700', color: colors.text.primary },
  productBrand: { fontSize: 15, color: colors.text.tertiary, marginTop: 2 },
  productQty: { fontSize: 13, color: colors.text.tertiary, marginTop: 2 },
  gradeBadge: {
    alignSelf: 'flex-start', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4, marginTop: 8,
  },
  gradeText: { color: colors.text.inverse, fontSize: 12, fontWeight: '700' },

  portionCard: {
    backgroundColor: colors.background.elevated, marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  portionLabel: { fontSize: 14, fontWeight: '600', color: colors.text.secondary },
  portionRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8 },
  portionInput: {
    backgroundColor: colors.background.surface, borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10,
    fontSize: 18, fontWeight: '700', color: colors.text.primary, minWidth: 80, textAlign: 'center',
  },
  portionUnit: { fontSize: 16, color: colors.text.tertiary, marginLeft: 8 },
  servingHint: { fontSize: 12, color: colors.text.tertiary, marginTop: 6 },

  nutritionCard: {
    backgroundColor: colors.background.elevated, marginHorizontal: 16, marginTop: 12, borderRadius: 16, padding: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3,
  },
  nutritionTitle: { fontSize: 15, fontWeight: '700', color: colors.text.primary, marginBottom: 10 },
  nutRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 6, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.background.surface,
  },
  nutLabel: { fontSize: 14, color: colors.text.secondary },
  nutValue: { fontSize: 14, fontWeight: '600', color: colors.text.primary },

  scanAgainBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    marginHorizontal: 16, marginTop: 16, paddingVertical: 12, borderRadius: 12,
    backgroundColor: colors.accentTint.green, borderWidth: 1, borderColor: colors.accentTint.green,
  },
  scanAgainText: { fontSize: 15, fontWeight: '600', color: colors.accent.green },

  // Footer
  footer: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    paddingHorizontal: 16, paddingBottom: 34, paddingTop: 12,
    backgroundColor: colors.background.elevated, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.border.subtle,
  },
  addBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: colors.accent.green, borderRadius: 14, paddingVertical: 14,
  },
  addBtnText: { color: colors.text.inverse, fontSize: 16, fontWeight: '700' },
});
}
