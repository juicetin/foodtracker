/**
 * ServingSizeSelector -- inline portion picker for ingredient editing.
 *
 * Shows current weight with a dropdown for KG standard portions,
 * common portions (100g, 1 serving), and a custom free-form gram input.
 * Debounces free-form input 300ms before calling onWeightChange.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  TextInput,
  Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { getKnowledgeGraphService } from '../../services/knowledge-graph';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ServingSizeSelectorProps {
  ingredientId: string;
  ingredientName: string;
  currentAmountG: number;
  onWeightChange: (grams: number) => void;
}

interface PortionOption {
  label: string;
  grams: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ServingSizeSelector({
  ingredientId,
  ingredientName,
  currentAmountG,
  onWeightChange,
}: ServingSizeSelectorProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [showCustom, setShowCustom] = useState(false);
  const [customText, setCustomText] = useState('');
  const [kgPortions, setKgPortions] = useState<PortionOption[]>([]);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const kg = await getKnowledgeGraphService();
        if (kg && !cancelled) {
          const dish = await kg.searchDish(ingredientName);
          if (dish && dish.defaultServingGrams && !cancelled) {
            setKgPortions([
              { label: `1 serving (${Math.round(dish.defaultServingGrams)}g)`, grams: dish.defaultServingGrams },
            ]);
          }
        }
      } catch {
        // KG not available
      }
    })();
    return () => { cancelled = true; };
  }, [ingredientName]);

  const handleSelectPortion = useCallback((grams: number) => {
    onWeightChange(grams);
    setShowDropdown(false);
    setShowCustom(false);
  }, [onWeightChange]);

  const handleCustomTextChange = useCallback((text: string) => {
    setCustomText(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const grams = parseFloat(text);
      if (!isNaN(grams) && grams > 0) {
        onWeightChange(grams);
      }
    }, 300);
  }, [onWeightChange]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const commonPortions: PortionOption[] = [
    { label: '100g', grams: 100 },
    { label: `1 serving (${Math.round(currentAmountG)}g)`, grams: currentAmountG },
    { label: '50g', grams: 50 },
    { label: '150g', grams: 150 },
    { label: '200g', grams: 200 },
    { label: '250g', grams: 250 },
  ];

  const allPortions = [...kgPortions, ...commonPortions];

  return (
    <View style={styles.container}>
      <Pressable style={styles.chip} onPress={() => setShowDropdown(true)}>
        <Text style={styles.chipText}>{Math.round(currentAmountG)}g</Text>
        <Ionicons name="chevron-down" size={14} color={colors.accent.amber} />
      </Pressable>

      <Modal
        visible={showDropdown}
        transparent
        animationType="fade"
        onRequestClose={() => { setShowDropdown(false); setShowCustom(false); }}
      >
        <Pressable
          style={styles.overlay}
          onPress={() => { setShowDropdown(false); setShowCustom(false); }}
        >
          <View style={styles.dropdown} onStartShouldSetResponder={() => true}>
            <Text style={styles.dropdownTitle}>Serving Size</Text>

            {allPortions.map((opt, i) => (
              <Pressable
                key={`${opt.label}-${i}`}
                style={styles.option}
                onPress={() => handleSelectPortion(opt.grams)}
              >
                <Text style={styles.optionText}>{opt.label}</Text>
              </Pressable>
            ))}

            {!showCustom ? (
              <Pressable
                style={styles.option}
                onPress={() => { setShowCustom(true); setCustomText(String(Math.round(currentAmountG))); }}
              >
                <Text style={[styles.optionText, { color: colors.accent.blue }]}>Custom...</Text>
              </Pressable>
            ) : (
              <View style={styles.customRow}>
                <TextInput
                  style={styles.customInput}
                  value={customText}
                  onChangeText={handleCustomTextChange}
                  keyboardType="numeric"
                  placeholder="grams"
                  placeholderTextColor={colors.input.placeholder}
                  autoFocus
                  returnKeyType="done"
                  onSubmitEditing={() => {
                    const g = parseFloat(customText);
                    if (!isNaN(g) && g > 0) handleSelectPortion(g);
                  }}
                />
                <Text style={styles.customUnit}>g</Text>
              </View>
            )}
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: { flexDirection: 'row', alignItems: 'center' },
    chip: {
      flexDirection: 'row', alignItems: 'center', gap: 4,
      backgroundColor: colors.accentTint.amber, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6,
      borderWidth: 1, borderColor: colors.accent.amber,
    },
    chipText: { fontSize: 13, fontWeight: '700', color: colors.accent.amber },

    overlay: {
      flex: 1, backgroundColor: colors.overlay,
      justifyContent: 'center', alignItems: 'center',
    },
    dropdown: {
      backgroundColor: colors.background.elevated, borderRadius: 16, padding: 16,
      width: 260, maxHeight: 400,
      shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15,
      shadowRadius: 12, elevation: 8,
    },
    dropdownTitle: {
      fontSize: 15, fontWeight: '700', color: colors.text.primary, marginBottom: 12,
    },
    option: {
      paddingVertical: 12, paddingHorizontal: 8,
      borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border.subtle,
    },
    optionText: { fontSize: 15, color: colors.text.secondary },
    customRow: {
      flexDirection: 'row', alignItems: 'center', paddingVertical: 8, paddingHorizontal: 8,
    },
    customInput: {
      flex: 1, fontSize: 16, fontWeight: '600', color: colors.text.primary,
      backgroundColor: colors.input.background, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8,
      borderWidth: 1, borderColor: colors.input.border,
    },
    customUnit: { fontSize: 15, color: colors.text.tertiary, marginLeft: 8 },
  });
}
