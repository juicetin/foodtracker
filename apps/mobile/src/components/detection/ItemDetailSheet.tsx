import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import BottomSheet, { BottomSheetView } from '@gorhom/bottom-sheet';
import {
  CONFIDENCE_COLORS,
  DetectedItem,
  getConfidenceLevel,
} from '../../services/detection/types';
import { CorrectionStore } from '../../services/detection/correctionStore';
import { PortionSlider } from './PortionSlider';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ItemDetailSheetProps {
  item: DetectedItem | null;
  visible: boolean;
  onDismiss: () => void;
  onUpdatePortion: (id: string, multiplier: number) => void;
  onCorrectItem: (id: string, newClassName: string) => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Rough kcal-per-gram estimate (Phase 2 proxy).
 * Real macro calculation wired in Phase 3 via nutrition service.
 */
const KCAL_PER_GRAM = 1.5;
const PROTEIN_PER_GRAM = 0.1;
const CARB_PER_GRAM = 0.2;
const FAT_PER_GRAM = 0.08;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Bottom sheet detail card for a detected food item.
 *
 * Per locked decision: "Tapping a bounding box or list item opens a detail
 * card via @gorhom/bottom-sheet showing: food name, confidence, portion
 * estimate, macros preview, edit button."
 *
 * Features:
 * - Tappable food name for correction flow
 * - Confidence badge with colored dot
 * - PortionSlider (0.5x-3.0x)
 * - Macros preview (Phase 2 rough estimates)
 * - Suggestion pill from CorrectionStore
 * - Reference object hint when suggestReference is true
 */
export function ItemDetailSheet({
  item,
  visible,
  onDismiss,
  onUpdatePortion,
  onCorrectItem,
}: ItemDetailSheetProps) {
  const bottomSheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ['40%', '70%'], []);

  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState('');
  const [suggestion, setSuggestion] = useState<string | null>(null);

  // Fetch correction suggestion when item changes
  useEffect(() => {
    if (item) {
      setSuggestion(null);
      setIsEditing(false);
      CorrectionStore.getSuggestion(item.className)
        .then((s) => setSuggestion(s))
        .catch(() => {
          // Silently ignore -- suggestion is optional
        });
    }
  }, [item?.id, item?.className]);

  // Open/close the sheet based on visibility
  useEffect(() => {
    if (visible && item) {
      bottomSheetRef.current?.snapToIndex(0);
    } else {
      bottomSheetRef.current?.close();
    }
  }, [visible, item]);

  const handleSheetChange = useCallback(
    (index: number) => {
      if (index === -1) {
        onDismiss();
      }
    },
    [onDismiss],
  );

  const handlePortionChange = useCallback(
    (multiplier: number) => {
      if (item) {
        onUpdatePortion(item.id, multiplier);
      }
    },
    [item, onUpdatePortion],
  );

  const handleStartEdit = useCallback(() => {
    if (item) {
      setEditText(item.className);
      setIsEditing(true);
    }
  }, [item]);

  const handleSubmitCorrection = useCallback(() => {
    if (item && editText.trim() && editText.trim() !== item.className) {
      const newName = editText.trim();
      CorrectionStore.recordCorrection(
        item.correctedFrom ?? item.className,
        newName,
        item.confidence,
      ).catch(() => {
        // Silently ignore -- correction store writes are best-effort
      });
      onCorrectItem(item.id, newName);
    }
    setIsEditing(false);
  }, [item, editText, onCorrectItem]);

  const handleAcceptSuggestion = useCallback(() => {
    if (item && suggestion) {
      CorrectionStore.recordCorrection(
        item.correctedFrom ?? item.className,
        suggestion,
        item.confidence,
      ).catch(() => {
        // Best-effort
      });
      onCorrectItem(item.id, suggestion);
      setSuggestion(null);
    }
  }, [item, suggestion, onCorrectItem]);

  const handleCancelEdit = useCallback(() => {
    setIsEditing(false);
  }, []);

  if (!item) return null;

  const level = getConfidenceLevel(item.confidence);
  const color = CONFIDENCE_COLORS[level];
  const effectiveWeightG =
    item.portionEstimate.weightG * item.portionMultiplier;

  // Phase 2 rough macros estimates
  const calories = Math.round(effectiveWeightG * KCAL_PER_GRAM);
  const protein = Math.round(effectiveWeightG * PROTEIN_PER_GRAM);
  const carbs = Math.round(effectiveWeightG * CARB_PER_GRAM);
  const fat = Math.round(effectiveWeightG * FAT_PER_GRAM);

  return (
    <BottomSheet
      ref={bottomSheetRef}
      index={-1}
      snapPoints={snapPoints}
      enablePanDownToClose
      onChange={handleSheetChange}
      backgroundStyle={styles.sheetBackground}
      handleIndicatorStyle={styles.handleIndicator}
    >
      <BottomSheetView style={styles.content}>
        {/* Header: food name + confidence */}
        <View style={styles.header}>
          {isEditing ? (
            <View style={styles.editRow}>
              <TextInput
                style={styles.editInput}
                value={editText}
                onChangeText={setEditText}
                onSubmitEditing={handleSubmitCorrection}
                autoFocus
                returnKeyType="done"
                placeholder="Food name"
              />
              <Pressable onPress={handleSubmitCorrection} style={styles.editButton}>
                <Text style={styles.editButtonText}>Save</Text>
              </Pressable>
              <Pressable onPress={handleCancelEdit} style={styles.cancelButton}>
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </Pressable>
            </View>
          ) : (
            <Pressable onPress={handleStartEdit} style={styles.nameRow}>
              <Text style={styles.foodName}>{item.className}</Text>
              <Text style={styles.editHint}>tap to edit</Text>
            </Pressable>
          )}

          <View style={styles.confidenceBadge}>
            <View style={[styles.confidenceDot, { backgroundColor: color }]} />
            <Text style={[styles.confidenceText, { color }]}>
              {Math.round(item.confidence * 100)}%
            </Text>
          </View>
        </View>

        {/* Suggestion pill */}
        {suggestion && !isEditing && (
          <Pressable onPress={handleAcceptSuggestion} style={styles.suggestionPill}>
            <Text style={styles.suggestionText}>
              Did you mean {suggestion}?
            </Text>
          </Pressable>
        )}

        {/* Corrected-from notice */}
        {item.correctedFrom && (
          <Text style={styles.correctedFrom}>
            Originally: {item.correctedFrom}
          </Text>
        )}

        {/* Portion section */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Portion</Text>
          <Text style={styles.portionDescription}>
            ~{Math.round(effectiveWeightG)}g
          </Text>
          <PortionSlider
            baseWeightG={item.portionEstimate.weightG}
            multiplier={item.portionMultiplier}
            onMultiplierChange={handlePortionChange}
            confidence={item.portionEstimate.confidence}
            method={item.portionEstimate.method}
          />

          {/* Reference object hint */}
          {item.portionEstimate.suggestReference && (
            <Text style={styles.referenceHint}>
              Include a plate or coin next to food for better estimates
            </Text>
          )}
        </View>

        {/* Macros preview */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Estimated Macros</Text>
          <View style={styles.macrosGrid}>
            <MacroCell label="Calories" value={`${calories}`} unit="kcal" />
            <MacroCell label="Protein" value={`${protein}`} unit="g" />
            <MacroCell label="Carbs" value={`${carbs}`} unit="g" />
            <MacroCell label="Fat" value={`${fat}`} unit="g" />
          </View>
        </View>
      </BottomSheetView>
    </BottomSheet>
  );
}

// ---------------------------------------------------------------------------
// MacroCell helper
// ---------------------------------------------------------------------------

function MacroCell({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <View style={styles.macroCell}>
      <Text style={styles.macroValue}>
        {value}
        <Text style={styles.macroUnit}> {unit}</Text>
      </Text>
      <Text style={styles.macroLabel}>{label}</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  sheetBackground: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
  },
  handleIndicator: {
    backgroundColor: '#CCCCCC',
    width: 40,
  },
  content: {
    paddingHorizontal: 20,
    paddingBottom: 24,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  nameRow: {
    flex: 1,
    marginRight: 12,
  },
  foodName: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1A1A1A',
    textTransform: 'capitalize',
  },
  editHint: {
    fontSize: 11,
    color: '#999',
    marginTop: 2,
  },
  editRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
  },
  editInput: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    color: '#1A1A1A',
    borderBottomWidth: 2,
    borderBottomColor: '#007AFF',
    paddingVertical: 4,
    marginRight: 8,
  },
  editButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#007AFF',
    borderRadius: 6,
    marginRight: 4,
  },
  editButtonText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  cancelButton: {
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  cancelButtonText: {
    color: '#999',
    fontSize: 13,
  },
  confidenceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#F5F5F5',
  },
  confidenceDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  confidenceText: {
    fontSize: 13,
    fontWeight: '600',
  },
  suggestionPill: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 14,
    backgroundColor: '#E3F2FD',
    marginBottom: 12,
  },
  suggestionText: {
    fontSize: 13,
    color: '#1565C0',
    fontWeight: '500',
  },
  correctedFrom: {
    fontSize: 12,
    color: '#999',
    fontStyle: 'italic',
    marginBottom: 8,
  },
  section: {
    marginTop: 16,
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  portionDescription: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1A1A1A',
    marginBottom: 4,
  },
  referenceHint: {
    fontSize: 12,
    color: '#F57C00',
    fontStyle: 'italic',
    marginTop: 4,
  },
  macrosGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  macroCell: {
    alignItems: 'center',
    flex: 1,
  },
  macroValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1A1A1A',
  },
  macroUnit: {
    fontSize: 12,
    fontWeight: '400',
    color: '#666',
  },
  macroLabel: {
    fontSize: 11,
    color: '#999',
    marginTop: 2,
  },
});
