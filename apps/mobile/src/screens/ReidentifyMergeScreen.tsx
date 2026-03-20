/**
 * ReidentifyMergeScreen -- Full-screen diff/merge for Gemini Nano re-identification.
 *
 * Flow:
 * 1. On mount: loads existing entry ingredients into left column, calls reidentifyEntry()
 * 2. After scan: two columns (left = previous/discard, right = new/keep)
 * 3. User drags items between columns (each drag is an EditCommand for undo/redo)
 * 4. Footer: Reset (restore post-scan state) and Save+Confirm (persist to SQLite)
 *
 * Left = discard side, Right = keep side (per locked decision).
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { opsqlite } from '../../db/client';
import type { RootStackParamList } from '../types';
import { MergeColumn } from '../components/edit/MergeColumn';
import {
  reidentifyEntry,
  applyMergeResult,
  type MergeItem,
} from '../services/entryEditor/reidentifyService';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A command for moving an item between columns (for undo/redo). */
interface MoveItemCommand {
  item: MergeItem;
  from: 'left' | 'right';
  to: 'left' | 'right';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ReidentifyMergeScreen() {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const route = useRoute<RouteProp<RootStackParamList, 'ReidentifyMerge'>>();
  const { entryId } = route.params;

  const [loading, setLoading] = useState(true);
  const [leftItems, setLeftItems] = useState<MergeItem[]>([]);
  const [rightItems, setRightItems] = useState<MergeItem[]>([]);

  // Undo/redo stack
  const [commandStack, setCommandStack] = useState<MoveItemCommand[]>([]);
  const [pointer, setPointer] = useState(-1);

  // Post-scan original state (for Reset)
  const originalLeftRef = useRef<MergeItem[]>([]);
  const originalRightRef = useRef<MergeItem[]>([]);

  // Load entry data and perform re-identification
  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        // Load existing ingredients from entry
        const ingRows = opsqlite.executeSync(
          `SELECT i.id, i.name, i.amount_g, i.calories, i.protein, i.carbs, i.fat, i.fiber,
                  COALESCE(d.name, '') as dish_name
           FROM ingredients i
           LEFT JOIN scanned_dishes d ON d.id = i.dish_id
           WHERE i.entry_id = ?
           ORDER BY i.created_at`,
          [entryId],
        ).rows as Array<Record<string, unknown>>;

        const existingItems: MergeItem[] = ingRows.map((row) => ({
          id: row.id as string,
          name: row.name as string,
          amountG: (row.amount_g as number) ?? 0,
          calories: (row.calories as number) ?? 0,
          protein: (row.protein as number) ?? 0,
          carbs: (row.carbs as number) ?? 0,
          fat: (row.fat as number) ?? 0,
          fiber: (row.fiber as number) ?? 0,
          dishName: (row.dish_name as string) ?? '',
          source: 'existing' as const,
        }));

        // Get photo URI for re-scan
        const photoRows = opsqlite.executeSync(
          'SELECT uri FROM photos WHERE entry_id = ? LIMIT 1',
          [entryId],
        ).rows as Array<Record<string, unknown>>;

        if (photoRows.length === 0) {
          if (!cancelled) {
            Alert.alert('Error', 'No photo found for this entry.');
            navigation.goBack();
          }
          return;
        }

        const photoUri = photoRows[0].uri as string;

        // Re-scan with Gemini Nano
        const candidates = await reidentifyEntry(photoUri);

        if (cancelled) return;

        // Flatten candidates into MergeItem[]
        const newItems: MergeItem[] = candidates.flatMap((c) => c.ingredients);

        // Post-scan state: existing in left (discard), new in right (keep)
        originalLeftRef.current = existingItems;
        originalRightRef.current = newItems;

        setLeftItems(existingItems);
        setRightItems(newItems);
        setLoading(false);
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err);
          Alert.alert('Re-identification Failed', message);
          navigation.goBack();
        }
      }
    }

    run();
    return () => { cancelled = true; };
  }, [entryId, navigation]);

  // Handle drag completion -- move item between columns
  const handleItemDragged = useCallback((item: MergeItem, targetSide: 'left' | 'right') => {
    const fromSide = targetSide === 'right' ? 'left' : 'right';

    // Already on target side? No-op
    if (fromSide === targetSide) return;

    const cmd: MoveItemCommand = { item, from: fromSide, to: targetSide };

    // Truncate redo history and push new command
    setCommandStack((prev) => {
      const truncated = prev.slice(0, pointer + 1);
      truncated.push(cmd);
      return truncated;
    });
    setPointer((p) => p + 1);

    // Execute the move
    if (targetSide === 'right') {
      // Move from left to right
      setLeftItems((prev) => prev.filter((i) => i.id !== item.id));
      setRightItems((prev) => [...prev, item]);
    } else {
      // Move from right to left
      setRightItems((prev) => prev.filter((i) => i.id !== item.id));
      setLeftItems((prev) => [...prev, item]);
    }
  }, [pointer]);

  // Undo
  const canUndo = pointer >= 0;
  const handleUndo = useCallback(() => {
    if (pointer < 0) return;

    const cmd = commandStack[pointer];

    // Reverse the move
    if (cmd.to === 'right') {
      setRightItems((prev) => prev.filter((i) => i.id !== cmd.item.id));
      setLeftItems((prev) => [...prev, cmd.item]);
    } else {
      setLeftItems((prev) => prev.filter((i) => i.id !== cmd.item.id));
      setRightItems((prev) => [...prev, cmd.item]);
    }

    setPointer((p) => p - 1);
  }, [pointer, commandStack]);

  // Redo
  const canRedo = pointer < commandStack.length - 1;
  const handleRedo = useCallback(() => {
    if (pointer >= commandStack.length - 1) return;

    const nextPointer = pointer + 1;
    const cmd = commandStack[nextPointer];

    // Re-execute the move
    if (cmd.to === 'right') {
      setLeftItems((prev) => prev.filter((i) => i.id !== cmd.item.id));
      setRightItems((prev) => [...prev, cmd.item]);
    } else {
      setRightItems((prev) => prev.filter((i) => i.id !== cmd.item.id));
      setLeftItems((prev) => [...prev, cmd.item]);
    }

    setPointer(nextPointer);
  }, [pointer, commandStack]);

  // Reset to post-scan state
  const handleReset = useCallback(() => {
    setLeftItems([...originalLeftRef.current]);
    setRightItems([...originalRightRef.current]);
    setCommandStack([]);
    setPointer(-1);
  }, []);

  // Save + Confirm
  const handleSaveConfirm = useCallback(() => {
    try {
      applyMergeResult(entryId, rightItems);
      navigation.goBack();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      Alert.alert('Save Failed', message);
    }
  }, [entryId, rightItems, navigation]);

  // Close (X)
  const handleClose = useCallback(() => {
    navigation.goBack();
  }, [navigation]);

  // Keep totals
  const keepTotals = useMemo(() => {
    let cal = 0, pro = 0, carb = 0, fat = 0;
    for (const item of rightItems) {
      cal += item.calories;
      pro += item.protein;
      carb += item.carbs;
      fat += item.fat;
    }
    return { calories: Math.round(cal), protein: Math.round(pro), carbs: Math.round(carb), fat: Math.round(fat) };
  }, [rightItems]);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#7C3AED" />
        <Text style={styles.loadingText}>Re-scanning with Gemini Nano...</Text>
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable style={styles.closeBtn} onPress={handleClose}>
          <Ionicons name="close" size={22} color="#374151" />
        </Pressable>
        <Text style={styles.headerTitle}>Re-identify</Text>
        <View style={styles.undoRedoGroup}>
          <Pressable
            style={[styles.undoRedoBtn, !canUndo && styles.undoRedoBtnDisabled]}
            onPress={handleUndo}
            disabled={!canUndo}
          >
            <Ionicons name="arrow-undo" size={18} color={canUndo ? '#3B82F6' : '#D1D5DB'} />
          </Pressable>
          <Pressable
            style={[styles.undoRedoBtn, !canRedo && styles.undoRedoBtnDisabled]}
            onPress={handleRedo}
            disabled={!canRedo}
          >
            <Ionicons name="arrow-redo" size={18} color={canRedo ? '#3B82F6' : '#D1D5DB'} />
          </Pressable>
        </View>
      </View>

      {/* Keep totals summary */}
      <View style={styles.totalsSummary}>
        <Text style={styles.totalsLabel}>Keep total:</Text>
        <Text style={styles.totalsValue}>
          {keepTotals.calories} kcal  P{keepTotals.protein}  C{keepTotals.carbs}  F{keepTotals.fat}
        </Text>
      </View>

      {/* Two columns */}
      <View style={styles.columnsContainer}>
        <MergeColumn
          title="Discard"
          items={leftItems}
          side="left"
          onItemDraggedOut={handleItemDragged}
          headerColor="#DC2626"
        />
        <View style={styles.columnSpacer} />
        <MergeColumn
          title="Keep"
          items={rightItems}
          side="right"
          onItemDraggedOut={handleItemDragged}
          headerColor="#16A34A"
        />
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Pressable style={styles.resetBtn} onPress={handleReset}>
          <Ionicons name="refresh" size={16} color="#6B7280" />
          <Text style={styles.resetBtnText}>Reset</Text>
        </Pressable>
        <Pressable style={styles.saveBtn} onPress={handleSaveConfirm}>
          <Ionicons name="checkmark-circle" size={18} color="#FFF" />
          <Text style={styles.saveBtnText}>Save + Confirm</Text>
        </Pressable>
      </View>
    </GestureHandlerRootView>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
    gap: 16,
  },
  loadingText: {
    fontSize: 16,
    color: '#7C3AED',
    fontWeight: '500',
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: '#FFF',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E5E7EB',
  },
  closeBtn: {
    padding: 4,
  },
  headerTitle: {
    flex: 1,
    fontSize: 17,
    fontWeight: '700',
    color: '#111827',
    textAlign: 'center',
    marginHorizontal: 8,
  },
  undoRedoGroup: {
    flexDirection: 'row',
    gap: 4,
  },
  undoRedoBtn: {
    padding: 6,
    borderRadius: 8,
    backgroundColor: '#F3F4F6',
  },
  undoRedoBtnDisabled: {
    opacity: 0.4,
  },

  // Totals summary
  totalsSummary: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: '#F0FDF4',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#BBF7D0',
  },
  totalsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#16A34A',
    marginRight: 6,
  },
  totalsValue: {
    fontSize: 12,
    fontWeight: '700',
    color: '#111827',
  },

  // Columns
  columnsContainer: {
    flex: 1,
    flexDirection: 'row',
    paddingHorizontal: 8,
    paddingTop: 8,
  },
  columnSpacer: {
    width: 8,
  },

  // Footer
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFF',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#E5E7EB',
  },
  resetBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    backgroundColor: '#F9FAFB',
  },
  resetBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6B7280',
  },
  saveBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: '#16A34A',
  },
  saveBtnText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#FFF',
  },
});
