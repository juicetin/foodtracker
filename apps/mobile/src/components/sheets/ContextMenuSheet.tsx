/**
 * ContextMenuSheet -- long-press action menu for food items.
 *
 * Shows 5 actions: copy to clipboard, copy to another day, move to other meal,
 * save as favorite, delete. Triggers haptic feedback on open.
 */

import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import BottomSheet, { BottomSheetView, BottomSheetBackdrop } from '@gorhom/bottom-sheet';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import type { DiaryEntry } from '../../services/diary/diaryQueries';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ContextMenuAction = 'copy-clipboard' | 'copy-day' | 'move-meal' | 'favorite' | 'delete';

interface ContextMenuSheetProps {
  entry: DiaryEntry | null;
  onDismiss: () => void;
  onAction: (action: ContextMenuAction, entry: DiaryEntry) => void;
}

// ---------------------------------------------------------------------------
// Menu items config
// ---------------------------------------------------------------------------

const MENU_ITEMS: Array<{
  action: ContextMenuAction;
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  destructive?: boolean;
}> = [
  { action: 'copy-clipboard', icon: 'clipboard-outline', label: 'Copy to Clipboard' },
  { action: 'copy-day', icon: 'calendar-outline', label: 'Copy to Another Day' },
  { action: 'move-meal', icon: 'swap-horizontal-outline', label: 'Move to Other Meal' },
  { action: 'favorite', icon: 'heart-outline', label: 'Save as Favorite' },
  { action: 'delete', icon: 'trash-outline', label: 'Delete', destructive: true },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ContextMenuSheet({ entry, onDismiss, onAction }: ContextMenuSheetProps) {
  const bottomSheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ['35%'], []);

  useEffect(() => {
    if (entry) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      bottomSheetRef.current?.snapToIndex(0);
    } else {
      bottomSheetRef.current?.close();
    }
  }, [entry]);

  const handleChange = useCallback(
    (index: number) => {
      if (index === -1) {
        onDismiss();
      }
    },
    [onDismiss],
  );

  const renderBackdrop = useCallback(
    (props: React.ComponentProps<typeof BottomSheetBackdrop>) => (
      <BottomSheetBackdrop {...props} appearsOnIndex={0} disappearsOnIndex={-1} pressBehavior="close" />
    ),
    [],
  );

  const handlePress = useCallback(
    (action: ContextMenuAction) => {
      if (!entry) return;
      bottomSheetRef.current?.close();
      // Delay action to allow sheet dismiss animation
      setTimeout(() => {
        onAction(action, entry);
      }, 150);
    },
    [entry, onAction],
  );

  const dishName = entry?.dishes.map((d) => d.name).join(', ') ?? '';

  return (
    <BottomSheet
      ref={bottomSheetRef}
      index={-1}
      snapPoints={snapPoints}
      enablePanDownToClose
      onChange={handleChange}
      backdropComponent={renderBackdrop}
      backgroundStyle={styles.sheetBackground}
      handleIndicatorStyle={styles.handleIndicator}
    >
      <BottomSheetView style={styles.content}>
        {/* Title */}
        <Text style={styles.title} numberOfLines={1}>
          {dishName}
        </Text>
        <View style={styles.titleDivider} />

        {/* Menu items */}
        {MENU_ITEMS.map((item) => (
          <Pressable
            key={item.action}
            style={styles.menuItem}
            onPress={() => handlePress(item.action)}
          >
            <Ionicons
              name={item.icon}
              size={22}
              color={item.destructive ? '#EF4444' : '#374151'}
            />
            <Text style={[styles.menuLabel, item.destructive && styles.menuLabelDestructive]}>
              {item.label}
            </Text>
          </Pressable>
        ))}
      </BottomSheetView>
    </BottomSheet>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  sheetBackground: {
    backgroundColor: '#FFFFFF',
  },
  handleIndicator: {
    backgroundColor: '#D1D5DB',
  },
  content: {
    paddingHorizontal: 16,
    paddingTop: 4,
  },

  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 8,
  },
  titleDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: '#E5E7EB',
    marginBottom: 4,
  },

  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 48,
    paddingHorizontal: 0,
    gap: 12,
  },
  menuLabel: {
    fontSize: 14,
    color: '#374151',
  },
  menuLabelDestructive: {
    color: '#EF4444',
  },
});
