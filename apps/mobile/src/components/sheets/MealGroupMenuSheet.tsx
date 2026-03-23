/**
 * MealGroupMenuSheet -- long-press action menu for meal group headers.
 *
 * Shows 3 actions: copy from another day, copy yesterday's meal, save as
 * meal template. Triggers haptic feedback on open.
 */

import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import BottomSheet, { BottomSheetView, BottomSheetBackdrop } from '@gorhom/bottom-sheet';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { MEAL_GROUP_CONFIG, type MealGroup } from '../../services/diary/mealGroups';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MealGroupAction = 'copy-from-date' | 'copy-yesterday' | 'save-template';

interface MealGroupMenuSheetProps {
  mealGroup: MealGroup | null;
  selectedDate: string;
  onDismiss: () => void;
  onAction: (action: MealGroupAction, mealGroup: MealGroup) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MealGroupMenuSheet({
  mealGroup,
  selectedDate: _selectedDate,
  onDismiss,
  onAction,
}: MealGroupMenuSheetProps) {
  const bottomSheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ['28%'], []);

  useEffect(() => {
    if (mealGroup) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      bottomSheetRef.current?.snapToIndex(0);
    } else {
      bottomSheetRef.current?.close();
    }
  }, [mealGroup]);

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
    (action: MealGroupAction) => {
      if (!mealGroup) return;
      bottomSheetRef.current?.close();
      setTimeout(() => {
        onAction(action, mealGroup);
      }, 150);
    },
    [mealGroup, onAction],
  );

  const mealLabel = mealGroup ? MEAL_GROUP_CONFIG[mealGroup].label : '';

  const menuItems: Array<{
    action: MealGroupAction;
    icon: keyof typeof Ionicons.glyphMap;
    label: string;
  }> = useMemo(
    () => [
      { action: 'copy-from-date', icon: 'calendar-outline', label: 'Copy from Another Day' },
      {
        action: 'copy-yesterday',
        icon: 'arrow-back-outline',
        label: `Copy Yesterday's ${mealLabel}`,
      },
      { action: 'save-template', icon: 'bookmark-outline', label: 'Save as Meal Template' },
    ],
    [mealLabel],
  );

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
        <Text style={styles.title}>{mealLabel}</Text>
        <View style={styles.titleDivider} />

        {/* Menu items */}
        {menuItems.map((item) => (
          <Pressable
            key={item.action}
            style={styles.menuItem}
            onPress={() => handlePress(item.action)}
          >
            <Ionicons name={item.icon} size={22} color="#374151" />
            <Text style={styles.menuLabel}>{item.label}</Text>
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
});
