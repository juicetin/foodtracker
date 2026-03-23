/**
 * QuickAccessTabs -- horizontal tab row for Recent/Frequent/Favorites/Recipes
 * with a scrollable item list below.
 *
 * Each tab shows relevant food items. Empty states guide the user on how
 * to populate each tab.
 */

import React, { useMemo } from 'react';
import { View, Text, Pressable, ScrollView, StyleSheet } from 'react-native';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface QuickAccessItem {
  id: string;
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  source: 'history' | 'frequent' | 'favourite' | 'recipe';
}

export type QuickAccessTabKey = 'recent' | 'frequent' | 'favorites' | 'recipes';

export interface QuickAccessTabsProps {
  activeTab: QuickAccessTabKey;
  onTabChange: (tab: QuickAccessTabKey) => void;
  items: QuickAccessItem[];
  onItemPress: (item: QuickAccessItem) => void;
}

// ---------------------------------------------------------------------------
// Tab configuration
// ---------------------------------------------------------------------------

const TABS: { key: QuickAccessTabKey; label: string }[] = [
  { key: 'recent', label: 'Recent' },
  { key: 'frequent', label: 'Frequent' },
  { key: 'favorites', label: 'Favorites' },
  { key: 'recipes', label: 'My Recipes' },
];

const EMPTY_STATES: Record<QuickAccessTabKey, { title: string; subtitle: string }> = {
  recent: {
    title: 'No recent foods',
    subtitle: 'Foods you log will appear here for quick re-logging.',
  },
  frequent: {
    title: 'No frequent foods',
    subtitle: 'Foods you log often will appear here.',
  },
  favorites: {
    title: 'No favorites yet',
    subtitle: 'Long-press any food item and tap Save as Favorite.',
  },
  recipes: {
    title: 'No saved recipes',
    subtitle: 'Save a meal as a recipe from the item detail sheet.',
  },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function QuickAccessTabs({
  activeTab,
  onTabChange,
  items,
  onItemPress,
}: QuickAccessTabsProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);
  const emptyState = EMPTY_STATES[activeTab];

  return (
    <View style={styles.container}>
      {/* Tab row */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.tabRow}
      >
        {TABS.map((tab) => {
          const isActive = tab.key === activeTab;
          return (
            <Pressable
              key={tab.key}
              onPress={() => onTabChange(tab.key)}
              style={[styles.tab, isActive ? styles.tabActive : styles.tabInactive]}
              accessibilityRole="tab"
              accessibilityState={{ selected: isActive }}
            >
              <Text style={[styles.tabText, isActive ? styles.tabTextActive : styles.tabTextInactive]}>
                {tab.label}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>

      {/* Item list */}
      {items.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyTitle}>{emptyState.title}</Text>
          <Text style={styles.emptySubtitle}>{emptyState.subtitle}</Text>
        </View>
      ) : (
        <ScrollView style={styles.itemList} showsVerticalScrollIndicator={false}>
          {items.map((item) => (
            <Pressable
              key={item.id}
              onPress={() => onItemPress(item)}
              style={styles.itemRow}
              accessibilityRole="button"
            >
              <View style={styles.itemLeft}>
                <Text style={styles.itemName} numberOfLines={1}>
                  {item.name}
                </Text>
              </View>
              <Text style={styles.itemCalories}>{item.calories} cal</Text>
            </Pressable>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      flex: 1,
    },
    tabRow: {
      flexDirection: 'row',
      gap: 8,
      paddingVertical: 8,
    },
    tab: {
      height: 36,
      paddingHorizontal: 16,
      borderRadius: 20,
      justifyContent: 'center',
      alignItems: 'center',
    },
    tabActive: {
      backgroundColor: colors.accent.green,
    },
    tabInactive: {
      backgroundColor: colors.background.surface,
    },
    tabText: {
      fontSize: 14,
      fontWeight: '600',
    },
    tabTextActive: {
      color: colors.text.inverse,
    },
    tabTextInactive: {
      color: colors.text.secondary,
    },
    itemList: {
      flex: 1,
      marginTop: 8,
    },
    itemRow: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: colors.background.elevated,
      borderRadius: 10,
      paddingHorizontal: 16,
      paddingVertical: 14,
      marginBottom: 4,
    },
    itemLeft: {
      flex: 1,
      marginRight: 12,
    },
    itemName: {
      fontSize: 16,
      fontWeight: '600',
      color: colors.text.primary,
    },
    itemCalories: {
      fontSize: 14,
      color: colors.text.secondary,
      fontWeight: '500',
    },
    emptyContainer: {
      alignItems: 'center',
      paddingVertical: 32,
      paddingHorizontal: 20,
    },
    emptyTitle: {
      fontSize: 15,
      fontWeight: '600',
      color: colors.text.tertiary,
      marginBottom: 6,
    },
    emptySubtitle: {
      fontSize: 13,
      color: colors.text.tertiary,
      textAlign: 'center',
      lineHeight: 18,
    },
  });
}
