import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface LogMealFABProps {
  itemCount: number;
  onPress: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Floating action button positioned at bottom-right with item count badge.
 *
 * Per locked decision: "Log Meal is a floating action button (FAB) at
 * bottom-right with item count badge."
 */
export function LogMealFAB({ itemCount, onPress }: LogMealFABProps) {
  const isDisabled = itemCount === 0;

  return (
    <Pressable
      onPress={onPress}
      disabled={isDisabled}
      style={[styles.fab, isDisabled && styles.fabDisabled]}
    >
      <Text style={[styles.fabText, isDisabled && styles.fabTextDisabled]}>
        Log Meal
      </Text>

      {itemCount > 0 && (
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{itemCount}</Text>
        </View>
      )}
    </Pressable>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  fab: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    height: 56,
    paddingHorizontal: 24,
    borderRadius: 28,
    backgroundColor: '#22C55E',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  fabDisabled: {
    backgroundColor: '#A5D6A7',
  },
  fabText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  fabTextDisabled: {
    color: 'rgba(255,255,255,0.6)',
  },
  badge: {
    position: 'absolute',
    top: -4,
    right: -4,
    minWidth: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#EF4444',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 4,
  },
  badgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
});
