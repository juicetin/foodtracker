import React, { useCallback } from 'react';
import { FlatList, StyleSheet, View } from 'react-native';
import { DetectedItem } from '../../services/detection/types';
import { DetectionListItem } from './DetectionListItem';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DetectionListProps {
  items: DetectedItem[];
  selectedItemId: string | null;
  onSelectItem: (id: string) => void;
  onRemoveItem: (id: string) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Scrollable list of detected items.
 *
 * Items are expected to already be sorted by confidence (highest first) via
 * the store's `activeItems()` selector.
 *
 * Per locked decision: "Items sorted by confidence (highest first) -- no
 * grouping by tier."
 */
export function DetectionList({
  items,
  selectedItemId,
  onSelectItem,
  onRemoveItem,
}: DetectionListProps) {
  const renderItem = useCallback(
    ({ item }: { item: DetectedItem }) => (
      <DetectionListItem
        item={item}
        isSelected={item.id === selectedItemId}
        onSelect={onSelectItem}
        onRemove={onRemoveItem}
      />
    ),
    [selectedItemId, onSelectItem, onRemoveItem],
  );

  const keyExtractor = useCallback((item: DetectedItem) => item.id, []);

  return (
    <View style={styles.container}>
      <FlatList
        data={items}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
