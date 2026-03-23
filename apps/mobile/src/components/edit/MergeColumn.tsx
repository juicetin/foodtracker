/**
 * MergeColumn -- Scrollable column (discard/keep) for merge view.
 *
 * Left column: "Discard" with red header.
 * Right column: "Keep" with green header.
 * Contains a vertical ScrollView of DraggableItem cards.
 */

import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { DraggableItem } from './DraggableItem';
import type { MergeItem } from '../../services/entryEditor/reidentifyService';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface MergeColumnProps {
  title: string;
  items: MergeItem[];
  side: 'left' | 'right';
  onItemDraggedOut: (item: MergeItem, targetSide: 'left' | 'right') => void;
  headerColor: string;
}

export function MergeColumn({
  title,
  items,
  side,
  onItemDraggedOut,
  headerColor,
}: MergeColumnProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  return (
    <View style={styles.column}>
      <View style={[styles.header, { backgroundColor: headerColor }]}>
        <Text style={styles.headerTitle}>{title}</Text>
        <View style={styles.countBadge}>
          <Text style={styles.countText}>{items.length}</Text>
        </View>
      </View>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        nestedScrollEnabled
      >
        {items.length === 0 && (
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>
              {side === 'left' ? 'Drag items here to discard' : 'Drag items here to keep'}
            </Text>
          </View>
        )}
        {items.map((item) => (
          <DraggableItem
            key={item.id}
            item={item}
            currentSide={side}
            onDragComplete={onItemDraggedOut}
          />
        ))}
      </ScrollView>
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    column: {
      flex: 1,
    },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderRadius: 10,
      marginBottom: 8,
    },
    headerTitle: {
      fontSize: 13,
      fontWeight: '700',
      color: '#FFF',
    },
    countBadge: {
      backgroundColor: 'rgba(255,255,255,0.3)',
      borderRadius: 10,
      paddingHorizontal: 8,
      paddingVertical: 2,
    },
    countText: {
      fontSize: 12,
      fontWeight: '700',
      color: '#FFF',
    },
    scroll: {
      flex: 1,
    },
    scrollContent: {
      paddingBottom: 16,
    },
    emptyState: {
      paddingVertical: 40,
      alignItems: 'center',
    },
    emptyText: {
      fontSize: 12,
      color: colors.text.tertiary,
      textAlign: 'center',
    },
  });
}
