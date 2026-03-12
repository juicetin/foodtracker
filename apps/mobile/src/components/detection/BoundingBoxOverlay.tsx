import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import {
  CONFIDENCE_COLORS,
  DetectedItem,
  getConfidenceLevel,
} from '../../services/detection/types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface BoundingBoxOverlayProps {
  items: DetectedItem[];
  photoWidth: number;
  photoHeight: number;
  displayWidth: number;
  displayHeight: number;
  selectedItemId: string | null;
  onSelectItem: (id: string) => void;
  onRemoveItem: (id: string) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Renders confidence-colored bounding boxes with floating label chips.
 *
 * Uses View-based absolute positioning (react-native-svg is not installed).
 * Normalised bbox coordinates (0-1) are scaled to display dimensions.
 */
export function BoundingBoxOverlay({
  items,
  photoWidth,
  photoHeight,
  displayWidth,
  displayHeight,
  selectedItemId,
  onSelectItem,
  onRemoveItem,
}: BoundingBoxOverlayProps) {
  const scaleX = displayWidth / (photoWidth || 1);
  const scaleY = displayHeight / (photoHeight || 1);

  const activeItems = items.filter((item) => !item.isRemoved);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
      {activeItems.map((item) => {
        const level = getConfidenceLevel(item.confidence);
        const color = CONFIDENCE_COLORS[level];
        const isSelected = item.id === selectedItemId;

        // Convert normalised (0-1) coords to display pixels
        const left = item.bbox.x * photoWidth * scaleX;
        const top = item.bbox.y * photoHeight * scaleY;
        const width = item.bbox.w * photoWidth * scaleX;
        const height = item.bbox.h * photoHeight * scaleY;

        const borderWidth = isSelected ? 3 : 2;

        return (
          <Pressable
            key={item.id}
            onPress={() => onSelectItem(item.id)}
            style={[
              styles.box,
              {
                left,
                top,
                width,
                height,
                borderColor: color,
                borderWidth,
                opacity: isSelected ? 1 : 0.85,
              },
            ]}
          >
            {/* Floating label chip above the box */}
            <View style={[styles.labelChip, { backgroundColor: color }]}>
              <Text style={styles.labelText} numberOfLines={1}>
                {item.className} {Math.round(item.confidence * 100)}%
              </Text>
              {/* X dismiss button */}
              <Pressable
                onPress={(e) => {
                  e.stopPropagation?.();
                  onRemoveItem(item.id);
                }}
                hitSlop={8}
                style={styles.dismissButton}
              >
                <Text style={styles.dismissText}>x</Text>
              </Pressable>
            </View>
          </Pressable>
        );
      })}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderRadius: 4,
  },
  labelChip: {
    position: 'absolute',
    top: -26,
    left: -2,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
    maxWidth: 200,
  },
  labelText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
    flexShrink: 1,
  },
  dismissButton: {
    marginLeft: 4,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: 'rgba(0,0,0,0.3)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dismissText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
    lineHeight: 14,
  },
});
