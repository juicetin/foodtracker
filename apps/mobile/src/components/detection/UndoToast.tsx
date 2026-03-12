import React, { useEffect, useRef } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  SlideInDown,
  SlideOutDown,
} from 'react-native-reanimated';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface UndoToastProps {
  itemName: string;
  onUndo: () => void;
  visible: boolean;
  onDismiss: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Auto-dismiss timeout in milliseconds. */
const AUTO_DISMISS_MS = 5000;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Animated undo toast that slides up from the bottom of the screen.
 *
 * Per locked decision: "Removal must be undoable -- show undo toast after
 * dismissal, item restored if tapped within timeout."
 *
 * Auto-dismisses after 5 seconds. Only one toast visible at a time (parent
 * controls visibility by passing a new itemName).
 */
export function UndoToast({
  itemName,
  onUndo,
  visible,
  onDismiss,
}: UndoToastProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (visible) {
      // Clear any existing timer
      if (timerRef.current) clearTimeout(timerRef.current);

      timerRef.current = setTimeout(() => {
        onDismiss();
      }, AUTO_DISMISS_MS);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [visible, itemName, onDismiss]);

  if (!visible) return null;

  return (
    <Animated.View
      entering={SlideInDown.duration(250)}
      exiting={SlideOutDown.duration(200)}
      style={styles.container}
    >
      <View style={styles.content}>
        <Text style={styles.message} numberOfLines={1}>
          {itemName} removed
        </Text>
        <Pressable
          onPress={() => {
            if (timerRef.current) clearTimeout(timerRef.current);
            onUndo();
          }}
          style={styles.undoButton}
        >
          <Text style={styles.undoText}>Undo</Text>
        </Pressable>
      </View>
    </Animated.View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 90, // Above the FAB
    left: 16,
    right: 16,
    zIndex: 100,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#323232',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  message: {
    color: '#fff',
    fontSize: 14,
    flex: 1,
    marginRight: 12,
  },
  undoButton: {
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  undoText: {
    color: '#8BC34A',
    fontSize: 14,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
});
