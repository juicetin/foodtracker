/**
 * PhotoViewer -- full-screen modal photo viewer with pinch-to-zoom.
 *
 * Uses react-native-gesture-handler Gesture.Pinch + Gesture.Pan
 * and react-native-reanimated for smooth animated zoom/pan.
 * Horizontal FlatList for swiping between multiple photos.
 * Page indicator dots at bottom when multiple photos.
 */

import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  Pressable,
  FlatList,
  Dimensions,
  Image,
} from 'react-native';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PhotoViewerProps {
  photos: Array<{ uri: string }>;
  initialIndex: number;
  visible: boolean;
  onClose: () => void;
}

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ---------------------------------------------------------------------------
// Single photo page with pinch-to-zoom + pan
// ---------------------------------------------------------------------------

function ZoomablePhoto({ uri }: { uri: string }) {
  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);

  const pinchGesture = Gesture.Pinch()
    .onUpdate((e) => {
      scale.value = savedScale.value * e.scale;
    })
    .onEnd(() => {
      if (scale.value < 1) {
        scale.value = withTiming(1);
        savedScale.value = 1;
        translateX.value = withTiming(0);
        translateY.value = withTiming(0);
        savedTranslateX.value = 0;
        savedTranslateY.value = 0;
      } else {
        savedScale.value = scale.value;
      }
    });

  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      if (savedScale.value > 1) {
        translateX.value = savedTranslateX.value + e.translationX;
        translateY.value = savedTranslateY.value + e.translationY;
      }
    })
    .onEnd(() => {
      savedTranslateX.value = translateX.value;
      savedTranslateY.value = translateY.value;
    });

  const composedGesture = Gesture.Simultaneous(pinchGesture, panGesture);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  return (
    <GestureDetector gesture={composedGesture}>
      <Animated.View style={[styles.photoPage, animatedStyle]}>
        <Image
          source={{ uri }}
          style={styles.fullImage}
          resizeMode="contain"
        />
      </Animated.View>
    </GestureDetector>
  );
}

// ---------------------------------------------------------------------------
// PhotoViewer component
// ---------------------------------------------------------------------------

export function PhotoViewer({ photos, initialIndex, visible, onClose }: PhotoViewerProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);

  const handleScroll = useCallback((e: { nativeEvent: { contentOffset: { x: number } } }) => {
    const index = Math.round(e.nativeEvent.contentOffset.x / SCREEN_WIDTH);
    setCurrentIndex(index);
  }, []);

  if (!visible || photos.length === 0) return null;

  return (
    <Modal
      visible={visible}
      transparent={false}
      animationType="fade"
      onRequestClose={onClose}
      statusBarTranslucent
    >
      <View style={styles.container}>
        {/* Close button */}
        <Pressable style={styles.closeBtn} onPress={onClose}>
          <Ionicons name="close" size={28} color="#FFF" />
        </Pressable>

        {/* Photo carousel */}
        <FlatList
          data={photos}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          initialScrollIndex={initialIndex}
          getItemLayout={(_, index) => ({
            length: SCREEN_WIDTH,
            offset: SCREEN_WIDTH * index,
            index,
          })}
          onMomentumScrollEnd={handleScroll}
          keyExtractor={(item, index) => `${item.uri}-${index}`}
          renderItem={({ item }) => (
            <View style={{ width: SCREEN_WIDTH, height: SCREEN_HEIGHT }}>
              <ZoomablePhoto uri={item.uri} />
            </View>
          )}
        />

        {/* Page indicator dots */}
        {photos.length > 1 && (
          <View style={styles.dotsRow}>
            {photos.map((_, i) => (
              <View
                key={i}
                style={[
                  styles.dot,
                  i === currentIndex && styles.dotActive,
                ]}
              />
            ))}
          </View>
        )}
      </View>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1, backgroundColor: '#000',
  },
  closeBtn: {
    position: 'absolute', top: 50, right: 20, zIndex: 10,
    width: 44, height: 44, borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center', alignItems: 'center',
  },
  photoPage: {
    width: SCREEN_WIDTH, height: SCREEN_HEIGHT,
    justifyContent: 'center', alignItems: 'center',
  },
  fullImage: {
    width: SCREEN_WIDTH, height: SCREEN_HEIGHT,
  },
  dotsRow: {
    position: 'absolute', bottom: 40,
    flexDirection: 'row', alignSelf: 'center', gap: 8,
  },
  dot: {
    width: 8, height: 8, borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.4)',
  },
  dotActive: {
    backgroundColor: '#FFF',
  },
});
