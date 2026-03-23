/**
 * FoodItemCard -- compact food item row with tap and long-press.
 *
 * Replaces ExpandableEntryCard. No toggle states (QA-03 fix).
 * Simple card with photo, name, calories, time, and macro pills.
 */

import React, { useCallback, useRef, useState } from 'react';
import { View, Text, Pressable, Image, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { DiaryEntry } from '../../services/diary/diaryQueries';

interface FoodItemCardProps {
  entry: DiaryEntry;
  onPress: () => void;
  onLongPress: () => void;
}

export function FoodItemCard({ entry, onPress, onLongPress }: FoodItemCardProps) {
  const [photoError, setPhotoError] = useState(false);
  const longPressedRef = useRef(false);

  const dishName = entry.dishes[0]?.name || 'Food Entry';
  const showPhoto = entry.photoUri && !photoError;

  const time = new Date(entry.createdAt).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  const handlePress = useCallback(() => {
    if (longPressedRef.current) {
      longPressedRef.current = false;
      return;
    }
    onPress();
  }, [onPress]);

  const handleLongPress = useCallback(() => {
    longPressedRef.current = true;
    onLongPress();
  }, [onLongPress]);

  return (
    <Pressable
      onPress={handlePress}
      onLongPress={handleLongPress}
      delayLongPress={500}
      style={styles.container}
    >
      {/* Photo or placeholder */}
      {showPhoto ? (
        <Image
          source={{ uri: entry.photoUri! }}
          style={styles.photo}
          resizeMode="cover"
          onError={() => setPhotoError(true)}
        />
      ) : (
        <View style={styles.photoPlaceholder}>
          <Text style={styles.placeholderEmoji}>🍽️</Text>
        </View>
      )}

      {/* Content */}
      <View style={styles.contentBlock}>
        <View style={styles.topRow}>
          <Text style={styles.dishName} numberOfLines={1} ellipsizeMode="tail">
            {dishName}
          </Text>
          <Text style={styles.calorieText}>
            {Math.round(entry.totalCalories)} kcal
          </Text>
        </View>

        <Text style={styles.timeText}>{time}</Text>

        {/* Macro pills */}
        <View style={styles.macroPills}>
          <View style={[styles.pill, { backgroundColor: '#EFF6FF' }]}>
            <Text style={[styles.pillText, { color: '#3B82F6' }]}>
              P {Math.round(entry.totalProtein)}g
            </Text>
          </View>
          <View style={[styles.pill, { backgroundColor: '#FFFBEB' }]}>
            <Text style={[styles.pillText, { color: '#D97706' }]}>
              C {Math.round(entry.totalCarbs)}g
            </Text>
          </View>
          <View style={[styles.pill, { backgroundColor: '#ECFDF5' }]}>
            <Text style={[styles.pillText, { color: '#059669' }]}>
              F {Math.round(entry.totalFat)}g
            </Text>
          </View>
        </View>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#FFFFFF',
    padding: 12,
    marginHorizontal: 16,
    marginBottom: 4,
    borderRadius: 12,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  photo: {
    width: 40,
    height: 40,
    borderRadius: 8,
    marginRight: 12,
  },
  photoPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: 8,
    marginRight: 12,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderEmoji: {
    fontSize: 20,
  },
  contentBlock: {
    flex: 1,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dishName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
    flex: 1,
    marginRight: 8,
  },
  calorieText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
  },
  timeText: {
    fontSize: 14,
    color: '#9CA3AF',
    marginTop: 2,
  },
  macroPills: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 6,
  },
  pill: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
  },
  pillText: {
    fontSize: 14,
    fontWeight: '500',
  },
});
