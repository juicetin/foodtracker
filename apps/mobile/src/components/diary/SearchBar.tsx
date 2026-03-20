/**
 * SearchBar — tappable search bar for the DiaryScreen.
 *
 * Not an editable TextInput; navigates to FoodSearchScreen on press.
 * Barcode icon is a no-op per user decision.
 */

import React from 'react';
import { Pressable, Text, View, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface SearchBarProps {
  onSearchPress: () => void;
}

export function SearchBar({ onSearchPress }: SearchBarProps) {
  return (
    <Pressable style={styles.container} onPress={onSearchPress}>
      <Ionicons name="search" size={18} color="#9CA3AF" />
      <Text style={styles.placeholder}>Search foods...</Text>
      <View style={styles.spacer} />
      <Pressable hitSlop={8}>
        <Ionicons name="barcode-outline" size={20} color="#9CA3AF" />
      </Pressable>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginHorizontal: 16,
    marginBottom: 12,
  },
  placeholder: {
    flex: 0,
    marginLeft: 10,
    fontSize: 15,
    color: '#9CA3AF',
  },
  spacer: {
    flex: 1,
  },
});
