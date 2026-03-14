/**
 * Free-form text input for meal description.
 *
 * Allows the user to type a meal description (e.g., "pad thai with shrimp")
 * which is injected into the VLM prompt for disambiguation.
 *
 * Returns null when disabled (hidden during detecting/idle states).
 */

import React from 'react';
import { StyleSheet, TextInput, View } from 'react-native';

interface MealTextInputProps {
  value: string;
  onChangeText: (text: string) => void;
  disabled?: boolean;
}

export function MealTextInput({ value, onChangeText, disabled }: MealTextInputProps) {
  if (disabled) return null;

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={onChangeText}
        placeholder="Describe your meal (optional)"
        placeholderTextColor="#999"
        returnKeyType="done"
        autoCorrect={false}
        autoCapitalize="none"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  input: {
    height: 40,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#D0D0D0',
    borderRadius: 8,
    paddingHorizontal: 12,
    fontSize: 14,
    color: '#333',
    backgroundColor: '#FAFAFA',
  },
});
