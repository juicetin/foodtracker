/**
 * Free-form text input for meal description.
 *
 * Allows the user to type a meal description (e.g., "pad thai with shrimp")
 * which is injected into the VLM prompt for disambiguation.
 *
 * Returns null when disabled (hidden during detecting/idle states).
 */

import React, { useMemo } from 'react';
import { StyleSheet, TextInput, View } from 'react-native';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface MealTextInputProps {
  value: string;
  onChangeText: (text: string) => void;
  disabled?: boolean;
}

export function MealTextInput({ value, onChangeText, disabled }: MealTextInputProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  if (disabled) return null;

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={onChangeText}
        placeholder="Describe your meal (optional)"
        placeholderTextColor={colors.input.placeholder}
        returnKeyType="done"
        autoCorrect={false}
        autoCapitalize="none"
      />
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      paddingHorizontal: 16,
      paddingVertical: 8,
    },
    input: {
      height: 40,
      borderWidth: StyleSheet.hairlineWidth,
      borderColor: colors.input.border,
      borderRadius: 8,
      paddingHorizontal: 12,
      fontSize: 14,
      color: colors.text.primary,
      backgroundColor: colors.input.background,
    },
  });
}
