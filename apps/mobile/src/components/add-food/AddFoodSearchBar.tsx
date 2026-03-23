/**
 * AddFoodSearchBar -- search input with camera, voice, and barcode action icons.
 *
 * Renders a TextInput with a search icon on the left and action icons
 * (camera, mic, barcode) on the right. Each icon has a 44x44px minimum
 * touch target per accessibility guidelines.
 */

import React, { useMemo } from 'react';
import { View, TextInput, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

export interface AddFoodSearchBarProps {
  value: string;
  onChangeText: (text: string) => void;
  onCameraPress: () => void;
  onVoicePress: () => void;
  onBarcodePress: () => void;
  onSubmit: () => void;
}

export function AddFoodSearchBar({
  value,
  onChangeText,
  onCameraPress,
  onVoicePress,
  onBarcodePress,
  onSubmit,
}: AddFoodSearchBarProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  return (
    <View style={styles.container}>
      <Ionicons name="search-outline" size={20} color={colors.text.tertiary} style={styles.searchIcon} />
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={onChangeText}
        placeholder="Search food..."
        placeholderTextColor={colors.input.placeholder}
        returnKeyType="search"
        onSubmitEditing={onSubmit}
      />
      <View style={styles.iconsRow}>
        <Pressable
          onPress={onCameraPress}
          style={styles.iconButton}
          accessibilityLabel="Take photo"
          accessibilityRole="button"
        >
          <Ionicons name="camera-outline" size={22} color={colors.text.tertiary} />
        </Pressable>
        <Pressable
          onPress={onVoicePress}
          style={styles.iconButton}
          accessibilityLabel="Voice input"
          accessibilityRole="button"
        >
          <Ionicons name="mic-outline" size={22} color={colors.text.tertiary} />
        </Pressable>
        <Pressable
          onPress={onBarcodePress}
          style={styles.iconButton}
          accessibilityLabel="Scan barcode"
          accessibilityRole="button"
        >
          <Ionicons name="barcode-outline" size={22} color={colors.text.tertiary} />
        </Pressable>
      </View>
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: colors.input.background,
      borderRadius: 12,
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderWidth: 1,
      borderColor: colors.input.border,
    },
    searchIcon: {
      marginRight: 8,
    },
    input: {
      flex: 1,
      fontSize: 16,
      color: colors.text.primary,
      paddingVertical: 4,
    },
    iconsRow: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 2,
    },
    iconButton: {
      minWidth: 44,
      minHeight: 44,
      justifyContent: 'center',
      alignItems: 'center',
    },
  });
}
