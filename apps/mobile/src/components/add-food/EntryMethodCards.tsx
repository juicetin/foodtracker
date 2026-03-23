/**
 * EntryMethodCards -- 2x2 grid of food entry method cards.
 *
 * Provides quick access to: Scan Photo, Scan Barcode, Quick Add Macros,
 * and From Gallery. Each card has an icon and label.
 */

import React, { useMemo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

export interface EntryMethodCardsProps {
  onScanPhoto: () => void;
  onScanBarcode: () => void;
  onQuickAdd: () => void;
  onFromGallery: () => void;
}

interface CardConfig {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
}

export function EntryMethodCards({
  onScanPhoto,
  onScanBarcode,
  onQuickAdd,
  onFromGallery,
}: EntryMethodCardsProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const cards: CardConfig[] = [
    { label: 'Scan Photo', icon: 'camera-outline', onPress: onScanPhoto },
    { label: 'Scan Barcode', icon: 'barcode-outline', onPress: onScanBarcode },
    { label: 'Quick Add Macros', icon: 'calculator-outline', onPress: onQuickAdd },
    { label: 'From Gallery', icon: 'images-outline', onPress: onFromGallery },
  ];

  return (
    <View style={styles.grid}>
      {cards.map((card) => (
        <Pressable
          key={card.label}
          onPress={card.onPress}
          style={styles.card}
          accessibilityRole="button"
          accessibilityLabel={card.label}
        >
          <Ionicons name={card.icon} size={32} color={colors.accent.green} />
          <Text style={styles.cardLabel}>{card.label}</Text>
        </Pressable>
      ))}
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    grid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12,
    },
    card: {
      width: '47%' as any,
      backgroundColor: colors.background.elevated,
      borderRadius: 12,
      padding: 16,
      alignItems: 'center',
      justifyContent: 'center',
      elevation: 1,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 4,
      flexGrow: 1,
      flexBasis: '45%',
    },
    cardLabel: {
      fontSize: 14,
      fontWeight: '600',
      color: colors.text.secondary,
      textAlign: 'center',
      marginTop: 8,
    },
  });
}
