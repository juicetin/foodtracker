/**
 * DateNavigator -- left/right arrows with date label, tap opens calendar.
 */

import React, { useMemo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { formatDateLabel } from '../../services/diary/diaryQueries';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface DateNavigatorProps {
  dateStr: string;
  isToday: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onDateTap: () => void;
}

export function DateNavigator({
  dateStr,
  isToday,
  onPrevious,
  onNext,
  onDateTap,
}: DateNavigatorProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  return (
    <View style={styles.container}>
      <Pressable
        onPress={onPrevious}
        style={styles.arrowButton}
        hitSlop={8}
      >
        <Ionicons name="chevron-back" size={24} color={colors.text.secondary} />
      </Pressable>

      <Pressable onPress={onDateTap} style={styles.dateLabelBtn}>
        <Text style={styles.dateText}>
          {isToday ? 'Today' : formatDateLabel(dateStr)}
        </Text>
      </Pressable>

      <Pressable
        onPress={onNext}
        style={[styles.arrowButton, isToday && { opacity: 0.3 }]}
        disabled={isToday}
        hitSlop={8}
      >
        <Ionicons name="chevron-forward" size={24} color={colors.text.secondary} />
      </Pressable>
    </View>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    container: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingHorizontal: 16,
      paddingVertical: 8,
    },
    arrowButton: {
      minWidth: 44,
      minHeight: 44,
      alignItems: 'center',
      justifyContent: 'center',
    },
    dateLabelBtn: {
      flex: 1,
      alignItems: 'center',
    },
    dateText: {
      fontSize: 20,
      fontWeight: '600',
      color: colors.text.primary,
    },
  });
}
