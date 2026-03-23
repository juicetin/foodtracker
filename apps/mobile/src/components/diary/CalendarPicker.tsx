/**
 * CalendarPicker -- modal calendar for date selection.
 *
 * Custom month grid with month navigation, selected/today highlighting.
 */

import React, { useMemo, useState } from 'react';
import { View, Text, Pressable, Modal, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { dateToStr, getTodayDateStr } from '../../services/diary/diaryQueries';
import { useTheme } from '../../theme/ThemeProvider';
import type { ThemeColors } from '../../theme/colors';

interface CalendarPickerProps {
  visible: boolean;
  selectedDate: string;
  onSelect: (dateStr: string) => void;
  onDismiss: () => void;
}

const DAY_LABELS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

export function CalendarPicker({
  visible,
  selectedDate,
  onSelect,
  onDismiss,
}: CalendarPickerProps) {
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  const [displayMonth, setDisplayMonth] = useState(() => {
    const d = new Date(selectedDate + 'T12:00:00');
    return new Date(d.getFullYear(), d.getMonth(), 1);
  });

  const todayStr = getTodayDateStr();

  const goToPrevMonth = () => {
    setDisplayMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1));
  };

  const goToNextMonth = () => {
    setDisplayMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1));
  };

  // Generate 42 cells (6 rows x 7 cols)
  const cells = useMemo(() => {
    const year = displayMonth.getFullYear();
    const month = displayMonth.getMonth();
    const firstDayOfMonth = new Date(year, month, 1).getDay(); // 0=Sun
    const daysInMonth = new Date(year, month + 1, 0).getDate();

    const result: Array<{ date: Date; dateStr: string; isCurrentMonth: boolean }> = [];

    // Previous month trailing days
    const prevMonthDays = new Date(year, month, 0).getDate();
    for (let i = firstDayOfMonth - 1; i >= 0; i--) {
      const d = new Date(year, month - 1, prevMonthDays - i);
      result.push({ date: d, dateStr: dateToStr(d), isCurrentMonth: false });
    }

    // Current month days
    for (let day = 1; day <= daysInMonth; day++) {
      const d = new Date(year, month, day);
      result.push({ date: d, dateStr: dateToStr(d), isCurrentMonth: true });
    }

    // Next month leading days (fill to 42)
    const remaining = 42 - result.length;
    for (let i = 1; i <= remaining; i++) {
      const d = new Date(year, month + 1, i);
      result.push({ date: d, dateStr: dateToStr(d), isCurrentMonth: false });
    }

    return result;
  }, [displayMonth]);

  const monthLabel = `${MONTH_NAMES[displayMonth.getMonth()]} ${displayMonth.getFullYear()}`;

  const handleSelect = (dateStr: string) => {
    onSelect(dateStr);
    onDismiss();
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onDismiss}
    >
      <Pressable style={styles.backdrop} onPress={onDismiss}>
        <Pressable style={styles.modal} onPress={() => {}}>
          {/* Month header */}
          <View style={styles.monthHeader}>
            <Pressable onPress={goToPrevMonth} style={styles.monthArrow}>
              <Ionicons name="chevron-back" size={20} color={colors.text.secondary} />
            </Pressable>
            <Text style={styles.monthLabel}>{monthLabel}</Text>
            <Pressable onPress={goToNextMonth} style={styles.monthArrow}>
              <Ionicons name="chevron-forward" size={20} color={colors.text.secondary} />
            </Pressable>
          </View>

          {/* Day of week labels */}
          <View style={styles.dayLabelsRow}>
            {DAY_LABELS.map((label, i) => (
              <View key={i} style={styles.dayLabelCell}>
                <Text style={styles.dayLabelText}>{label}</Text>
              </View>
            ))}
          </View>

          {/* Month grid */}
          <View style={styles.grid}>
            {cells.map((cell, i) => {
              const isSelected = cell.dateStr === selectedDate;
              const isToday = cell.dateStr === todayStr;

              return (
                <Pressable
                  key={i}
                  style={styles.dayCell}
                  onPress={() => handleSelect(cell.dateStr)}
                >
                  <View style={[styles.dayCellInner, isSelected && styles.dayCellSelected]}>
                    <Text
                      style={[
                        styles.dayText,
                        !cell.isCurrentMonth && styles.dayTextOtherMonth,
                        isToday && !isSelected && styles.dayTextToday,
                        isSelected && styles.dayTextSelected,
                      ]}
                    >
                      {cell.date.getDate()}
                    </Text>
                  </View>
                </Pressable>
              );
            })}
          </View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    backdrop: {
      flex: 1,
      backgroundColor: colors.overlay,
      justifyContent: 'center',
      alignItems: 'center',
    },
    modal: {
      backgroundColor: colors.background.elevated,
      borderRadius: 16,
      padding: 16,
      elevation: 8,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 12,
      width: 320,
    },
    monthHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 12,
    },
    monthArrow: {
      minWidth: 36,
      minHeight: 36,
      alignItems: 'center',
      justifyContent: 'center',
    },
    monthLabel: {
      fontSize: 16,
      fontWeight: '600',
      color: colors.text.primary,
    },
    dayLabelsRow: {
      flexDirection: 'row',
      marginBottom: 4,
    },
    dayLabelCell: {
      flex: 1,
      alignItems: 'center',
      paddingVertical: 4,
    },
    dayLabelText: {
      fontSize: 12,
      color: colors.text.tertiary,
      fontWeight: '500',
    },
    grid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
    },
    dayCell: {
      width: '14.28%',
      alignItems: 'center',
      paddingVertical: 2,
    },
    dayCellInner: {
      width: 40,
      height: 40,
      borderRadius: 20,
      alignItems: 'center',
      justifyContent: 'center',
    },
    dayCellSelected: {
      backgroundColor: colors.accent.green,
    },
    dayText: {
      fontSize: 14,
      color: colors.text.primary,
    },
    dayTextOtherMonth: {
      color: colors.border.default,
    },
    dayTextToday: {
      color: colors.accent.green,
      textDecorationLine: 'underline',
    },
    dayTextSelected: {
      color: colors.text.inverse,
      fontWeight: '600',
    },
  });
}
