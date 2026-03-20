/**
 * WeekOverviewBar — 7-day bar showing entry presence per day (Mon-Sun).
 *
 * Filled green circles for days with entries, hollow gray for empty days.
 * Selected date gets a thicker green border. Today gets a small dot below.
 */

import React, { useMemo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { dateToStr } from '../../services/diary/diaryQueries';

const DAY_LABELS = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

interface WeekOverviewBarProps {
  selectedDate: string;
  onSelectDate: (dateStr: string) => void;
  entryPresence: Map<string, number>;
}

export function WeekOverviewBar({
  selectedDate,
  onSelectDate,
  entryPresence,
}: WeekOverviewBarProps) {
  const todayStr = useMemo(() => new Date().toISOString().split('T')[0], []);

  const weekDates = useMemo(() => {
    const d = new Date(selectedDate + 'T12:00:00');
    const dayOfWeek = d.getDay(); // 0=Sun, 1=Mon, ...
    const mondayOffset = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
    const monday = new Date(d);
    monday.setDate(d.getDate() + mondayOffset);

    const dates: string[] = [];
    for (let i = 0; i < 7; i++) {
      const day = new Date(monday);
      day.setDate(monday.getDate() + i);
      dates.push(dateToStr(day));
    }
    return dates;
  }, [selectedDate]);

  return (
    <View style={styles.container}>
      {weekDates.map((dateStr, i) => {
        const hasEntries = entryPresence.has(dateStr);
        const isSelected = dateStr === selectedDate;
        const isToday = dateStr === todayStr;

        return (
          <Pressable key={dateStr} onPress={() => onSelectDate(dateStr)} style={styles.dayColumn}>
            <View
              style={[
                styles.circle,
                hasEntries && styles.circleFilled,
                isSelected && styles.circleSelected,
              ]}
            >
              <Text
                style={[
                  styles.dayLabel,
                  hasEntries && styles.dayLabelFilled,
                  isSelected && styles.dayLabelSelected,
                ]}
              >
                {DAY_LABELS[i]}
              </Text>
            </View>
            {isToday && <View style={styles.todayDot} />}
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 8,
  },
  dayColumn: {
    alignItems: 'center',
  },
  circle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
  },
  circleFilled: {
    backgroundColor: '#16A34A',
    borderColor: '#16A34A',
  },
  circleSelected: {
    borderWidth: 2.5,
    borderColor: '#16A34A',
  },
  dayLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9CA3AF',
  },
  dayLabelFilled: {
    color: '#FFFFFF',
  },
  dayLabelSelected: {
    color: '#16A34A',
  },
  todayDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#16A34A',
    marginTop: 4,
  },
});
