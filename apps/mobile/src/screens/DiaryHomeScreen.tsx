/**
 * DiaryHomeScreen -- diary-first home screen with meal-type grouping.
 *
 * Replaces the separate Home + Diary screens with a unified view.
 * Shows macro summary, date navigation, week overview, and meal groups.
 */

import React, { useCallback, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, RefreshControl, SafeAreaView } from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { runOnJS } from 'react-native-reanimated';
import type { RootStackParamList } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import {
  loadEntriesGroupedByMeal,
  MEAL_GROUPS,
  type MealGroup,
  computeMealGroupTotals,
} from '../services/diary/mealGroups';
import {
  computeDayTotals,
  getTodayDateStr,
  dateToStr,
  loadWeekEntryPresence,
  type DiaryEntry,
} from '../services/diary/diaryQueries';
import {
  MacroSummaryHeader,
  DateNavigator,
  CalendarPicker,
  MealGroupSection,
  WeekOverviewBar,
} from '../components/diary';

const SWIPE_THRESHOLD = 50;

export default function DiaryHomeScreen() {
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { nutritionGoals } = usePreferencesStore();

  // State
  const [selectedDate, setSelectedDate] = useState(getTodayDateStr());
  const [groupedEntries, setGroupedEntries] = useState<Map<MealGroup, DiaryEntry[]>>(new Map());
  const [weekPresence, setWeekPresence] = useState<Map<string, number>>(new Map());
  const [refreshing, setRefreshing] = useState(false);
  const [calendarVisible, setCalendarVisible] = useState(false);

  // Plan 04 will use these for bottom sheets
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [contextMenuEntry, setContextMenuEntry] = useState<DiaryEntry | null>(null);
  const [menuMealGroup, setMenuMealGroup] = useState<MealGroup | null>(null);

  const isToday = selectedDate === getTodayDateStr();

  // Flatten all entries for day totals
  const allEntries: DiaryEntry[] = [];
  groupedEntries.forEach((entries) => allEntries.push(...entries));
  const dayTotals = computeDayTotals(allEntries);

  // Check if all groups are empty
  const hasAnyEntries = allEntries.length > 0;

  // Data loading
  const refresh = useCallback(() => {
    setGroupedEntries(loadEntriesGroupedByMeal(selectedDate));
    setWeekPresence(loadWeekEntryPresence(selectedDate));
  }, [selectedDate]);

  useFocusEffect(
    useCallback(() => {
      refresh();
    }, [refresh]),
  );

  // Date navigation
  const goToPreviousDay = useCallback(() => {
    setSelectedDate((prev) => {
      const d = new Date(prev + 'T12:00:00');
      d.setDate(d.getDate() - 1);
      return dateToStr(d);
    });
  }, []);

  const goToNextDay = useCallback(() => {
    setSelectedDate((prev) => {
      const d = new Date(prev + 'T12:00:00');
      d.setDate(d.getDate() + 1);
      const next = dateToStr(d);
      return next > getTodayDateStr() ? prev : next;
    });
  }, []);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    refresh();
    setRefreshing(false);
  }, [refresh]);

  // Swipe gesture for date navigation
  const swipeGesture = Gesture.Pan()
    .activeOffsetX([-20, 20])
    .failOffsetY([-10, 10])
    .onEnd((event) => {
      if (event.translationX > SWIPE_THRESHOLD) {
        runOnJS(goToPreviousDay)();
      } else if (event.translationX < -SWIPE_THRESHOLD) {
        runOnJS(goToNextDay)();
      }
    });

  // Handlers
  const handleAddFood = useCallback(
    (mealGroup: MealGroup) => {
      nav.navigate('AddFood', { mealType: mealGroup });
    },
    [nav],
  );

  const handleItemPress = useCallback((entryId: string) => {
    setSelectedEntryId(entryId);
    // Plan 04: ItemDetailSheet will open here
  }, []);

  const handleItemLongPress = useCallback((entry: DiaryEntry) => {
    setContextMenuEntry(entry);
    // Plan 04: ContextMenuSheet will open here
  }, []);

  const handleHeaderLongPress = useCallback((mealGroup: MealGroup) => {
    setMenuMealGroup(mealGroup);
    // Plan 04: MealGroupMenuSheet will open here
  }, []);

  return (
    <SafeAreaView style={styles.screen}>
      {/* Date navigator */}
      <DateNavigator
        dateStr={selectedDate}
        isToday={isToday}
        onPrevious={goToPreviousDay}
        onNext={goToNextDay}
        onDateTap={() => setCalendarVisible(true)}
      />

      {/* Macro summary header */}
      <MacroSummaryHeader totals={dayTotals} goals={nutritionGoals} />

      {/* Week overview bar */}
      <WeekOverviewBar
        selectedDate={selectedDate}
        onSelectDate={setSelectedDate}
        entryPresence={weekPresence}
      />

      {/* Scrollable meal groups */}
      <GestureDetector gesture={swipeGesture}>
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor="#16A34A"
            />
          }
        >
          {MEAL_GROUPS.map((group) => (
            <MealGroupSection
              key={group}
              mealGroup={group}
              entries={groupedEntries.get(group) ?? []}
              onAddFood={handleAddFood}
              onItemPress={handleItemPress}
              onItemLongPress={handleItemLongPress}
              onHeaderLongPress={handleHeaderLongPress}
            />
          ))}

          {/* Empty state */}
          {!hasAnyEntries && (
            <View style={styles.emptyState}>
              <Text style={styles.emptyText}>No meals logged</Text>
              <Text style={styles.emptySubtext}>
                Tap + to add your first meal, or take a photo of your food.
              </Text>
            </View>
          )}

          <View style={{ height: 100 }} />
        </ScrollView>
      </GestureDetector>

      {/* Calendar picker modal */}
      <CalendarPicker
        visible={calendarVisible}
        selectedDate={selectedDate}
        onSelect={(dateStr) => setSelectedDate(dateStr)}
        onDismiss={() => setCalendarVisible(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: 8,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6B7280',
    marginBottom: 6,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    paddingHorizontal: 32,
  },
});
