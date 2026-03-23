/**
 * DiaryHomeScreen -- diary-first home screen with meal-type grouping.
 *
 * Replaces the separate Home + Diary screens with a unified view.
 * Shows macro summary, date navigation, week overview, and meal groups.
 * Tap item -> ItemDetailSheet, long-press item -> ContextMenuSheet,
 * long-press meal header -> MealGroupMenuSheet.
 */

import React, { useCallback, useMemo, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, RefreshControl, SafeAreaView, Alert } from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { runOnJS } from 'react-native-reanimated';
import * as Clipboard from 'expo-clipboard';
import type { RootStackParamList } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import { useFoodLogStore } from '../store/useFoodLogStore';
import {
  loadEntriesGroupedByMeal,
  MEAL_GROUPS,
  type MealGroup,
  MEAL_GROUP_CONFIG,
} from '../services/diary/mealGroups';
import {
  computeDayTotals,
  getTodayDateStr,
  dateToStr,
  loadWeekEntryPresence,
  type DiaryEntry,
} from '../services/diary/diaryQueries';
import {
  copyEntryToDate,
  moveEntryToMeal,
  copyAllEntriesFromDate,
} from '../services/diary/copyMoveService';
import { addFavourite } from '../services/favourites';
import {
  MacroSummaryHeader,
  DateNavigator,
  CalendarPicker,
  MealGroupSection,
  WeekOverviewBar,
} from '../components/diary';
import { ItemDetailSheet } from '../components/sheets/ItemDetailSheet';
import { ContextMenuSheet, type ContextMenuAction } from '../components/sheets/ContextMenuSheet';
import { MealGroupMenuSheet, type MealGroupAction } from '../components/sheets/MealGroupMenuSheet';
import { useTheme } from '../theme/ThemeProvider';
import type { ThemeColors } from '../theme/colors';

const SWIPE_THRESHOLD = 50;

export default function DiaryHomeScreen() {
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { nutritionGoals } = usePreferencesStore();
  const { deleteEntry } = useFoodLogStore();
  const { colors } = useTheme();
  const styles = useMemo(() => createStyles(colors), [colors]);

  // State
  const [selectedDate, setSelectedDate] = useState(getTodayDateStr());
  const [groupedEntries, setGroupedEntries] = useState<Map<MealGroup, DiaryEntry[]>>(new Map());
  const [weekPresence, setWeekPresence] = useState<Map<string, number>>(new Map());
  const [refreshing, setRefreshing] = useState(false);
  const [calendarVisible, setCalendarVisible] = useState(false);

  // Bottom sheet state
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [contextMenuEntry, setContextMenuEntry] = useState<DiaryEntry | null>(null);
  const [menuMealGroup, setMenuMealGroup] = useState<MealGroup | null>(null);

  // Date picker context for copy operations
  const [datePickerContext, setDatePickerContext] = useState<{
    action: string;
    entryId?: string;
    mealGroup?: MealGroup;
  } | null>(null);

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
  }, []);

  const handleItemLongPress = useCallback((entry: DiaryEntry) => {
    setContextMenuEntry(entry);
  }, []);

  const handleHeaderLongPress = useCallback((mealGroup: MealGroup) => {
    setMenuMealGroup(mealGroup);
  }, []);

  // -- Detail sheet handlers --

  const handleEdit = useCallback(
    (entryId: string) => {
      setSelectedEntryId(null);
      nav.navigate('EntryDetail', { entryId });
    },
    [nav],
  );

  const handleDelete = useCallback(
    (entryId: string) => {
      setSelectedEntryId(null);
      Alert.alert('Delete Item?', 'This will remove the item from your diary.', [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await deleteEntry(entryId);
            refresh();
          },
        },
      ]);
    },
    [deleteEntry, refresh],
  );

  // -- Context menu action handler --

  const handleContextAction = useCallback(
    (action: ContextMenuAction, entry: DiaryEntry) => {
      setContextMenuEntry(null);

      switch (action) {
        case 'copy-clipboard': {
          const dishNames = entry.dishes.map((d) => d.name).join(', ');
          const text = `${dishNames}\n${Math.round(entry.totalCalories)} cal | P${Math.round(entry.totalProtein)}g C${Math.round(entry.totalCarbs)}g F${Math.round(entry.totalFat)}g`;
          Clipboard.setStringAsync(text);
          Alert.alert('Copied', 'Meal info copied to clipboard.');
          break;
        }

        case 'copy-day': {
          // Open calendar picker to select target date
          setDatePickerContext({ action: 'copy-entry', entryId: entry.id });
          setCalendarVisible(true);
          break;
        }

        case 'move-meal': {
          const otherMeals = MEAL_GROUPS.filter((g) => g !== entry.mealType);
          Alert.alert(
            'Move to Other Meal',
            'Select the meal to move this item to:',
            [
              ...otherMeals.map((g) => ({
                text: MEAL_GROUP_CONFIG[g].label,
                onPress: () => {
                  moveEntryToMeal(entry.id, g);
                  refresh();
                },
              })),
              { text: 'Cancel', style: 'cancel' as const },
            ],
          );
          break;
        }

        case 'favorite': {
          const name = entry.dishes.length > 0
            ? entry.dishes.map((d) => d.name).join(', ')
            : 'Food Item';
          addFavourite({
            name,
            totalCalories: entry.totalCalories,
            totalProtein: entry.totalProtein,
            totalCarbs: entry.totalCarbs,
            totalFat: entry.totalFat,
          });
          Alert.alert('Saved', `"${name}" added to favourites.`);
          break;
        }

        case 'delete': {
          handleDelete(entry.id);
          break;
        }
      }
    },
    [refresh, handleDelete],
  );

  // -- Meal group menu action handler --

  const handleMealGroupAction = useCallback(
    (action: MealGroupAction, mealGroup: MealGroup) => {
      setMenuMealGroup(null);

      switch (action) {
        case 'copy-from-date': {
          setDatePickerContext({ action: 'copy-from-date', mealGroup });
          setCalendarVisible(true);
          break;
        }

        case 'copy-yesterday': {
          const yesterday = new Date();
          yesterday.setDate(yesterday.getDate() - 1);
          const yesterdayStr = dateToStr(yesterday);
          const count = copyAllEntriesFromDate(yesterdayStr, selectedDate, mealGroup);
          refresh();
          Alert.alert('Copied', `Copied ${count} item${count !== 1 ? 's' : ''} from yesterday.`);
          break;
        }

        case 'save-template': {
          Alert.alert('Coming Soon', 'Meal templates will be available in a future update.');
          break;
        }
      }
    },
    [selectedDate, refresh],
  );

  // -- Calendar picker date selection handler --

  const handleCalendarSelect = useCallback(
    (dateStr: string) => {
      if (datePickerContext) {
        const ctx = datePickerContext;
        setDatePickerContext(null);
        setCalendarVisible(false);

        if (ctx.action === 'copy-entry' && ctx.entryId) {
          copyEntryToDate(ctx.entryId, dateStr);
          refresh();
          Alert.alert('Copied', 'Entry copied to selected date.');
        } else if (ctx.action === 'copy-from-date' && ctx.mealGroup) {
          const count = copyAllEntriesFromDate(dateStr, selectedDate, ctx.mealGroup);
          refresh();
          Alert.alert('Copied', `Copied ${count} item${count !== 1 ? 's' : ''} from selected date.`);
        }
      } else {
        // Normal date navigation
        setSelectedDate(dateStr);
      }
    },
    [datePickerContext, selectedDate, refresh],
  );

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
              tintColor={colors.accent.green}
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
        onSelect={handleCalendarSelect}
        onDismiss={() => {
          setCalendarVisible(false);
          setDatePickerContext(null);
        }}
      />

      {/* Bottom sheets -- rendered outside GestureDetector to avoid gesture conflicts */}
      <ItemDetailSheet
        entryId={selectedEntryId}
        onDismiss={() => setSelectedEntryId(null)}
        onEdit={handleEdit}
        onDelete={handleDelete}
      />

      <ContextMenuSheet
        entry={contextMenuEntry}
        onDismiss={() => setContextMenuEntry(null)}
        onAction={handleContextAction}
      />

      <MealGroupMenuSheet
        mealGroup={menuMealGroup}
        selectedDate={selectedDate}
        onDismiss={() => setMenuMealGroup(null)}
        onAction={handleMealGroupAction}
      />
    </SafeAreaView>
  );
}

function createStyles(colors: ThemeColors) {
  return StyleSheet.create({
    screen: {
      flex: 1,
      backgroundColor: colors.background.primary,
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
      color: colors.text.tertiary,
      marginBottom: 6,
    },
    emptySubtext: {
      fontSize: 14,
      color: colors.input.placeholder,
      textAlign: 'center',
      paddingHorizontal: 32,
    },
  });
}
