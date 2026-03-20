import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserPreferences, UxMode } from '../types';
import { detectLocale } from '../services/packs/localeDetector';
import { type TimePeriodBoundary, DEFAULT_BOUNDARIES } from '../services/diary/timePeriods';

/** Map device locale to our region codes. */
function localeToRegion(locale: string): UserPreferences['region'] {
  const l = locale.toLowerCase();
  if (l.startsWith('en-au')) return 'AU';
  if (l.startsWith('en-us')) return 'US';
  if (l.startsWith('en-ca')) return 'CA';
  if (l.startsWith('en-gb') || l.startsWith('en-ie')) return 'UK';
  if (l.startsWith('fr')) return 'FR';
  // Default: detect by language
  if (l.startsWith('en')) return 'US';
  return 'global';
}

interface PreferencesState extends UserPreferences {
  regionAutoDetected: boolean;
  diaryDisplayMode: 'consumed' | 'remaining';
  timePeriodBoundaries: TimePeriodBoundary;
  /** Whether daily macro notifications are enabled. */
  notificationsEnabled: boolean;
  /** Hour (0-23) for the daily notification. Default: 21 (9 PM). */
  notificationHour: number;
  /** Minute (0-59) for the daily notification. Default: 0. */
  notificationMinute: number;
  /** Whether Health Connect weight import is enabled (opt-in). */
  healthConnectEnabled: boolean;
  // Actions
  setRegion: (region: UserPreferences['region']) => void;
  setUnits: (units: UserPreferences['units']) => void;
  setNutritionGoals: (goals: UserPreferences['nutritionGoals']) => void;
  setDarkMode: (darkMode: boolean) => void;
  setDiaryDisplayMode: (mode: 'consumed' | 'remaining') => void;
  setTimePeriodBoundaries: (b: TimePeriodBoundary) => void;
  setUxMode: (mode: UxMode) => void;
  setNotificationsEnabled: (enabled: boolean) => void;
  setNotificationTime: (hour: number, minute: number) => void;
  setHealthConnectEnabled: (enabled: boolean) => void;
  initRegionFromLocale: () => void;
}

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set, get) => ({
      // Default values
      region: 'AU',
      units: 'metric',
      nutritionGoals: {
        calories: 2000,
        protein: 150,
        carbs: 200,
        fat: 65,
      },
      darkMode: false,
      uxMode: 'confirm-only' as UxMode,
      regionAutoDetected: false,
      diaryDisplayMode: 'consumed',
      timePeriodBoundaries: DEFAULT_BOUNDARIES,
      notificationsEnabled: false,
      notificationHour: 21,
      notificationMinute: 0,
      healthConnectEnabled: false,

      // Actions
      setRegion: (region) => set({ region }),
      setUnits: (units) => set({ units }),
      setNutritionGoals: (goals) => set({ nutritionGoals: goals }),
      setDarkMode: (darkMode) => set({ darkMode }),
      setDiaryDisplayMode: (mode) => set({ diaryDisplayMode: mode }),
      setTimePeriodBoundaries: (b) => set({ timePeriodBoundaries: b }),
      setUxMode: (uxMode) => set({ uxMode }),
      setNotificationsEnabled: (enabled) => set({ notificationsEnabled: enabled }),
      setNotificationTime: (hour, minute) => set({ notificationHour: hour, notificationMinute: minute }),
      setHealthConnectEnabled: (enabled) => set({ healthConnectEnabled: enabled }),
      initRegionFromLocale: () => {
        // Only auto-detect once — user manual override takes precedence
        if (get().regionAutoDetected) return;
        try {
          const locale = detectLocale();
          const region = localeToRegion(locale);
          set({ region, regionAutoDetected: true });
        } catch {
          set({ regionAutoDetected: true });
        }
      },
    }),
    {
      name: 'user-preferences',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
