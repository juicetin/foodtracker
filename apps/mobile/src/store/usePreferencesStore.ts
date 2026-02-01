import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserPreferences } from '../types';

interface PreferencesState extends UserPreferences {
  // Actions
  setRegion: (region: UserPreferences['region']) => void;
  setUnits: (units: UserPreferences['units']) => void;
  setNutritionGoals: (goals: UserPreferences['nutritionGoals']) => void;
  setDarkMode: (darkMode: boolean) => void;
}

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
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

      // Actions
      setRegion: (region) => set({ region }),
      setUnits: (units) => set({ units }),
      setNutritionGoals: (goals) => set({ nutritionGoals: goals }),
      setDarkMode: (darkMode) => set({ darkMode }),
    }),
    {
      name: 'user-preferences',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
