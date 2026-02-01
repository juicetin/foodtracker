import { create } from 'zustand';
import { FoodEntry, Photo } from '../types';

interface FoodLogState {
  entries: FoodEntry[];
  selectedPhotos: Photo[];
  isProcessing: boolean;

  // Actions
  addEntry: (entry: FoodEntry) => void;
  updateEntry: (id: string, updates: Partial<FoodEntry>) => void;
  deleteEntry: (id: string) => void;
  setSelectedPhotos: (photos: Photo[]) => void;
  clearSelectedPhotos: () => void;
  setIsProcessing: (isProcessing: boolean) => void;
  getTodayEntries: () => FoodEntry[];
  getTodayTotals: () => {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
}

export const useFoodLogStore = create<FoodLogState>((set, get) => ({
  entries: [],
  selectedPhotos: [],
  isProcessing: false,

  addEntry: (entry) =>
    set((state) => ({
      entries: [...state.entries, entry],
    })),

  updateEntry: (id, updates) =>
    set((state) => ({
      entries: state.entries.map((entry) =>
        entry.id === id ? { ...entry, ...updates } : entry
      ),
    })),

  deleteEntry: (id) =>
    set((state) => ({
      entries: state.entries.filter((entry) => entry.id !== id),
    })),

  setSelectedPhotos: (photos) =>
    set({ selectedPhotos: photos }),

  clearSelectedPhotos: () =>
    set({ selectedPhotos: [] }),

  setIsProcessing: (isProcessing) =>
    set({ isProcessing }),

  getTodayEntries: () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return get().entries.filter((entry) => {
      const entryDate = new Date(entry.createdAt);
      entryDate.setHours(0, 0, 0, 0);
      return entryDate.getTime() === today.getTime();
    });
  },

  getTodayTotals: () => {
    const todayEntries = get().getTodayEntries();
    return todayEntries.reduce(
      (totals, entry) => ({
        calories: totals.calories + entry.totalCalories,
        protein: totals.protein + entry.totalProtein,
        carbs: totals.carbs + entry.totalCarbs,
        fat: totals.fat + entry.totalFat,
      }),
      { calories: 0, protein: 0, carbs: 0, fat: 0 }
    );
  },
}));
