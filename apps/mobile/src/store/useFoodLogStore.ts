import { create } from 'zustand';
import { eq, and } from 'drizzle-orm';
import { userDb } from '../../db/client';
import { foodEntries } from '../../db/schema';
import { FoodEntry, Photo } from '../types';

interface FoodLogState {
  entries: FoodEntry[];
  selectedPhotos: Photo[];
  isProcessing: boolean;

  // Actions
  addEntry: (entry: Omit<FoodEntry, 'id' | 'createdAt' | 'updatedAt' | 'isSynced' | 'isDeleted' | 'entryDate' | 'photos' | 'ingredients'> & { photos?: Photo[]; ingredients?: FoodEntry['ingredients'] }) => Promise<void>;
  updateEntry: (id: string, updates: Partial<FoodEntry>) => Promise<void>;
  deleteEntry: (id: string) => Promise<void>;
  loadTodayEntries: () => Promise<void>;
  setSelectedPhotos: (photos: Photo[]) => void;
  clearSelectedPhotos: () => void;
  setIsProcessing: (isProcessing: boolean) => void;
  getTodayTotals: () => {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
}

function getTodayDateStr(): string {
  return new Date().toISOString().split('T')[0];
}

export const useFoodLogStore = create<FoodLogState>((set, get) => ({
  entries: [],
  selectedPhotos: [],
  isProcessing: false,

  addEntry: async (entryData) => {
    const id = crypto.randomUUID();
    const now = new Date().toISOString();
    const entryDate = getTodayDateStr();

    // Write to SQLite first
    await userDb.insert(foodEntries).values({
      id,
      mealType: entryData.mealType,
      entryDate,
      totalCalories: entryData.totalCalories,
      totalProtein: entryData.totalProtein,
      totalCarbs: entryData.totalCarbs,
      totalFat: entryData.totalFat,
      notes: entryData.notes ?? null,
      createdAt: now,
      updatedAt: now,
      isSynced: false,
      isDeleted: false,
    });

    // Refresh cache from SQLite
    await get().loadTodayEntries();
  },

  updateEntry: async (id, updates) => {
    // Write to SQLite first
    await userDb
      .update(foodEntries)
      .set({
        ...updates,
        updatedAt: new Date().toISOString(),
      })
      .where(eq(foodEntries.id, id));

    // Refresh cache from SQLite
    await get().loadTodayEntries();
  },

  deleteEntry: async (id) => {
    // Soft-delete: set isDeleted = true
    await userDb
      .update(foodEntries)
      .set({
        isDeleted: true,
        updatedAt: new Date().toISOString(),
      })
      .where(eq(foodEntries.id, id));

    // Refresh cache from SQLite
    await get().loadTodayEntries();
  },

  loadTodayEntries: async () => {
    const todayStr = getTodayDateStr();
    const rows = await userDb
      .select()
      .from(foodEntries)
      .where(
        and(
          eq(foodEntries.entryDate, todayStr),
          eq(foodEntries.isDeleted, false)
        )
      );

    // Map DB rows to FoodEntry type (photos and ingredients loaded separately)
    const entries: FoodEntry[] = rows.map((row) => ({
      id: row.id,
      createdAt: row.createdAt ?? new Date().toISOString(),
      entryDate: row.entryDate,
      mealType: row.mealType as FoodEntry['mealType'],
      photos: [],
      ingredients: [],
      totalCalories: row.totalCalories ?? 0,
      totalProtein: row.totalProtein ?? 0,
      totalCarbs: row.totalCarbs ?? 0,
      totalFat: row.totalFat ?? 0,
      notes: row.notes ?? undefined,
      updatedAt: row.updatedAt ?? new Date().toISOString(),
      isSynced: row.isSynced ?? false,
      isDeleted: row.isDeleted ?? false,
    }));

    set({ entries });
  },

  setSelectedPhotos: (photos) => set({ selectedPhotos: photos }),

  clearSelectedPhotos: () => set({ selectedPhotos: [] }),

  setIsProcessing: (isProcessing) => set({ isProcessing }),

  getTodayTotals: () => {
    const entries = get().entries;
    return entries.reduce(
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
