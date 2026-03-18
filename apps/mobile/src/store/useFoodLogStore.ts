import { create } from 'zustand';
import { eq, and } from 'drizzle-orm';
import { userDb, opsqlite } from '../../db/client';
import { foodEntries } from '../../db/schema';
import type { FoodEntry, Photo, ScanResult } from '../types';
import type { MealType } from '../services/detection/types';

/** Generate a UUID without relying on crypto.randomUUID (unavailable in some RN runtimes). */
function generateId(): string {
  const hex = '0123456789abcdef';
  let id = '';
  for (let i = 0; i < 36; i++) {
    if (i === 8 || i === 13 || i === 18 || i === 23) {
      id += '-';
    } else if (i === 14) {
      id += '4'; // UUID v4
    } else {
      id += hex[Math.floor(Math.random() * 16)];
    }
  }
  return id;
}

interface FoodLogState {
  entries: FoodEntry[];
  selectedPhotos: Photo[];
  isProcessing: boolean;

  // Actions
  addEntry: (entry: Omit<FoodEntry, 'id' | 'createdAt' | 'updatedAt' | 'isSynced' | 'isDeleted' | 'entryDate' | 'photos' | 'ingredients'> & { photos?: Photo[]; ingredients?: FoodEntry['ingredients'] }) => Promise<void>;
  logScanResult: (result: ScanResult, mealType: MealType) => Promise<void>;
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
    const id = generateId();
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

  logScanResult: async (result, mealType) => {
    const entryId = generateId();
    const now = new Date().toISOString();
    const entryDate = getTodayDateStr();

    // Calculate totals (nutrition stored at originalAmount_g, scale by current amount_g)
    const totals = result.dishes.reduce(
      (acc, dish) => {
        dish.ingredients.forEach((ing) => {
          const s = ing.originalAmount_g > 0 ? ing.amount_g / ing.originalAmount_g : 1;
          acc.calories += ing.calories * s;
          acc.protein  += ing.protein  * s;
          acc.carbs    += ing.carbs    * s;
          acc.fat      += ing.fat      * s;
        });
        return acc;
      },
      { calories: 0, protein: 0, carbs: 0, fat: 0 },
    );

    await userDb.insert(foodEntries).values({
      id: entryId,
      mealType,
      entryDate,
      totalCalories: Math.round(totals.calories),
      totalProtein:  Math.round(totals.protein),
      totalCarbs:    Math.round(totals.carbs),
      totalFat:      Math.round(totals.fat),
      createdAt: now,
      updatedAt: now,
      isSynced: false,
      isDeleted: false,
    });

    // Photo
    const photoId = generateId();
    opsqlite.execute(
      'INSERT INTO photos (id, entry_id, uri, uploaded_at) VALUES (?, ?, ?, ?)',
      [photoId, entryId, result.photoUri, now],
    );

    // Dishes + ingredients
    for (const dish of result.dishes) {
      const dishId = generateId();
      opsqlite.execute(
        'INSERT INTO scanned_dishes (id, entry_id, name, cuisine, portion_scale, created_at) VALUES (?, ?, ?, ?, ?, ?)',
        [dishId, entryId, dish.name, dish.cuisine ?? null, dish.portionScale, now],
      );
      for (const ing of dish.ingredients) {
        const s = ing.originalAmount_g > 0 ? ing.amount_g / ing.originalAmount_g : 1;
        opsqlite.execute(
          `INSERT INTO ingredients
            (id, entry_id, dish_id, name, quantity, unit, amount_g, original_amount_g,
             calories, protein, carbs, fat, fiber, database_source, user_modified, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, 'g', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            generateId(), entryId, dishId, ing.name,
            ing.amount_g, ing.amount_g, ing.originalAmount_g,
            ing.calories * s, ing.protein * s, ing.carbs * s, ing.fat * s, ing.fiber * s,
            ing.nutritionSource === 'kg' ? 'USDA' : null,
            ing.userModified ? 1 : 0,
            now, now,
          ],
        );
      }
    }

    await get().loadTodayEntries();
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
