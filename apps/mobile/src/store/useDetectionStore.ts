import { create } from 'zustand';
import { autoDetectMealType, type MealType } from '../services/detection/types';
import type { ScannedDish, ScannedIngredient, ScanResult } from '../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DetectionState {
  photoUri: string | null;
  dishes: ScannedDish[];
  isAnalyzing: boolean;
  mealType: MealType;
  isMock: boolean;
  /** Reason mock data was used (only set when isMock is true). */
  mockReason?: 'unavailable' | 'error' | 'empty';
  /** Number of photos still being processed in the background. */
  pendingPhotos: number;
  totalPhotos: number;
}

interface DetectionStore extends DetectionState {
  setScanResult: (result: ScanResult) => void;
  /** Append dishes from an additional photo scan (multi-photo). */
  addScanResult: (result: ScanResult) => void;
  setPendingPhotos: (pending: number, total: number) => void;
  setAnalyzing: (val: boolean) => void;
  setMealType: (type: MealType) => void;
  /** Edit an ingredient's name, weight, or nutrition. Marks it userModified. */
  updateIngredient: (
    dishId: string,
    ingId: string,
    update: Partial<Pick<ScannedIngredient, 'name' | 'amount_g' | 'calories' | 'protein' | 'carbs' | 'fat' | 'fiber' | 'sodium'>>,
  ) => void;
  /** Scale all non-userModified ingredients proportionally. */
  updateDishScale: (dishId: string, scale: number) => void;
  updateDishName: (dishId: string, name: string) => void;
  removeIngredient: (dishId: string, ingId: string) => void;
  removeDish: (dishId: string) => void;
  reset: () => void;
  getDishTotals: (dishId: string) => { calories: number; protein: number; carbs: number; fat: number };
  getTotalNutrition: () => { calories: number; protein: number; carbs: number; fat: number };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ingScale(ing: ScannedIngredient): number {
  return ing.originalAmount_g > 0 ? ing.amount_g / ing.originalAmount_g : 1;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const initialState: DetectionState = {
  photoUri: null,
  dishes: [],
  isAnalyzing: false,
  mealType: autoDetectMealType(),
  isMock: false,
  mockReason: undefined,
  pendingPhotos: 0,
  totalPhotos: 0,
};

export const useDetectionStore = create<DetectionStore>((set, get) => ({
  ...initialState,

  setScanResult: (result) =>
    set({ photoUri: result.photoUri, dishes: result.dishes, isMock: result.isMock, mockReason: result.mockReason }),

  addScanResult: (result) =>
    set((state) => ({
      dishes: [...state.dishes, ...result.dishes],
      pendingPhotos: Math.max(0, state.pendingPhotos - 1),
    })),

  setPendingPhotos: (pending, total) => set({ pendingPhotos: pending, totalPhotos: total }),

  setAnalyzing: (val) => set({ isAnalyzing: val }),

  setMealType: (type) => set({ mealType: type }),

  updateIngredient: (dishId, ingId, update) =>
    set((state) => ({
      dishes: state.dishes.map((dish) =>
        dish.id !== dishId
          ? dish
          : {
              ...dish,
              ingredients: dish.ingredients.map((ing) =>
                ing.id !== ingId
                  ? ing
                  : { ...ing, ...update, userModified: true },
              ),
            },
      ),
    })),

  updateDishScale: (dishId, scale) =>
    set((state) => ({
      dishes: state.dishes.map((dish) =>
        dish.id !== dishId
          ? dish
          : {
              ...dish,
              portionScale: scale,
              ingredients: dish.ingredients.map((ing) =>
                ing.userModified
                  ? ing
                  : { ...ing, amount_g: ing.originalAmount_g * scale },
              ),
            },
      ),
    })),

  updateDishName: (dishId, name) =>
    set((state) => ({
      dishes: state.dishes.map((d) => (d.id !== dishId ? d : { ...d, name })),
    })),

  removeIngredient: (dishId, ingId) =>
    set((state) => ({
      dishes: state.dishes.map((dish) =>
        dish.id !== dishId
          ? dish
          : { ...dish, ingredients: dish.ingredients.filter((i) => i.id !== ingId) },
      ),
    })),

  removeDish: (dishId) =>
    set((state) => ({ dishes: state.dishes.filter((d) => d.id !== dishId) })),

  reset: () => set({ ...initialState, mealType: autoDetectMealType() }),

  getDishTotals: (dishId) => {
    const dish = get().dishes.find((d) => d.id === dishId);
    if (!dish) return { calories: 0, protein: 0, carbs: 0, fat: 0 };
    return dish.ingredients.reduce(
      (acc, ing) => {
        const s = ingScale(ing);
        return {
          calories: acc.calories + ing.calories * s,
          protein:  acc.protein  + ing.protein  * s,
          carbs:    acc.carbs    + ing.carbs    * s,
          fat:      acc.fat      + ing.fat      * s,
        };
      },
      { calories: 0, protein: 0, carbs: 0, fat: 0 },
    );
  },

  getTotalNutrition: () => {
    const { dishes } = get();
    const store = get();
    return dishes.reduce(
      (acc, dish) => {
        const t = store.getDishTotals(dish.id);
        return {
          calories: acc.calories + t.calories,
          protein:  acc.protein  + t.protein,
          carbs:    acc.carbs    + t.carbs,
          fat:      acc.fat      + t.fat,
        };
      },
      { calories: 0, protein: 0, carbs: 0, fat: 0 },
    );
  },
}));
