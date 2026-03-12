import { create } from 'zustand';
import {
  DetectedItem,
  DetectionSessionState,
  MealType,
  autoDetectMealType,
} from '../services/detection/types';

// ---------------------------------------------------------------------------
// Store interface
// ---------------------------------------------------------------------------

interface DetectionStore extends DetectionSessionState {
  // Actions
  setPhoto: (uri: string, width: number, height: number) => void;
  setItems: (items: DetectedItem[]) => void;
  setDetecting: (detecting: boolean) => void;
  removeItem: (id: string) => void;
  restoreItem: (id: string) => void;
  updatePortion: (id: string, multiplier: number) => void;
  correctItem: (id: string, newClassName: string) => void;
  setMealType: (type: MealType) => void;
  selectItem: (id: string | null) => void;
  reset: () => void;
  // Computed
  activeItems: () => DetectedItem[];
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: DetectionSessionState = {
  photoUri: null,
  photoWidth: 0,
  photoHeight: 0,
  items: [],
  isDetecting: false,
  mealType: autoDetectMealType(),
  selectedItemId: null,
};

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/**
 * Zustand store for detection session state.
 *
 * Unlike useFoodLogStore, this store is ephemeral -- it does NOT persist to
 * SQLite.  The session lives only while the user reviews detected items.
 * Persistence happens when the user taps "Log Meal" (handled elsewhere).
 *
 * Follows write-first pattern conceptually: mutations update Zustand state
 * directly (no DB round-trip needed for ephemeral data).
 */
export const useDetectionStore = create<DetectionStore>((set, get) => ({
  ...initialState,

  // -- Actions ---------------------------------------------------------------

  setPhoto: (uri, width, height) =>
    set({ photoUri: uri, photoWidth: width, photoHeight: height }),

  setItems: (items) => set({ items }),

  setDetecting: (detecting) => set({ isDetecting: detecting }),

  removeItem: (id) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id
          ? { ...item, isRemoved: true, removedAt: Date.now() }
          : item,
      ),
      // Deselect if the removed item was selected
      selectedItemId: state.selectedItemId === id ? null : state.selectedItemId,
    })),

  restoreItem: (id) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id
          ? { ...item, isRemoved: false, removedAt: undefined }
          : item,
      ),
    })),

  updatePortion: (id, multiplier) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id
          ? { ...item, portionMultiplier: Math.min(3.0, Math.max(0.5, multiplier)) }
          : item,
      ),
    })),

  correctItem: (id, newClassName) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id
          ? {
              ...item,
              correctedFrom: item.correctedFrom ?? item.className,
              className: newClassName,
            }
          : item,
      ),
    })),

  setMealType: (type) => set({ mealType: type }),

  selectItem: (id) => set({ selectedItemId: id }),

  reset: () => set({ ...initialState, mealType: autoDetectMealType() }),

  // -- Computed --------------------------------------------------------------

  activeItems: () => {
    const { items } = get();
    return items
      .filter((i) => !i.isRemoved)
      .sort((a, b) => b.confidence - a.confidence);
  },
}));
