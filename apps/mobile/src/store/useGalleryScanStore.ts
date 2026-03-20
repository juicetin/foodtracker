/**
 * Gallery scan state store -- Zustand + persist via AsyncStorage.
 *
 * Tracks scan progress, last result, auto-scan preference, and errors.
 */

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  triggerForegroundDrain,
  registerGalleryScan,
  unregisterGalleryScan,
} from '../services/gallery/galleryScanScheduler';

interface ScanProgress {
  done: number;
  total: number;
}

interface ScanResult {
  classified: number;
  foodPhotos: number;
  mealGroups: number;
}

interface GalleryScanState {
  // State
  isScanning: boolean;
  progress: ScanProgress | null;
  lastScanResult: ScanResult | null;
  error: string | null;
  scanEnabled: boolean;

  // Actions
  startManualScan: () => Promise<void>;
  setScanEnabled: (enabled: boolean) => Promise<void>;
  reset: () => void;
}

export const useGalleryScanStore = create<GalleryScanState>()(
  persist(
    (set, get) => ({
      isScanning: false,
      progress: null,
      lastScanResult: null,
      error: null,
      scanEnabled: false,

      startManualScan: async () => {
        if (get().isScanning) return;

        set({ isScanning: true, progress: null, error: null });

        try {
          const result = await triggerForegroundDrain((done, total) => {
            set({ progress: { done, total } });
          });

          set({
            isScanning: false,
            progress: null,
            lastScanResult: result,
          });
        } catch (err) {
          set({
            isScanning: false,
            progress: null,
            error: err instanceof Error ? err.message : 'Scan failed',
          });
        }
      },

      setScanEnabled: async (enabled: boolean) => {
        try {
          if (enabled) {
            await registerGalleryScan();
          } else {
            await unregisterGalleryScan();
          }
          set({ scanEnabled: enabled });
        } catch (err) {
          set({
            error: err instanceof Error ? err.message : 'Failed to toggle auto-scan',
          });
        }
      },

      reset: () => {
        set({ error: null, progress: null });
      },
    }),
    {
      name: 'gallery-scan-store',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        scanEnabled: state.scanEnabled,
        lastScanResult: state.lastScanResult,
      }),
    },
  ),
);
