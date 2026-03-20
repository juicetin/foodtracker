/**
 * Sync state store — Zustand + persist via AsyncStorage.
 *
 * Tracks Google Sign-In status, last sync time, pending conflicts,
 * and user preferences for sync behavior (WiFi-only, auto-resolve).
 */

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { SyncConflict, SyncStatus } from '../services/sync/types';

interface SyncState {
  // State
  signedIn: boolean;
  userEmail: string | null;
  lastSyncAt: string | null;
  syncStatus: SyncStatus;
  pendingConflicts: SyncConflict[];
  wifiOnly: boolean;
  autoResolve: boolean;

  // Actions
  setSignedIn: (signedIn: boolean, email?: string | null) => void;
  setLastSyncAt: (timestamp: string | null) => void;
  setSyncStatus: (status: SyncStatus) => void;
  setPendingConflicts: (conflicts: SyncConflict[]) => void;
  setWifiOnly: (wifiOnly: boolean) => void;
  setAutoResolve: (autoResolve: boolean) => void;
  clearSync: () => void;
}

export const useSyncStore = create<SyncState>()(
  persist(
    (set) => ({
      // Defaults
      signedIn: false,
      userEmail: null,
      lastSyncAt: null,
      syncStatus: 'idle',
      pendingConflicts: [],
      wifiOnly: true,
      autoResolve: false,

      // Actions
      setSignedIn: (signedIn, email) =>
        set({ signedIn, userEmail: email ?? null }),
      setLastSyncAt: (timestamp) => set({ lastSyncAt: timestamp }),
      setSyncStatus: (status) => set({ syncStatus: status }),
      setPendingConflicts: (conflicts) => set({ pendingConflicts: conflicts }),
      setWifiOnly: (wifiOnly) => set({ wifiOnly }),
      setAutoResolve: (autoResolve) => set({ autoResolve }),
      clearSync: () =>
        set({
          signedIn: false,
          userEmail: null,
          lastSyncAt: null,
          syncStatus: 'idle',
          pendingConflicts: [],
        }),
    }),
    {
      name: 'sync-state',
      storage: createJSONStorage(() => AsyncStorage),
    },
  ),
);
