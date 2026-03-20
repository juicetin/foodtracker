/**
 * Sync state store — Zustand + persist via AsyncStorage.
 *
 * Tracks Google Sign-In status, last sync time, pending conflicts,
 * and user preferences for sync behavior (WiFi-only, auto-resolve).
 */

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { SyncConflict, SyncStatus, FtpSyncStatus } from '../services/sync/types';

interface SyncState {
  // State -- Google Drive
  signedIn: boolean;
  userEmail: string | null;
  lastSyncAt: string | null;
  syncStatus: SyncStatus;
  pendingConflicts: SyncConflict[];
  wifiOnly: boolean;
  autoResolve: boolean;

  // State -- FTP
  ftpEnabled: boolean;
  ftpHost: string | null;
  lastFtpSyncAt: string | null;
  ftpSyncStatus: FtpSyncStatus;

  // Actions -- Google Drive
  setSignedIn: (signedIn: boolean, email?: string | null) => void;
  setLastSyncAt: (timestamp: string | null) => void;
  setSyncStatus: (status: SyncStatus) => void;
  setPendingConflicts: (conflicts: SyncConflict[]) => void;
  setWifiOnly: (wifiOnly: boolean) => void;
  setAutoResolve: (autoResolve: boolean) => void;
  clearSync: () => void;

  // Actions -- FTP
  setFtpEnabled: (enabled: boolean) => void;
  setFtpHost: (host: string | null) => void;
  setLastFtpSyncAt: (timestamp: string | null) => void;
  setFtpSyncStatus: (status: FtpSyncStatus) => void;
}

export const useSyncStore = create<SyncState>()(
  persist(
    (set) => ({
      // Defaults -- Google Drive
      signedIn: false,
      userEmail: null,
      lastSyncAt: null,
      syncStatus: 'idle',
      pendingConflicts: [],
      wifiOnly: true,
      autoResolve: false,

      // Defaults -- FTP
      ftpEnabled: false,
      ftpHost: null,
      lastFtpSyncAt: null,
      ftpSyncStatus: 'idle',

      // Actions -- Google Drive
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

      // Actions -- FTP
      setFtpEnabled: (enabled) => set({ ftpEnabled: enabled }),
      setFtpHost: (host) => set({ ftpHost: host }),
      setLastFtpSyncAt: (timestamp) => set({ lastFtpSyncAt: timestamp }),
      setFtpSyncStatus: (status) => set({ ftpSyncStatus: status }),
    }),
    {
      name: 'sync-state',
      storage: createJSONStorage(() => AsyncStorage),
    },
  ),
);
