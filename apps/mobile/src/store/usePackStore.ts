/**
 * Zustand store for pack UI state.
 *
 * Provides reactive state for download progress, pack statuses,
 * and the isNutritionReady derived flag.
 */

import { create } from 'zustand';
import { PackManager } from '../services/packs/packManager';
import type {
  PackEntry,
  InstalledPack,
  DownloadProgress,
  PackStatus,
} from '../services/packs/types';

interface PackState {
  installedPacks: InstalledPack[];
  downloadProgress: Map<string, DownloadProgress>;
  packStatuses: Map<string, PackStatus>;

  /** True when at least one nutrition pack is installed. */
  isNutritionReady: boolean;

  /** Refresh installed packs from the database. */
  loadInstalledPacks: () => Promise<void>;

  /** Download a pack with progress tracking. */
  downloadPack: (pack: PackEntry, apiKey?: string) => Promise<void>;

  /** Delete an installed pack. */
  deletePack: (packId: string) => Promise<void>;
}

export const usePackStore = create<PackState>((set, get) => ({
  installedPacks: [],
  downloadProgress: new Map(),
  packStatuses: new Map(),
  isNutritionReady: false,

  loadInstalledPacks: async () => {
    const packs = await PackManager.getInstalledPacks();
    const statuses = new Map<string, PackStatus>();
    for (const pack of packs) {
      statuses.set(pack.id, 'installed');
    }

    set({
      installedPacks: packs,
      packStatuses: statuses,
      isNutritionReady: packs.some((p) => p.type === 'nutrition'),
    });
  },

  downloadPack: async (pack: PackEntry, apiKey?: string) => {
    const { packStatuses, downloadProgress } = get();

    // Set status to downloading
    const newStatuses = new Map(packStatuses);
    newStatuses.set(pack.id, 'downloading');
    set({ packStatuses: newStatuses });

    try {
      await PackManager.downloadPack(
        pack,
        (progress) => {
          const updatedProgress = new Map(get().downloadProgress);
          updatedProgress.set(pack.id, progress);
          set({ downloadProgress: updatedProgress });
        },
        apiKey
      );

      // Success -- refresh installed packs
      await get().loadInstalledPacks();

      // Clean up progress
      const cleanProgress = new Map(get().downloadProgress);
      cleanProgress.delete(pack.id);
      set({ downloadProgress: cleanProgress });
    } catch (error) {
      // Set error status
      const errStatuses = new Map(get().packStatuses);
      errStatuses.set(pack.id, 'error');
      set({ packStatuses: errStatuses });
      throw error;
    }
  },

  deletePack: async (packId: string) => {
    await PackManager.deletePack(packId);
    await get().loadInstalledPacks();
  },
}));
