import { requireNativeModule } from 'expo-modules-core';

const AiPackDeliveryNative = requireNativeModule('AiPackDelivery');

export type AiPackStatus = 'completed' | 'pending' | 'downloading' | 'not_installed' | 'unknown';

export const aiPackDeliveryModule = {
  /**
   * Get the download/install status of an AI pack.
   * Returns 'unknown' on any error (safe default).
   */
  getPackStatus(packName: string): Promise<AiPackStatus> {
    return AiPackDeliveryNative.getPackStatus(packName);
  },

  /**
   * Get the local filesystem path for a completed AI pack's assets.
   * Returns null if pack is not completed or on any error.
   */
  getPackLocation(packName: string): Promise<string | null> {
    return AiPackDeliveryNative.getPackLocation(packName);
  },

  /**
   * Request download of an AI pack.
   * Returns false on any error (safe default).
   */
  requestDownload(packName: string): Promise<boolean> {
    return AiPackDeliveryNative.requestDownload(packName);
  },
};
