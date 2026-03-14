/**
 * Device RAM detection and VLM tier selection.
 *
 * Uses expo-device totalMemory to determine which SmolVLM tier
 * the device can support. Returns 'none' for devices with
 * insufficient RAM (<4GB) or when RAM detection is unavailable.
 */

import * as Device from 'expo-device';
import { VlmTier, VlmTierConfig, VLM_TIER_CONFIG } from './vlmTypes';

const BYTES_PER_GB = 1024 ** 3;

/**
 * Detect the appropriate VLM tier based on device RAM.
 *
 * Thresholds:
 * - >= 8GB: 'high'  (SmolVLM2-2.2B Q4_K_M)
 * - >= 6GB: 'mid'   (SmolVLM-500M Q8_0)
 * - >= 4GB: 'budget' (SmolVLM-256M Q8_0)
 * - < 4GB or null: 'none' (VLM not recommended)
 */
export function detectVlmTier(): VlmTier {
  const totalBytes = Device.totalMemory;
  if (totalBytes == null) return 'none';

  const totalGB = totalBytes / BYTES_PER_GB;

  if (totalGB >= 8) return 'high';
  if (totalGB >= 6) return 'mid';
  if (totalGB >= 4) return 'budget';
  return 'none';
}

/**
 * Get the VLM tier configuration for the current device.
 * Returns null if the device cannot support VLM (tier === 'none').
 */
export function getVlmTierConfig(): VlmTierConfig | null {
  const tier = detectVlmTier();
  if (tier === 'none') return null;
  return VLM_TIER_CONFIG[tier];
}
