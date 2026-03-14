/**
 * VLM Model Download Screen.
 *
 * Lets the user download the appropriate SmolVLM model for their device.
 * Auto-detects the best tier based on device RAM, shows download progress,
 * and allows deletion of installed models.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import * as Device from 'expo-device';
import {
  detectVlmTier,
  VLM_TIER_CONFIG,
} from '../services/vlm';
import type { VlmTier, VlmTierConfig } from '../services/vlm';
import { PackManager } from '../services/packs/packManager';
import type { DownloadProgress, PackEntry } from '../services/packs/types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BYTES_PER_GB = 1024 ** 3;
const BYTES_PER_MB = 1024 ** 2;

/** Format bytes to a human-readable string (e.g. "175 MB", "1.11 GB"). */
function formatBytes(bytes: number): string {
  if (bytes >= BYTES_PER_GB) {
    return `${(bytes / BYTES_PER_GB).toFixed(2)} GB`;
  }
  return `${Math.round(bytes / BYTES_PER_MB)} MB`;
}

/** Map tier to human-readable display name. */
function tierDisplayName(tier: VlmTier): string {
  switch (tier) {
    case 'budget':
      return 'Budget (SmolVLM 256M)';
    case 'mid':
      return 'Standard (SmolVLM 500M)';
    case 'high':
      return 'High (SmolVLM2 2.2B)';
    case 'none':
      return 'Not Available';
  }
}

/** HuggingFace direct download URLs for each tier. */
const TIER_DOWNLOAD_URLS: Record<
  Exclude<VlmTier, 'none'>,
  { modelUrl: string; mmprojUrl: string }
> = {
  budget: {
    modelUrl:
      'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf',
    mmprojUrl:
      'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-f16.gguf',
  },
  mid: {
    modelUrl:
      'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf',
    mmprojUrl:
      'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf',
  },
  high: {
    modelUrl:
      'https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf',
    mmprojUrl:
      'https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf',
  },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type ScreenStatus =
  | 'checking'
  | 'not-installed'
  | 'downloading'
  | 'installed'
  | 'error'
  | 'unavailable';

export default function VlmDownloadScreen() {
  const [status, setStatus] = useState<ScreenStatus>('checking');
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tier, setTier] = useState<VlmTier>('none');
  const [tierConfig, setTierConfig] = useState<VlmTierConfig | null>(null);

  const deviceRamGB =
    Device.totalMemory != null
      ? (Device.totalMemory / BYTES_PER_GB).toFixed(1)
      : 'Unknown';

  // -----------------------------------------------------------------------
  // On mount: detect tier and check install status
  // -----------------------------------------------------------------------
  useEffect(() => {
    (async () => {
      const detected = detectVlmTier();
      setTier(detected);

      if (detected === 'none') {
        setStatus('unavailable');
        return;
      }

      const config = VLM_TIER_CONFIG[detected];
      setTierConfig(config);

      const installed = await PackManager.isPackInstalled(config.modelId);
      setStatus(installed ? 'installed' : 'not-installed');
    })();
  }, []);

  // -----------------------------------------------------------------------
  // Download handler
  // -----------------------------------------------------------------------
  const handleDownload = useCallback(async () => {
    if (tier === 'none' || !tierConfig) return;

    const urls = TIER_DOWNLOAD_URLS[tier as Exclude<VlmTier, 'none'>];
    const packEntry: PackEntry = {
      id: tierConfig.modelId,
      name: tierConfig.modelFile,
      type: 'vlm',
      version: '1.0.0',
      sizeBytes: tierConfig.modelSize,
      sha256: '', // Skip integrity check for initial release (HuggingFace official repos)
      url: urls.modelUrl,
      mmprojUrl: urls.mmprojUrl,
      mmprojSizeBytes: tierConfig.mmprojSize,
      mmprojSha256: '', // Skip integrity check for initial release
      description: `SmolVLM ${tier} tier for food identification`,
    };

    setStatus('downloading');
    setProgress(null);
    setError(null);

    try {
      await PackManager.downloadPack(packEntry, (p) => {
        setProgress(p);
      });
      setStatus('installed');
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Download failed';
      setError(message);
      setStatus('error');
    }
  }, [tier, tierConfig]);

  // -----------------------------------------------------------------------
  // Delete handler
  // -----------------------------------------------------------------------
  const handleDelete = useCallback(async () => {
    if (!tierConfig) return;

    Alert.alert(
      'Delete VLM Model',
      'This will remove the downloaded model. You can re-download it later.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await PackManager.deletePack(tierConfig.modelId);
              setStatus('not-installed');
            } catch (err) {
              const message =
                err instanceof Error ? err.message : 'Delete failed';
              setError(message);
              setStatus('error');
            }
          },
        },
      ]
    );
  }, [tierConfig]);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  if (status === 'checking') {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.checkingText}>Checking device...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <Text style={styles.title}>VLM Model</Text>
      <Text style={styles.subtitle}>
        {tier === 'none'
          ? 'Device not supported'
          : `Tier: ${tierDisplayName(tier)} -- Device RAM: ${deviceRamGB} GB`}
      </Text>

      {/* Unavailable */}
      {status === 'unavailable' && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Not Available</Text>
          <Text style={styles.cardBody}>
            Your device has less than 4GB RAM. VLM is not available.
          </Text>
        </View>
      )}

      {/* Tier info card */}
      {tierConfig && status !== 'unavailable' && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>{tierDisplayName(tier)}</Text>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Model</Text>
            <Text style={styles.infoValue}>{tierConfig.modelFile}</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Download Size</Text>
            <Text style={styles.infoValue}>
              {formatBytes(tierConfig.totalDownload)}
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>RAM Required</Text>
            <Text style={styles.infoValue}>
              {formatBytes(tierConfig.runtimeRam)}
            </Text>
          </View>
        </View>
      )}

      {/* Status-specific UI */}
      {status === 'not-installed' && (
        <Pressable style={styles.downloadButton} onPress={handleDownload}>
          <Text style={styles.downloadButtonText}>Download</Text>
        </Pressable>
      )}

      {status === 'downloading' && progress && (
        <View style={styles.progressSection}>
          <View style={styles.progressBarBg}>
            <View
              style={[
                styles.progressBarFill,
                { width: `${Math.round(progress.fraction * 100)}%` },
              ]}
            />
          </View>
          <Text style={styles.progressText}>
            {Math.round(progress.fraction * 100)}% --{' '}
            {formatBytes(progress.totalBytesWritten)} /{' '}
            {formatBytes(progress.totalBytesExpected)}
          </Text>
        </View>
      )}

      {status === 'downloading' && !progress && (
        <View style={styles.progressSection}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.progressText}>Starting download...</Text>
        </View>
      )}

      {status === 'installed' && (
        <View style={styles.installedSection}>
          <Text style={styles.checkmark}>Installed</Text>
          <Pressable style={styles.deleteButton} onPress={handleDelete}>
            <Text style={styles.deleteButtonText}>Delete Model</Text>
          </Pressable>
        </View>
      )}

      {status === 'error' && (
        <View style={styles.errorSection}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={handleDownload}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 24,
  },
  checkingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#F8F8F8',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  cardBody: {
    fontSize: 15,
    color: '#666',
    lineHeight: 22,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E8E8E8',
  },
  infoLabel: {
    fontSize: 15,
    color: '#555',
  },
  infoValue: {
    fontSize: 15,
    color: '#333',
    fontWeight: '500',
    flexShrink: 1,
    textAlign: 'right',
    maxWidth: '60%',
  },
  downloadButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  downloadButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
  },
  progressSection: {
    alignItems: 'center',
    gap: 12,
  },
  progressBarBg: {
    width: '100%',
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#22C55E',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: '#666',
  },
  installedSection: {
    alignItems: 'center',
    gap: 16,
  },
  checkmark: {
    fontSize: 18,
    fontWeight: '600',
    color: '#22C55E',
  },
  deleteButton: {
    borderWidth: 1,
    borderColor: '#FF3B30',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  deleteButtonText: {
    color: '#FF3B30',
    fontSize: 16,
    fontWeight: '500',
  },
  errorSection: {
    alignItems: 'center',
    gap: 12,
  },
  errorText: {
    fontSize: 14,
    color: '#FF3B30',
    textAlign: 'center',
  },
  retryButton: {
    backgroundColor: '#FF9500',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 32,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
