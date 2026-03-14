/**
 * Pack system type definitions.
 *
 * Supports both nutrition database packs and ML model packs through a
 * generic versioned manifest format. Community packs follow the same
 * manifest structure as first-party packs.
 */

/** Remote manifest listing all available packs. */
export interface PackManifest {
  version: number;
  lastUpdated: string; // ISO 8601
  packs: PackEntry[];
}

/** A single pack entry in the manifest. */
export interface PackEntry {
  id: string; // e.g., 'usda-core', 'afcd', 'yolo-v1', 'smolvlm-256m'
  name: string; // Human-readable
  type: 'nutrition' | 'model' | 'knowledge-graph' | 'vlm';
  version: string; // Semver
  sizeBytes: number; // Download size (compressed)
  sha256: string; // Integrity hash
  url: string; // R2 relative path
  region?: string; // 'AU', 'UK', 'FR'
  locale?: string; // 'en-AU', 'en-GB', 'fr'
  description: string;
  requiredAppVersion?: string;
  mmprojUrl?: string; // Companion mmproj download URL (VLM packs only)
  mmprojSha256?: string; // Integrity hash for mmproj file
  mmprojSizeBytes?: number; // mmproj file size in bytes
}

/** A pack that has been downloaded and installed locally. */
export interface InstalledPack {
  id: string;
  name: string;
  type: 'nutrition' | 'model' | 'knowledge-graph' | 'vlm';
  version: string;
  filePath: string;
  mmprojFilePath?: string; // Local path to mmproj file (VLM packs only)
  sizeBytes: number | null;
  sha256: string | null;
  region: string | null;
  installedAt: string;
  lastChecked: string | null;
}

/** Download progress callback payload. */
export interface DownloadProgress {
  totalBytesWritten: number;
  totalBytesExpected: number;
  fraction: number;
}

/** Lifecycle status for a pack in the UI. */
export type PackStatus =
  | 'not-installed'
  | 'downloading'
  | 'installed'
  | 'update-available'
  | 'error';
