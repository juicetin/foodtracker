/**
 * Pack manifest fetching and comparison utilities.
 *
 * Fetches the remote pack manifest from R2, parses it, and provides
 * utilities for comparing against locally installed packs.
 *
 * Phase 1: Uses API key header (X-API-Key) for R2 access.
 * Phase 6: Will add full app attestation (Play Integrity / App Attest).
 */

import type { PackManifest, PackEntry, InstalledPack } from './types';

/**
 * Fetch the pack manifest from a remote URL.
 *
 * @param manifestUrl - Full URL to the manifest JSON
 * @param apiKey - API key for R2 access (X-API-Key header)
 * @returns Parsed PackManifest
 * @throws If the fetch fails or the response is not OK
 */
export async function fetchManifest(
  manifestUrl: string,
  apiKey: string
): Promise<PackManifest> {
  const response = await fetch(manifestUrl, {
    headers: {
      'X-API-Key': apiKey,
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to fetch pack manifest: ${response.status} ${response.statusText}`
    );
  }

  const data: PackManifest = await response.json();
  return data;
}

/**
 * Compare simple semver strings (e.g., "1.0.0" vs "2.0.0").
 * Returns true if versionA < versionB.
 */
function isVersionNewer(installed: string, available: string): boolean {
  const parse = (v: string) => v.split('.').map(Number);
  const a = parse(installed);
  const b = parse(available);

  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    if (bv > av) return true;
    if (bv < av) return false;
  }
  return false;
}

/**
 * Find packs in the manifest that have newer versions than installed.
 *
 * Only returns updates for packs that are already installed locally.
 * New packs that are not installed are NOT included.
 *
 * @param manifest - Remote pack manifest
 * @param installed - List of locally installed packs
 * @returns Pack entries with newer versions available
 */
export function getAvailableUpdates(
  manifest: PackManifest,
  installed: InstalledPack[]
): PackEntry[] {
  const installedMap = new Map(installed.map((p) => [p.id, p]));

  return manifest.packs.filter((entry) => {
    const local = installedMap.get(entry.id);
    if (!local) return false; // Not installed, not an "update"
    return isVersionNewer(local.version, entry.version);
  });
}

/**
 * Filter packs by type (nutrition or model).
 */
export function getPacksByType(
  manifest: PackManifest,
  type: 'nutrition' | 'model'
): PackEntry[] {
  return manifest.packs.filter((p) => p.type === type);
}

/**
 * Filter nutrition packs by region.
 */
export function getPacksByRegion(
  manifest: PackManifest,
  region: string
): PackEntry[] {
  return manifest.packs.filter((p) => p.region === region);
}
