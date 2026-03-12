/**
 * Device locale detection and regional nutrition pack suggestion.
 *
 * Uses expo-localization to detect the device locale, then maps it to the
 * appropriate regional nutrition database pack. This enables auto-suggestion
 * of regional databases (AFCD for Australia, CoFID for UK, CIQUAL for France)
 * during onboarding or in Settings > Data & Storage.
 */

import { getLocales } from 'expo-localization';

/** Regional pack suggestion result for Settings display. */
export interface SupportedRegion {
  packId: string;
  region: string;
  name: string;
  locale: string;
}

/**
 * Exact locale-to-pack mappings.
 * Keyed by lowercase locale string (e.g., 'en-au').
 */
const LOCALE_PACK_MAP: Record<string, string> = {
  'en-au': 'afcd',
  'en-gb': 'cofid',
};

/**
 * Language prefix-to-pack mappings.
 * Used when no exact locale match is found.
 * Keyed by lowercase language code (e.g., 'fr').
 */
const LANGUAGE_PACK_MAP: Record<string, string> = {
  fr: 'ciqual',
};

/**
 * All supported regional packs for display in Settings.
 */
const SUPPORTED_REGIONS: SupportedRegion[] = [
  {
    packId: 'afcd',
    region: 'Australia',
    name: 'Australian Food Composition Database',
    locale: 'en-AU',
  },
  {
    packId: 'cofid',
    region: 'United Kingdom',
    name: 'Composition of Foods Integrated Dataset',
    locale: 'en-GB',
  },
  {
    packId: 'ciqual',
    region: 'France',
    name: 'Table de composition nutritionnelle CIQUAL',
    locale: 'fr-FR',
  },
];

/**
 * Detect the device locale.
 *
 * Returns a normalized lowercase locale string with region
 * (e.g., 'en-au', 'fr-fr'). Falls back to 'en-us' if detection fails.
 *
 * @returns Normalized lowercase locale string
 */
export function detectLocale(): string {
  const locales = getLocales();

  if (!locales || locales.length === 0) {
    return 'en-us';
  }

  const primary = locales[0];
  return primary.languageTag.toLowerCase();
}

/**
 * Suggest a regional nutrition pack based on device locale.
 *
 * Mapping:
 * - en-AU -> 'afcd' (Australian Food Composition Database)
 * - en-GB -> 'cofid' (UK Composition of Foods Integrated Dataset)
 * - fr-* (any French locale) -> 'ciqual' (French CIQUAL database)
 * - All others -> null (USDA is the default, no regional suggestion needed)
 *
 * @param locale - Locale string (e.g., 'en-AU', 'fr-FR', 'fr-CA')
 * @returns Pack ID string or null if no regional suggestion
 */
export function suggestRegionalPack(locale: string): string | null {
  if (!locale) {
    return null;
  }

  const normalized = locale.toLowerCase();

  // Check exact locale match first
  if (LOCALE_PACK_MAP[normalized]) {
    return LOCALE_PACK_MAP[normalized];
  }

  // Check language prefix match (e.g., 'fr-CA' -> 'fr' -> 'ciqual')
  const languageCode = normalized.split('-')[0];
  if (LANGUAGE_PACK_MAP[languageCode]) {
    return LANGUAGE_PACK_MAP[languageCode];
  }

  return null;
}

/**
 * Get the list of all supported regional packs.
 *
 * Used by Settings > Data & Storage to display available regional databases
 * for download (Phase 3 builds the UI).
 *
 * @returns Array of supported regions with pack metadata
 */
export function getSupportedRegions(): SupportedRegion[] {
  return [...SUPPORTED_REGIONS];
}
