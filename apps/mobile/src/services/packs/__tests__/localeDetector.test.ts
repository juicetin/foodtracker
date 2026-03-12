/**
 * Tests for locale detection and regional pack suggestion.
 *
 * Mocks expo-localization to test locale detection without device APIs.
 */

// Mock expo-localization
jest.mock('expo-localization', () => ({
  getLocales: jest.fn(() => [
    { languageTag: 'en-US', languageCode: 'en', regionCode: 'US' },
  ]),
}));

import { getLocales } from 'expo-localization';
import {
  detectLocale,
  suggestRegionalPack,
  getSupportedRegions,
} from '../localeDetector';

const mockGetLocales = getLocales as jest.MockedFunction<typeof getLocales>;

describe('localeDetector', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('detectLocale', () => {
    it('returns normalized locale string from device', () => {
      mockGetLocales.mockReturnValue([
        { languageTag: 'en-AU', languageCode: 'en', regionCode: 'AU' } as ReturnType<typeof getLocales>[0],
      ]);

      const locale = detectLocale();
      expect(locale).toBe('en-au');
    });

    it('returns lowercase locale with region', () => {
      mockGetLocales.mockReturnValue([
        { languageTag: 'fr-FR', languageCode: 'fr', regionCode: 'FR' } as ReturnType<typeof getLocales>[0],
      ]);

      const locale = detectLocale();
      expect(locale).toBe('fr-fr');
    });

    it('falls back to en-us when locales are empty', () => {
      mockGetLocales.mockReturnValue([]);

      const locale = detectLocale();
      expect(locale).toBe('en-us');
    });
  });

  describe('suggestRegionalPack', () => {
    it('returns afcd for en-AU locale', () => {
      expect(suggestRegionalPack('en-AU')).toBe('afcd');
    });

    it('returns cofid for en-GB locale', () => {
      expect(suggestRegionalPack('en-GB')).toBe('cofid');
    });

    it('returns ciqual for fr-FR locale', () => {
      expect(suggestRegionalPack('fr-FR')).toBe('ciqual');
    });

    it('returns ciqual for fr-CA locale (fr-* prefix matching)', () => {
      expect(suggestRegionalPack('fr-CA')).toBe('ciqual');
    });

    it('returns null for en-US locale (no regional suggestion -- USDA is default)', () => {
      expect(suggestRegionalPack('en-US')).toBeNull();
    });

    it('returns null for de-DE locale (unsupported region)', () => {
      expect(suggestRegionalPack('de-DE')).toBeNull();
    });

    it('handles case-insensitive input', () => {
      expect(suggestRegionalPack('EN-AU')).toBe('afcd');
      expect(suggestRegionalPack('en-au')).toBe('afcd');
      expect(suggestRegionalPack('FR-fr')).toBe('ciqual');
    });

    it('returns null for empty string', () => {
      expect(suggestRegionalPack('')).toBeNull();
    });
  });

  describe('getSupportedRegions', () => {
    it('returns list of all available regional packs', () => {
      const regions = getSupportedRegions();

      expect(regions.length).toBeGreaterThanOrEqual(3);

      const packIds = regions.map((r) => r.packId);
      expect(packIds).toContain('afcd');
      expect(packIds).toContain('cofid');
      expect(packIds).toContain('ciqual');
    });

    it('each region has required fields', () => {
      const regions = getSupportedRegions();

      for (const region of regions) {
        expect(region.packId).toBeTruthy();
        expect(region.region).toBeTruthy();
        expect(region.name).toBeTruthy();
        expect(region.locale).toBeTruthy();
      }
    });
  });
});
