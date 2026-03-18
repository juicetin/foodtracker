/**
 * Unit tests for geminiNanoService — Tier 0 Gemini Nano food identification.
 *
 * Uses Jest auto-mock at src/__mocks__/gemini-nano.ts for the native module.
 * Tests cover: JSON parse, parse failure fallback, availability caching,
 * not_supported status, and raw output capture.
 */

jest.mock('gemini-nano');

// Fresh imports per test to reset module-level cache
let geminiNanoService: typeof import('../geminiNanoService')['geminiNanoService'];
let mockModule: typeof import('../../../__mocks__/gemini-nano')['geminiNanoModule'];

beforeEach(() => {
  jest.resetModules();
  jest.mock('gemini-nano');
  // Re-require fresh instances
  const service = require('../geminiNanoService');
  const mock = require('gemini-nano');
  geminiNanoService = service.geminiNanoService;
  mockModule = mock.geminiNanoModule;
});

describe('geminiNanoService', () => {
  describe('identify()', () => {
    it('parses valid JSON response into VlmFoodResult', async () => {
      const validJson = JSON.stringify({
        dishes: [{ name: 'pad thai', cuisine: 'Thai', ingredients: ['noodles'] }],
      });
      mockModule.identifyFood.mockResolvedValue(validJson);

      const result = await geminiNanoService.identify('file:///photo.jpg');

      expect(result).toEqual({
        dishes: [{ name: 'pad thai', cuisine: 'Thai', ingredients: ['noodles'] }],
      });
    });

    it('returns { dishes: [] } on JSON parse failure', async () => {
      mockModule.identifyFood.mockResolvedValue('not valid json');

      const result = await geminiNanoService.identify('file:///photo.jpg');

      expect(result).toEqual({ dishes: [] });
    });

    it('captures raw output accessible via getLastRawOutput()', async () => {
      const rawString = 'some raw output from gemini nano';
      mockModule.identifyFood.mockResolvedValue(rawString);

      await geminiNanoService.identify('file:///photo.jpg');

      expect(geminiNanoService.getLastRawOutput()).toBe(rawString);
    });
  });

  describe('isAvailable()', () => {
    it('caches availability result (calls native only once)', async () => {
      mockModule.checkAvailability.mockResolvedValue('available');

      const first = await geminiNanoService.isAvailable();
      const second = await geminiNanoService.isAvailable();

      expect(first).toBe(true);
      expect(second).toBe(true);
      expect(mockModule.checkAvailability).toHaveBeenCalledTimes(1);
    });

    it('returns false when status is not_supported', async () => {
      mockModule.checkAvailability.mockResolvedValue('not_supported');

      const result = await geminiNanoService.isAvailable();

      expect(result).toBe(false);
    });
  });
});
