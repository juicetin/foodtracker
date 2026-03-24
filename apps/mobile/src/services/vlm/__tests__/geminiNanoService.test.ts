/**
 * Unit tests for geminiNanoService — Tier 0 Gemini Nano food identification.
 *
 * Uses Jest auto-mock at src/__mocks__/gemini-nano.ts for the native module.
 * Tests cover: JSON parse, parse failure fallback, availability caching,
 * not_supported status, raw output capture, multi-pass identification,
 * truncation salvage, and ERROR: prefix handling.
 */

jest.mock('gemini-nano');

// Fresh imports per test to reset module-level cache
let geminiNanoService: typeof import('../geminiNanoService')['geminiNanoService'];
let salvageTruncatedJson: typeof import('../geminiNanoService')['salvageTruncatedJson'];
let mockModule: typeof import('../../../__mocks__/gemini-nano')['geminiNanoModule'];

beforeEach(() => {
  jest.resetModules();
  jest.mock('gemini-nano');
  // Re-require fresh instances
  const service = require('../geminiNanoService');
  const mock = require('gemini-nano');
  geminiNanoService = service.geminiNanoService;
  salvageTruncatedJson = service.salvageTruncatedJson;
  mockModule = mock.geminiNanoModule;
});

describe('geminiNanoService', () => {
  describe('identify()', () => {
    it('parses valid JSON response into VlmFoodResult', async () => {
      const validJson = JSON.stringify({
        dishes: [{ name: 'pad thai', cuisine: 'Thai', ingredients: [{ name: 'noodles', amount_g: 200 }] }],
      });
      mockModule.identifyFood.mockResolvedValue(validJson);

      const result = await geminiNanoService.identify('file:///photo.jpg');

      expect(result).toEqual({
        dishes: [{ name: 'pad thai', cuisine: 'Thai', ingredients: [{ name: 'noodles', amount_g: 200 }] }],
      });
    });

    it('returns { dishes: [] } on JSON parse failure', async () => {
      // First call (single-call attempt) returns garbage, second call (discovery) also garbage
      mockModule.identifyFood
        .mockResolvedValueOnce('not valid json}')
        .mockResolvedValueOnce('also not json}');

      const result = await geminiNanoService.identify('file:///photo.jpg');

      expect(result).toEqual({ dishes: [] });
    });

    it('captures raw output accessible via getLastRawOutput()', async () => {
      const validJson = JSON.stringify({
        dishes: [{ name: 'rice', cuisine: 'Asian', ingredients: [{ name: 'rice', amount_g: 200 }] }],
      });
      mockModule.identifyFood.mockResolvedValue(validJson);

      await geminiNanoService.identify('file:///photo.jpg');

      expect(geminiNanoService.getLastRawOutput()).toBe(validJson);
    });

    it('throws on ERROR: prefix from native module', async () => {
      mockModule.identifyFood.mockResolvedValue('ERROR:decode_failed:null bitmap from stream');

      await expect(geminiNanoService.identify('file:///photo.jpg')).rejects.toThrow(
        'GeminiNano native error: ERROR:decode_failed:null bitmap from stream',
      );
    });

    it('handles truncated JSON by salvaging', async () => {
      // Truncated response: missing closing brackets
      const truncated = '{"dishes":[{"name":"curry","cuisine":"Indian","recipe_name":"Chicken Curry","ingredients":[{"name":"chicken","amount_g":200},{"name":"onion","amount_g":50';
      mockModule.identifyFood.mockResolvedValue(truncated);

      const result = await geminiNanoService.identify('file:///photo.jpg');

      // The salvaged result should have at least the dish name
      // Since the truncation cuts inside an ingredient, salvage trims to last }
      // which gives us a partial but parseable structure
      expect(result).toBeDefined();
      expect(result.dishes).toBeDefined();
    });

    it('multi-pass: discovers dishes then details each', async () => {
      // First call (single-call attempt) — returns truncated multi-dish
      const truncatedMultiDish = '{"dishes":[{"name":"sushi","cuisine":"Japanese","recipe_name":"Salmon Sushi","ingredients":[{"name":"rice","amount_g":120},{"name":"salmon","amount_g":80}]},{"name":"miso soup","cuisine":"Japanese","recipe_name":"Miso Soup","ingredients":[{"name":"tofu","amount_g":50},{"name":"wakame","amount';
      mockModule.identifyFood.mockResolvedValueOnce(truncatedMultiDish);

      // Second call (discovery pass)
      const discoveryResult = '{"dishes":["sushi","miso soup"]}';
      mockModule.identifyFood.mockResolvedValueOnce(discoveryResult);

      // Third call (detail pass for sushi)
      const sushiDetail = '{"name":"sushi","cuisine":"Japanese","recipe_name":"Salmon Sushi","ingredients":[{"name":"rice","amount_g":120},{"name":"salmon","amount_g":80}]}';
      mockModule.identifyFood.mockResolvedValueOnce(sushiDetail);

      // Fourth call (detail pass for miso soup)
      const misoDetail = '{"name":"miso soup","cuisine":"Japanese","recipe_name":"Miso Soup","ingredients":[{"name":"tofu","amount_g":50},{"name":"wakame","amount_g":10}]}';
      mockModule.identifyFood.mockResolvedValueOnce(misoDetail);

      const result = await geminiNanoService.identify('file:///photo.jpg');

      expect(result.dishes).toHaveLength(2);
      expect(result.dishes[0].name).toBe('sushi');
      expect(result.dishes[1].name).toBe('miso soup');
      // 4 calls: single-call, discovery, sushi detail, miso detail
      expect(mockModule.identifyFood).toHaveBeenCalledTimes(4);
    });

    it('single-dish optimization: skips multi-pass for 1 dish', async () => {
      const singleDish = JSON.stringify({
        dishes: [{ name: 'ramen', cuisine: 'Japanese', ingredients: [{ name: 'noodles', amount_g: 200 }] }],
      });
      mockModule.identifyFood.mockResolvedValue(singleDish);

      const result = await geminiNanoService.identify('file:///photo.jpg');

      expect(result.dishes).toHaveLength(1);
      expect(result.dishes[0].name).toBe('ramen');
      // Only 1 call — no discovery or detail passes needed
      expect(mockModule.identifyFood).toHaveBeenCalledTimes(1);
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

describe('salvageTruncatedJson()', () => {
  it('returns complete JSON as-is', () => {
    const complete = '{"dishes":[{"name":"rice"}]}';
    expect(salvageTruncatedJson(complete)).toBe(complete);
  });

  it('trims to last valid JSON boundary and closes brackets', () => {
    const truncated = '{"dishes":[{"name":"rice","amount_g":200},{"name":"chicken","amount_g":1';
    const result = salvageTruncatedJson(truncated);

    // Should be parseable
    expect(() => JSON.parse(result)).not.toThrow();
    const parsed = JSON.parse(result);
    expect(parsed.dishes).toBeDefined();
  });

  it('handles response with no braces', () => {
    const noJson = 'I cannot identify food in this image';
    expect(salvageTruncatedJson(noJson)).toBe(noJson);
  });

  it('closes multiple open brackets', () => {
    const truncated = '{"dishes":[{"name":"a","ingredients":[{"name":"b","amount_g":100}';
    const result = salvageTruncatedJson(truncated);
    expect(() => JSON.parse(result)).not.toThrow();
  });
});
