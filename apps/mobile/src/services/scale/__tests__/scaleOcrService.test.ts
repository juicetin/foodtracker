/**
 * Unit tests for scaleOcrService -- Scale weight extraction via Gemini Nano.
 *
 * Uses Jest auto-mock at src/__mocks__/gemini-nano.ts for the native module.
 */

jest.mock('gemini-nano');

import { geminiNanoModule } from 'gemini-nano';
import {
  readScaleWeight,
  parseScaleResponse,
  convertToGrams,
  SCALE_OCR_PROMPT,
} from '../scaleOcrService';
import type { ScaleReading } from '../scaleOcrService';

const mockModule = geminiNanoModule as jest.Mocked<typeof geminiNanoModule>;

beforeEach(() => {
  jest.clearAllMocks();
});

describe('scaleOcrService', () => {
  describe('SCALE_OCR_PROMPT', () => {
    it('is defined and non-empty', () => {
      expect(SCALE_OCR_PROMPT).toBeTruthy();
      expect(SCALE_OCR_PROMPT.length).toBeGreaterThan(10);
    });
  });

  describe('readScaleWeight()', () => {
    it('returns ScaleReading with source gemini-nano when Gemini Nano returns valid JSON', async () => {
      mockModule.identifyFood.mockResolvedValue(
        JSON.stringify({ weight: 250, unit: 'g' }),
      );

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).toEqual<ScaleReading>({
        weightG: 250,
        unit: 'g',
        confidence: 'high',
        source: 'gemini-nano',
      });
    });

    it('returns null when Gemini Nano returns JSON with null weight', async () => {
      mockModule.identifyFood.mockResolvedValue(
        JSON.stringify({ weight: null, unit: null }),
      );

      const result = await readScaleWeight('file:///no-scale.jpg');

      expect(result).toBeNull();
    });

    it('converts kg to grams', async () => {
      mockModule.identifyFood.mockResolvedValue(
        JSON.stringify({ weight: 1.5, unit: 'kg' }),
      );

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).not.toBeNull();
      expect(result!.weightG).toBe(1500);
    });

    it('converts oz to grams', async () => {
      mockModule.identifyFood.mockResolvedValue(
        JSON.stringify({ weight: 1, unit: 'oz' }),
      );

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).not.toBeNull();
      expect(result!.weightG).toBeCloseTo(28.35, 1);
    });

    it('rejects unreasonable values (< 0.1g) and returns null', async () => {
      mockModule.identifyFood.mockResolvedValue(
        JSON.stringify({ weight: 0.01, unit: 'g' }),
      );

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).toBeNull();
    });

    it('rejects unreasonable values (> 50000g) and returns null', async () => {
      mockModule.identifyFood.mockResolvedValue(
        JSON.stringify({ weight: 60000, unit: 'g' }),
      );

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).toBeNull();
    });

    it('returns null gracefully when Gemini Nano throws', async () => {
      mockModule.identifyFood.mockRejectedValue(new Error('AICore busy'));

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).toBeNull();
    });

    it('returns null gracefully when Gemini Nano returns unparseable text', async () => {
      mockModule.identifyFood.mockResolvedValue('I see a kitchen scale showing 250g');

      const result = await readScaleWeight('file:///scale.jpg');

      expect(result).toBeNull();
    });
  });

  describe('parseScaleResponse()', () => {
    it('parses valid JSON with weight and unit', () => {
      const result = parseScaleResponse('{"weight": 100, "unit": "g"}');
      expect(result).toEqual({ weight: 100, unit: 'g' });
    });

    it('handles comma-separated numbers correctly', () => {
      const result = parseScaleResponse('{"weight": "1,234", "unit": "g"}');
      expect(result).toEqual({ weight: 1234, unit: 'g' });
    });

    it('returns null for null weight', () => {
      const result = parseScaleResponse('{"weight": null, "unit": null}');
      expect(result).toBeNull();
    });

    it('returns null for unparseable text', () => {
      const result = parseScaleResponse('not json at all');
      expect(result).toBeNull();
    });

    it('returns null for weight outside valid range', () => {
      const result = parseScaleResponse('{"weight": 0.05, "unit": "g"}');
      expect(result).toBeNull();
    });
  });

  describe('convertToGrams()', () => {
    it('passes through grams unchanged', () => {
      expect(convertToGrams(500, 'g')).toBe(500);
    });

    it('converts kg to grams', () => {
      expect(convertToGrams(1.5, 'kg')).toBe(1500);
    });

    it('converts oz to grams', () => {
      expect(convertToGrams(1, 'oz')).toBeCloseTo(28.3495, 2);
    });

    it('converts lb to grams', () => {
      expect(convertToGrams(1, 'lb')).toBeCloseTo(453.592, 1);
    });

    it('defaults to grams for unknown unit', () => {
      expect(convertToGrams(100, 'unknown')).toBe(100);
    });
  });
});
