/**
 * Tests for VLM scan pipeline (scanFood).
 *
 * Verifies Gemini Nano as Tier 0 identification,
 * mock fallback when unavailable, KG nutrition lookup,
 * and model source tracking.
 */

import { geminiNanoModule } from 'gemini-nano';
import { geminiNanoService } from '../geminiNanoService';
import { getMockScanResult } from '../geminiNanoMock';
import {
  getKnowledgeGraphService,
  type KnowledgeGraphService,
} from '../../knowledge-graph';
import { scanFood, getLastVlmSource, _resetVlmSource } from '../vlmPipeline';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

jest.mock('gemini-nano');

jest.mock('../geminiNanoService', () => ({
  geminiNanoService: {
    isAvailable: jest.fn().mockResolvedValue(false),
    identify: jest.fn().mockResolvedValue({ dishes: [] }),
    getLastRawOutput: jest.fn().mockReturnValue(null),
    _resetCache: jest.fn(),
  },
}));

jest.mock('../geminiNanoMock', () => ({
  getMockScanResult: jest.fn().mockReturnValue({
    dishes: [
      {
        name: 'Mock Chicken Rice',
        cuisine: 'Asian',
        ingredients: [
          { name: 'rice', amount_g: 180 },
          { name: 'chicken', amount_g: 150 },
        ],
      },
    ],
  }),
}));

jest.mock('../../knowledge-graph', () => ({
  getKnowledgeGraphService: jest.fn().mockResolvedValue(null),
}));

const mockGeminiModule = geminiNanoModule as jest.Mocked<typeof geminiNanoModule>;
const mockGeminiService = geminiNanoService as unknown as {
  isAvailable: jest.Mock;
  identify: jest.Mock;
  getLastRawOutput: jest.Mock;
  _resetCache: jest.Mock;
};
const mockGetKG = getKnowledgeGraphService as jest.Mock;
const mockGetMock = getMockScanResult as jest.Mock;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  jest.clearAllMocks();
  mockGeminiModule.checkAvailability.mockResolvedValue('not_supported');
  mockGeminiService.isAvailable.mockResolvedValue(false);
  mockGeminiService.identify.mockResolvedValue({ dishes: [] });
  mockGetKG.mockResolvedValue(null);
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Tier 0/1 routing', () => {
  it('uses Gemini Nano when checkAvailability returns "available"', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({
      dishes: [
        { name: 'Sushi', cuisine: 'Japanese', ingredients: [{ name: 'rice', amount_g: 120 }, { name: 'salmon', amount_g: 80 }] },
      ],
    });

    const result = await scanFood('file:///test.jpg');

    expect(mockGeminiService.identify).toHaveBeenCalledWith('file:///test.jpg');
    expect(result.isMock).toBe(false);
    expect(result.dishes).toHaveLength(1);
    expect(result.dishes[0].name).toBe('Sushi');
  });

  it('falls through to mock when checkAvailability returns "not_supported"', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('not_supported');

    const result = await scanFood('file:///test.jpg');

    expect(mockGeminiService.identify).not.toHaveBeenCalled();
    expect(result.isMock).toBe(true);
    expect(result.dishes).toHaveLength(1);
    expect(result.dishes[0].name).toBe('Mock Chicken Rice');
  });

  it('falls through to mock when Gemini Nano returns empty dishes', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({ dishes: [] });

    const result = await scanFood('file:///test.jpg');

    expect(mockGeminiService.identify).toHaveBeenCalled();
    expect(result.isMock).toBe(true);
  });

  it('falls through to mock when Gemini Nano throws', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockRejectedValue(new Error('AICore crashed'));

    const result = await scanFood('file:///test.jpg');

    expect(result.isMock).toBe(true);
    expect(result.dishes.length).toBeGreaterThan(0);
  });
});

describe('scanFood result structure', () => {
  it('returns ScanResult with photoUri, dishes, isMock', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({
      dishes: [
        { name: 'Pad Thai', cuisine: 'Thai', ingredients: [{ name: 'noodles', amount_g: 200 }] },
      ],
    });

    const result = await scanFood('file:///food.jpg');

    expect(result.photoUri).toBe('file:///food.jpg');
    expect(result.isMock).toBe(false);
    expect(result.dishes[0]).toHaveProperty('id');
    expect(result.dishes[0]).toHaveProperty('name', 'Pad Thai');
    expect(result.dishes[0]).toHaveProperty('cuisine', 'Thai');
    expect(result.dishes[0]).toHaveProperty('portionScale', 1.0);
    expect(result.dishes[0].ingredients).toHaveLength(1);
    expect(result.dishes[0].ingredients[0]).toHaveProperty('name', 'noodles');
    expect(result.dishes[0].ingredients[0]).toHaveProperty('amount_g', 200);
    expect(result.dishes[0].ingredients[0]).toHaveProperty('calories');
  });

  it('assigns unique IDs to dishes and ingredients', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({
      dishes: [
        { name: 'Dish A', cuisine: null, ingredients: [{ name: 'ing1', amount_g: 50 }, { name: 'ing2', amount_g: 30 }] },
        { name: 'Dish B', cuisine: null, ingredients: [{ name: 'ing3', amount_g: 100 }] },
      ],
    });

    const result = await scanFood('file:///test.jpg');

    const ids = [
      result.dishes[0].id,
      result.dishes[1].id,
      result.dishes[0].ingredients[0].id,
      result.dishes[0].ingredients[1].id,
      result.dishes[1].ingredients[0].id,
    ];
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });
});

describe('KG nutrition lookup', () => {
  it('uses KG nutrition when available', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({
      dishes: [
        { name: 'Rice Bowl', cuisine: 'Asian', ingredients: [{ name: 'rice', amount_g: 200 }] },
      ],
    });

    const mockKGService = {
      searchDish: jest.fn().mockResolvedValue(null),
      // USDA returns null → KG recipe path is used unconditionally
      lookupUsdaIngredient: jest.fn().mockResolvedValue(null),
      calculateDishNutrition: jest.fn().mockResolvedValue({
        calories: 260,
        protein: 5,
        carbs: 57,
        fat: 0.4,
        weightGrams: 200,
        source: 'recipe',
      }),
    } as unknown as KnowledgeGraphService;
    mockGetKG.mockResolvedValue(mockKGService);

    const result = await scanFood('file:///test.jpg');

    expect(mockKGService.lookupUsdaIngredient).toHaveBeenCalledWith('rice', 200);
    expect(mockKGService.calculateDishNutrition).toHaveBeenCalledWith('rice', 200);
    expect(result.dishes[0].ingredients[0].nutritionSource).toBe('kg');
    expect(result.dishes[0].ingredients[0].calories).toBe(260);
  });

  it('falls back to proxy nutrition when KG unavailable', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({
      dishes: [
        { name: 'Toast', cuisine: null, ingredients: [{ name: 'bread', amount_g: 60 }] },
      ],
    });
    mockGetKG.mockResolvedValue(null);

    const result = await scanFood('file:///test.jpg');

    expect(result.dishes[0].ingredients[0].nutritionSource).toBe('proxy');
    // Proxy: 60g * 1.5 = 90 kcal
    expect(result.dishes[0].ingredients[0].calories).toBe(90);
  });
});

describe('getLastVlmSource', () => {
  beforeEach(() => {
    _resetVlmSource();
  });

  it('returns null before any scan', () => {
    expect(getLastVlmSource()).toBeNull();
  });

  it('returns "gemini-nano" after successful Gemini Nano scan', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('available');
    mockGeminiService.identify.mockResolvedValue({
      dishes: [{ name: 'Sushi', cuisine: 'Japanese', ingredients: [{ name: 'rice', amount_g: 100 }] }],
    });

    await scanFood('file:///test.jpg');

    expect(getLastVlmSource()).toBe('gemini-nano');
  });

  it('returns "mock" when mock fallback is used', async () => {
    mockGeminiModule.checkAvailability.mockResolvedValue('not_supported');

    await scanFood('file:///test.jpg');

    expect(getLastVlmSource()).toBe('mock');
  });

  it('returns null before any scan', () => {
    expect(getLastVlmSource()).toBeNull();
  });
});
