/**
 * Tests for VLM identification pipeline.
 *
 * Verifies primary VLM identification with retry, positional
 * assignment to bounding boxes, and text fallback support.
 * VLM is required (throws when not ready).
 */

import { vlmService } from '../vlmService';
import {
  getKnowledgeGraphService,
  type KnowledgeGraphService,
} from '../../knowledge-graph';
import type { DetectedItem } from '../../detection/types';
import type { VlmFoodResult } from '../vlmTypes';
import { runVlmIdentification, identifyWithRetry, assignDishesToBoxes } from '../vlmPipeline';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

jest.mock('../vlmService', () => ({
  vlmService: {
    isReady: false,
    identify: jest.fn(),
  },
}));

jest.mock('../../knowledge-graph', () => ({
  getKnowledgeGraphService: jest.fn().mockResolvedValue(null),
}));

const mockVlmService = vlmService as unknown as {
  isReady: boolean;
  identify: jest.Mock;
};

const mockGetKG = getKnowledgeGraphService as jest.Mock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeItem(overrides: Partial<DetectedItem> = {}): DetectedItem {
  return {
    id: 'item-1',
    className: 'Food Region',
    confidence: 0.85,
    bbox: { x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
    portionEstimate: {
      weightG: 200,
      confidence: 'medium',
      method: 'geometry',
      suggestReference: false,
      details: {},
    },
    portionMultiplier: 1.0,
    isRemoved: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  jest.clearAllMocks();
  mockVlmService.isReady = false;
  mockVlmService.identify.mockReset();
  mockGetKG.mockResolvedValue(null);
});

describe('identifyWithRetry', () => {
  it('succeeds on first attempt (identify called once)', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [{ name: 'Pad Thai', cuisine: 'Thai', ingredients: ['noodles'] }],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const result = await identifyWithRetry('file:///photo.jpg', 'pad thai');

    expect(result).toEqual(vlmResult);
    expect(mockVlmService.identify).toHaveBeenCalledTimes(1);
    expect(mockVlmService.identify).toHaveBeenCalledWith('file:///photo.jpg', 'pad thai');
  });

  it('succeeds on second attempt after first failure (identify called twice)', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [{ name: 'Green Curry', cuisine: 'Thai', ingredients: ['coconut'] }],
    };
    mockVlmService.identify.mockRejectedValueOnce(new Error('Timeout'));
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const result = await identifyWithRetry('file:///photo.jpg');

    expect(result).toEqual(vlmResult);
    expect(mockVlmService.identify).toHaveBeenCalledTimes(2);
  });

  it('returns empty dishes after both attempts fail (no throw)', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockRejectedValueOnce(new Error('Timeout'));
    mockVlmService.identify.mockRejectedValueOnce(new Error('Timeout again'));

    const result = await identifyWithRetry('file:///photo.jpg');

    expect(result).toEqual({ dishes: [] });
    expect(mockVlmService.identify).toHaveBeenCalledTimes(2);
  });

  it('passes userText to both attempts', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockRejectedValueOnce(new Error('Timeout'));
    mockVlmService.identify.mockResolvedValueOnce({ dishes: [] });

    await identifyWithRetry('file:///photo.jpg', 'ramen with egg');

    expect(mockVlmService.identify).toHaveBeenCalledTimes(2);
    expect(mockVlmService.identify).toHaveBeenNthCalledWith(1, 'file:///photo.jpg', 'ramen with egg');
    expect(mockVlmService.identify).toHaveBeenNthCalledWith(2, 'file:///photo.jpg', 'ramen with egg');
  });
});

describe('assignDishesToBoxes', () => {
  it('assigns by area descending (largest box gets first dish)', () => {
    const items = [
      makeItem({ id: 'small', bbox: { x: 0, y: 0, w: 0.1, h: 0.1 } }),  // area 0.01
      makeItem({ id: 'large', bbox: { x: 0, y: 0, w: 0.5, h: 0.5 } }),  // area 0.25
      makeItem({ id: 'medium', bbox: { x: 0, y: 0, w: 0.3, h: 0.2 } }), // area 0.06
    ];

    const result = assignDishesToBoxes(items, ['Pad Thai', 'Green Curry', 'Rice']);

    expect(result.get('large')).toBe('Pad Thai');
    expect(result.get('medium')).toBe('Green Curry');
    expect(result.get('small')).toBe('Rice');
  });

  it('handles more dishes than boxes (extra dishes go unmatched)', () => {
    const items = [
      makeItem({ id: 'item-1', bbox: { x: 0, y: 0, w: 0.5, h: 0.5 } }),
    ];

    const result = assignDishesToBoxes(items, ['Pad Thai', 'Green Curry', 'Rice']);

    expect(result.size).toBe(1);
    expect(result.get('item-1')).toBe('Pad Thai');
  });

  it('handles more boxes than dishes (extra boxes get no assignment)', () => {
    const items = [
      makeItem({ id: 'large', bbox: { x: 0, y: 0, w: 0.5, h: 0.5 } }),
      makeItem({ id: 'medium', bbox: { x: 0, y: 0, w: 0.3, h: 0.2 } }),
      makeItem({ id: 'small', bbox: { x: 0, y: 0, w: 0.1, h: 0.1 } }),
    ];

    const result = assignDishesToBoxes(items, ['Pad Thai']);

    expect(result.size).toBe(1);
    expect(result.get('large')).toBe('Pad Thai');
    expect(result.has('medium')).toBe(false);
    expect(result.has('small')).toBe(false);
  });

  it('skips removed items', () => {
    const items = [
      makeItem({ id: 'removed', bbox: { x: 0, y: 0, w: 0.8, h: 0.8 }, isRemoved: true }),
      makeItem({ id: 'active', bbox: { x: 0, y: 0, w: 0.3, h: 0.3 } }),
    ];

    const result = assignDishesToBoxes(items, ['Pad Thai']);

    expect(result.size).toBe(1);
    expect(result.get('active')).toBe('Pad Thai');
    expect(result.has('removed')).toBe(false);
  });
});

describe('runVlmIdentification', () => {
  it('throws when VLM not ready', async () => {
    mockVlmService.isReady = false;
    const items = [makeItem()];
    await expect(runVlmIdentification('file:///photo.jpg', items)).rejects.toThrow(
      'VLM model is not loaded',
    );
    expect(mockVlmService.identify).not.toHaveBeenCalled();
  });

  it('matches VLM dishes to items by positional assignment (area descending)', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [
        { name: 'Pad Thai', cuisine: 'Thai', ingredients: ['noodles', 'shrimp'] },
        { name: 'Green Curry', cuisine: 'Thai', ingredients: ['coconut'] },
      ],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const items = [
      makeItem({ id: 'small', bbox: { x: 0, y: 0, w: 0.2, h: 0.2 } }),
      makeItem({ id: 'large', bbox: { x: 0, y: 0, w: 0.5, h: 0.5 } }),
    ];
    const result = await runVlmIdentification('file:///photo.jpg', items);

    // Largest bbox gets first VLM dish
    const largeItem = result.find((i) => i.id === 'large')!;
    const smallItem = result.find((i) => i.id === 'small')!;

    expect(largeItem.vlmLabel).toBe('Pad Thai');
    expect(largeItem.vlmCuisine).toBe('Thai');
    expect(largeItem.vlmIngredients).toEqual(['noodles', 'shrimp']);

    expect(smallItem.vlmLabel).toBe('Green Curry');
    expect(smallItem.vlmCuisine).toBe('Thai');
    expect(smallItem.vlmIngredients).toEqual(['coconut']);
  });

  it('on empty VLM dishes, returns items unchanged', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockResolvedValueOnce({ dishes: [] });

    const items = [makeItem({ id: 'item-1' })];
    const result = await runVlmIdentification('file:///photo.jpg', items);

    expect(result).toHaveLength(1);
    expect(result[0].vlmLabel).toBeUndefined();
    // isRefining should stay as-is for caller to handle fallback
    expect(result[0].id).toBe('item-1');
  });

  it('passes userText to identify', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockResolvedValueOnce({ dishes: [] });

    const items = [makeItem()];
    await runVlmIdentification('file:///photo.jpg', items, 'pad thai with shrimp');

    expect(mockVlmService.identify).toHaveBeenCalledWith(
      'file:///photo.jpg',
      'pad thai with shrimp',
    );
  });

  it('KG nutrition lookup called for matched VLM dishes', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [
        { name: 'Massaman Curry', cuisine: 'Thai', ingredients: ['potato'] },
      ],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const mockKGService = {
      searchDish: jest.fn().mockResolvedValue({
        id: 42,
        canonicalName: 'massaman curry',
        avgCaloriesPerServing: 300,
        avgProteinPerServing: 15,
        avgCarbsPerServing: 25,
        avgFatPerServing: 18,
        defaultServingGrams: 250,
      }),
      calculateDishNutrition: jest.fn().mockResolvedValue({
        calories: 240,
        protein: 12,
        carbs: 20,
        fat: 14,
        weightGrams: 200,
        source: 'recipe',
      }),
    } as unknown as KnowledgeGraphService;
    mockGetKG.mockResolvedValue(mockKGService);

    const items = [makeItem({ id: 'item-1' })];
    await runVlmIdentification('file:///photo.jpg', items);

    expect(mockKGService.searchDish).toHaveBeenCalledWith('Massaman Curry');
  });

  it('uses retry logic (succeeds on second VLM attempt)', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [{ name: 'Ramen', cuisine: 'Japanese', ingredients: ['noodles'] }],
    };
    mockVlmService.identify.mockRejectedValueOnce(new Error('Timeout'));
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const items = [makeItem({ id: 'item-1' })];
    const result = await runVlmIdentification('file:///photo.jpg', items);

    expect(result[0].vlmLabel).toBe('Ramen');
    expect(mockVlmService.identify).toHaveBeenCalledTimes(2);
  });

  it('returns items unchanged when VLM fails after retry', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockRejectedValueOnce(new Error('Fail 1'));
    mockVlmService.identify.mockRejectedValueOnce(new Error('Fail 2'));

    const items = [makeItem({ id: 'item-1' })];
    const result = await runVlmIdentification('file:///photo.jpg', items);

    expect(result).toHaveLength(1);
    expect(result[0].vlmLabel).toBeUndefined();
  });
});
