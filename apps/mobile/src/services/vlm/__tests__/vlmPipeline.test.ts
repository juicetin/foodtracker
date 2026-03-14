/**
 * Tests for VLM refinement pipeline.
 *
 * Verifies YOLO-to-VLM matching, KG nutrition bridge,
 * and that VLM is required (throws when not ready).
 */

import { vlmService } from '../vlmService';
import {
  getKnowledgeGraphService,
  type KnowledgeGraphService,
} from '../../knowledge-graph';
import type { DetectedItem } from '../../detection/types';
import type { VlmFoodResult } from '../vlmTypes';
import { runVlmRefinement } from '../vlmPipeline';

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
    className: 'Curry',
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

describe('runVlmRefinement', () => {
  it('throws when VLM not ready', async () => {
    mockVlmService.isReady = false;
    const items = [makeItem()];
    await expect(runVlmRefinement('file:///photo.jpg', items)).rejects.toThrow(
      'VLM model is not loaded',
    );
    expect(mockVlmService.identify).not.toHaveBeenCalled();
  });

  it('propagates VLM inference errors', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockRejectedValueOnce(new Error('Inference failed'));
    const items = [makeItem()];
    await expect(runVlmRefinement('file:///photo.jpg', items)).rejects.toThrow(
      'Inference failed',
    );
  });

  it('matches VLM dish to YOLO item by substring', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [
        { name: 'Massaman Curry', cuisine: 'Thai', ingredients: ['potato', 'peanut'] },
      ],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const items = [makeItem({ id: 'item-1', className: 'Curry' })];
    const result = await runVlmRefinement('file:///photo.jpg', items);

    expect(result[0].vlmLabel).toBe('Massaman Curry');
    expect(result[0].vlmCuisine).toBe('Thai');
    expect(result[0].vlmIngredients).toEqual(['potato', 'peanut']);
  });

  it('matches VLM dish to YOLO item by word overlap', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [
        { name: 'Pad Thai Noodles', cuisine: 'Thai', ingredients: ['noodles', 'shrimp'] },
      ],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const items = [makeItem({ id: 'item-1', className: 'Pad Thai' })];
    const result = await runVlmRefinement('file:///photo.jpg', items);

    expect(result[0].vlmLabel).toBe('Pad Thai Noodles');
  });

  it('unmatched VLM dishes do not create phantom items', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [
        { name: 'Pad Thai', cuisine: 'Thai', ingredients: ['noodles'] },
        { name: 'Green Curry', cuisine: 'Thai', ingredients: ['coconut'] },
        { name: 'Mango Sticky Rice', cuisine: 'Thai', ingredients: ['mango'] },
      ],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const items = [
      makeItem({ id: 'item-1', className: 'Pad Thai' }),
      makeItem({ id: 'item-2', className: 'Curry' }),
    ];
    const result = await runVlmRefinement('file:///photo.jpg', items);

    // Only 2 items returned (same as input), no phantom items
    expect(result).toHaveLength(2);
  });

  it('unmatched YOLO items keep original labels', async () => {
    mockVlmService.isReady = true;
    const vlmResult: VlmFoodResult = {
      dishes: [
        { name: 'Massaman Curry', cuisine: 'Thai', ingredients: ['potato'] },
      ],
    };
    mockVlmService.identify.mockResolvedValueOnce(vlmResult);

    const items = [
      makeItem({ id: 'item-1', className: 'Curry' }),
      makeItem({ id: 'item-2', className: 'Rice' }),
      makeItem({ id: 'item-3', className: 'Salad' }),
    ];
    const result = await runVlmRefinement('file:///photo.jpg', items);

    // Only item-1 should have vlmLabel (matched "Curry" to "Massaman Curry")
    expect(result[0].vlmLabel).toBe('Massaman Curry');
    // Other items should keep original labels (no vlmLabel)
    expect(result[1].vlmLabel).toBeUndefined();
    expect(result[2].vlmLabel).toBeUndefined();
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

    const items = [makeItem({ id: 'item-1', className: 'Curry' })];
    await runVlmRefinement('file:///photo.jpg', items);

    expect(mockKGService.searchDish).toHaveBeenCalledWith('Massaman Curry');
  });

  it('passes userText to vlmService.identify', async () => {
    mockVlmService.isReady = true;
    mockVlmService.identify.mockResolvedValueOnce({ dishes: [] });

    const items = [makeItem()];
    await runVlmRefinement('file:///photo.jpg', items, 'pad thai with shrimp');

    expect(mockVlmService.identify).toHaveBeenCalledWith(
      'file:///photo.jpg',
      'pad thai with shrimp',
    );
  });
});
