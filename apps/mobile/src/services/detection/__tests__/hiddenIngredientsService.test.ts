/**
 * Tests for hiddenIngredientsService — KG ingredient enrichment for detected dishes.
 */

import { enrichDishesWithKgIngredients } from '../hiddenIngredientsService';
import type { ScannedDish } from '../../../types';

// ── Mocks ──

const mockSearchDish = jest.fn();
const mockGetCanonicalRecipe = jest.fn();
const mockGetRecipeIngredients = jest.fn();

jest.mock('../../knowledge-graph', () => ({
  getKnowledgeGraphService: jest.fn(async () => ({
    searchDish: mockSearchDish,
    getCanonicalRecipe: mockGetCanonicalRecipe,
    getRecipeIngredients: mockGetRecipeIngredients,
  })),
}));

// ── Fixtures ──

function makeDish(overrides: Partial<ScannedDish> = {}): ScannedDish {
  return {
    id: 'dish-1',
    name: 'Carbonara',
    cuisine: 'Italian',
    photoUri: 'file:///photo.jpg',
    ingredients: [],
    portionScale: 1.0,
    ...overrides,
  };
}

function makeIngredient(name: string, amount_g: number) {
  return {
    id: `ing-${name}`,
    name,
    amount_g,
    originalAmount_g: amount_g,
    calories: 100,
    protein: 10,
    carbs: 20,
    fat: 5,
    fiber: 0,
    sodium: 0,
    nutritionSource: 'kg' as const,
    userModified: false,
  };
}

const MOCK_KG_DISH = {
  id: 42,
  canonicalName: 'carbonara',
  avgCaloriesPerServing: 500,
  avgProteinPerServing: 25,
  avgCarbsPerServing: 50,
  avgFatPerServing: 20,
  defaultServingGrams: 300,
};

const MOCK_KG_RECIPE = {
  id: 101,
  dishId: 42,
  name: 'Spaghetti Carbonara',
  source: null,
  totalWeightGrams: 600,
  servings: 2,
  isCanonical: true,
};

const MOCK_KG_INGREDIENTS = [
  {
    id: 1,
    recipeId: 101,
    usdaFdcId: 1001,
    ingredientName: 'egg',
    quantityGrams: 100,
    caloriesPer100g: 155,
    proteinPer100g: 13,
    fatPer100g: 11,
    carbsPer100g: 1.1,
  },
  {
    id: 2,
    recipeId: 101,
    usdaFdcId: 1002,
    ingredientName: 'pancetta',
    quantityGrams: 80,
    caloriesPer100g: 380,
    proteinPer100g: 12,
    fatPer100g: 36,
    carbsPer100g: 0.5,
  },
  {
    id: 3,
    recipeId: 101,
    usdaFdcId: 1003,
    ingredientName: 'parmesan',
    quantityGrams: 50,
    caloriesPer100g: 431,
    proteinPer100g: 38,
    fatPer100g: 29,
    carbsPer100g: 4.1,
  },
];

// ── Tests ──

beforeEach(() => {
  jest.clearAllMocks();
});

describe('enrichDishesWithKgIngredients', () => {
  it('returns dishes unchanged when all dishes already have ingredients', async () => {
    const dish = makeDish({
      ingredients: [makeIngredient('egg', 50), makeIngredient('bacon', 30)],
    });

    const result = await enrichDishesWithKgIngredients([dish]);

    expect(result).toHaveLength(1);
    expect(result[0].ingredients).toHaveLength(2);
    expect(result[0].ingredients[0].name).toBe('egg');
    // KG should NOT be called when dish already has ingredients
    expect(mockSearchDish).not.toHaveBeenCalled();
  });

  it('fills ingredient names from KG when dish has empty ingredients array', async () => {
    const dish = makeDish({ ingredients: [] });

    mockSearchDish.mockResolvedValue(MOCK_KG_DISH);
    mockGetCanonicalRecipe.mockResolvedValue(MOCK_KG_RECIPE);
    mockGetRecipeIngredients.mockResolvedValue(MOCK_KG_INGREDIENTS);

    const result = await enrichDishesWithKgIngredients([dish]);

    expect(result).toHaveLength(1);
    expect(result[0].ingredients).toHaveLength(3);
    expect(result[0].ingredients.map((i) => i.name)).toEqual([
      'egg',
      'pancetta',
      'parmesan',
    ]);
    // Verify nutrition is scaled from per-100g values
    const egg = result[0].ingredients[0];
    expect(egg.amount_g).toBe(100);
    expect(egg.calories).toBeCloseTo(155);
    expect(egg.protein).toBeCloseTo(13);
  });

  it('returns dish unchanged when KG has no match for dish name', async () => {
    const dish = makeDish({ name: 'UnknownFood123', ingredients: [] });

    mockSearchDish.mockResolvedValue(null);

    const result = await enrichDishesWithKgIngredients([dish]);

    expect(result).toHaveLength(1);
    expect(result[0].ingredients).toHaveLength(0);
    expect(mockGetCanonicalRecipe).not.toHaveBeenCalled();
  });

  it('marks KG ingredients with kgInferred=true for UI distinction', async () => {
    const dish = makeDish({ ingredients: [] });

    mockSearchDish.mockResolvedValue(MOCK_KG_DISH);
    mockGetCanonicalRecipe.mockResolvedValue(MOCK_KG_RECIPE);
    mockGetRecipeIngredients.mockResolvedValue(MOCK_KG_INGREDIENTS);

    const result = await enrichDishesWithKgIngredients([dish]);

    // All ingredients should be marked as KG-inferred
    for (const ing of result[0].ingredients) {
      expect((ing as any).kgInferred).toBe(true);
    }
    // kgInferredIngredients list should be populated on the dish
    expect(result[0].kgInferredIngredients).toEqual([
      'egg',
      'pancetta',
      'parmesan',
    ]);
  });
});
