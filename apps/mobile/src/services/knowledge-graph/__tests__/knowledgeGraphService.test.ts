/**
 * Tests for KnowledgeGraphService.
 *
 * Mocks the op-sqlite connection to test the full search -> recipe -> nutrition flow.
 */

// ── Mock db/client ──
const mockExecute = jest.fn().mockResolvedValue({ rows: [] });
const mockClose = jest.fn();
const mockConnection = {
  execute: mockExecute,
  close: mockClose,
};

jest.mock('../../../../db/client', () => ({
  openNutritionDb: jest.fn(() => mockConnection),
}));

// Mock SymSpellIndex so we can control its behavior
jest.mock('../symspellIndex', () => {
  return {
    SymSpellIndex: jest.fn().mockImplementation(() => ({
      loadFromDb: jest.fn().mockResolvedValue(undefined),
      lookup: jest.fn().mockReturnValue([]),
    })),
  };
});

import { KnowledgeGraphService } from '../knowledgeGraphService';
import { SymSpellIndex } from '../symspellIndex';

// ── Mock data ──

const MOCK_DISH = {
  id: 1,
  canonical_name: 'pad thai',
  avg_calories_per_serving: 400,
  avg_protein_per_serving: 15,
  avg_carbs_per_serving: 50,
  avg_fat_per_serving: 14,
  default_serving_grams: 300,
};

const MOCK_RECIPE = {
  id: 10,
  dish_id: 1,
  name: 'Classic Pad Thai',
  source: 'RecipeDB',
  total_weight_grams: 500,
  servings: 2,
  is_canonical: 1,
};

const MOCK_INGREDIENTS = [
  {
    id: 100,
    recipe_id: 10,
    usda_fdc_id: 2001,
    ingredient_name: 'rice noodles',
    quantity_grams: 200,
    calories_per_100g: 360,
    protein_per_100g: 3.4,
    fat_per_100g: 0.6,
    carbs_per_100g: 83,
  },
  {
    id: 101,
    recipe_id: 10,
    usda_fdc_id: 2002,
    ingredient_name: 'shrimp',
    quantity_grams: 150,
    calories_per_100g: 85,
    protein_per_100g: 20.1,
    fat_per_100g: 0.5,
    carbs_per_100g: 0.0,
  },
  {
    id: 102,
    recipe_id: 10,
    usda_fdc_id: 2003,
    ingredient_name: 'peanuts',
    quantity_grams: 30,
    calories_per_100g: 567,
    protein_per_100g: 25.8,
    fat_per_100g: 49.2,
    carbs_per_100g: 16.1,
  },
  {
    id: 103,
    recipe_id: 10,
    usda_fdc_id: 2004,
    ingredient_name: 'vegetable oil',
    quantity_grams: 30,
    calories_per_100g: 884,
    protein_per_100g: 0.0,
    fat_per_100g: 100,
    carbs_per_100g: 0.0,
  },
];

describe('KnowledgeGraphService', () => {
  let service: KnowledgeGraphService;

  beforeEach(async () => {
    jest.clearAllMocks();
    mockExecute.mockResolvedValue({ rows: [] });
    service = new KnowledgeGraphService();
    await service.open('/mock/path/knowledge-graph.db');
  });

  afterEach(() => {
    service.close();
  });

  describe('searchDish', () => {
    it('finds a dish via FTS5 match', async () => {
      // FTS5 on dish_fts returns the dish
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });

      const result = await service.searchDish('pad thai');

      expect(result).not.toBeNull();
      expect(result!.canonicalName).toBe('pad thai');
      expect(result!.id).toBe(1);
      expect(mockExecute).toHaveBeenCalledWith(
        expect.stringContaining('dish_fts'),
        expect.arrayContaining(['pad thai*'])
      );
    });

    it('normalizes Title Case input to lowercase', async () => {
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });

      const result = await service.searchDish('Pad Thai');

      expect(result).not.toBeNull();
      expect(result!.canonicalName).toBe('pad thai');
      expect(mockExecute).toHaveBeenCalledWith(
        expect.stringContaining('dish_fts'),
        expect.arrayContaining(['pad thai*'])
      );
    });

    it('falls back to dish_alias_fts when dish_fts misses', async () => {
      // First FTS5 miss
      mockExecute.mockResolvedValueOnce({ rows: [] });
      // Alias FTS5 hit
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });

      const result = await service.searchDish('thai stir fry noodles');

      expect(result).not.toBeNull();
      expect(mockExecute).toHaveBeenCalledTimes(2);
    });

    it('falls back to SymSpell when both FTS5 searches miss', async () => {
      // Both FTS5 miss
      mockExecute.mockResolvedValueOnce({ rows: [] });
      mockExecute.mockResolvedValueOnce({ rows: [] });

      // SymSpell returns a match
      const symspellInstance = (SymSpellIndex as jest.Mock).mock.results[0]
        ?.value;
      if (symspellInstance) {
        symspellInstance.lookup.mockReturnValue([
          { term: 'pad thai', distance: 1, dishId: 1 },
        ]);
      }

      // DB lookup by ID returns the dish
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });

      const result = await service.searchDish('pad thia');

      expect(result).not.toBeNull();
      expect(result!.canonicalName).toBe('pad thai');
    });

    it('returns null when all search methods fail', async () => {
      // Both FTS5 miss
      mockExecute.mockResolvedValueOnce({ rows: [] });
      mockExecute.mockResolvedValueOnce({ rows: [] });
      // SymSpell returns no matches
      const symspellInstance = (SymSpellIndex as jest.Mock).mock.results[0]
        ?.value;
      if (symspellInstance) {
        symspellInstance.lookup.mockReturnValue([]);
      }

      const result = await service.searchDish('completely unknown dish');

      expect(result).toBeNull();
    });

    it('normalizes hyphens and underscores to spaces', async () => {
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });

      await service.searchDish('Pad_Thai');

      expect(mockExecute).toHaveBeenCalledWith(
        expect.stringContaining('dish_fts'),
        expect.arrayContaining(['pad thai*'])
      );
    });
  });

  describe('calculateDishNutrition', () => {
    it('returns MacroResult with source="recipe" when recipe exists', async () => {
      // searchDish finds the dish
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });
      // getCanonicalRecipe finds the recipe
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_RECIPE] });
      // getRecipeIngredients returns ingredients with USDA data
      mockExecute.mockResolvedValueOnce({ rows: MOCK_INGREDIENTS });

      const result = await service.calculateDishNutrition('pad thai', 300);

      expect(result).not.toBeNull();
      expect(result!.source).toBe('recipe');
      expect(result!.weightGrams).toBe(300);
      // Verify scaling: 300g portion from 500g recipe = 0.6 scale factor
      // Total recipe calories: (200*360/100) + (150*85/100) + (30*567/100) + (30*884/100)
      //                      = 720 + 127.5 + 170.1 + 265.2 = 1282.8
      // Scaled: 1282.8 * 300/500 = 769.68
      expect(result!.calories).toBeCloseTo(769.68, 0);
    });

    it('returns null when dish not found', async () => {
      // searchDish returns no results
      mockExecute.mockResolvedValueOnce({ rows: [] });
      mockExecute.mockResolvedValueOnce({ rows: [] });
      const symspellInstance = (SymSpellIndex as jest.Mock).mock.results[0]
        ?.value;
      if (symspellInstance) {
        symspellInstance.lookup.mockReturnValue([]);
      }

      const result = await service.calculateDishNutrition('unknown dish', 300);

      expect(result).toBeNull();
    });

    it('falls back to dish averages when no recipe exists', async () => {
      // searchDish finds the dish
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });
      // getCanonicalRecipe returns no recipe
      mockExecute.mockResolvedValueOnce({ rows: [] });

      const result = await service.calculateDishNutrition('pad thai', 300);

      expect(result).not.toBeNull();
      expect(result!.source).toBe('dish_average');
      expect(result!.weightGrams).toBe(300);
      // Dish averages: 400 cal per 300g serving, scaled to 300g portion
      // Scale factor: 300 / 300 (default_serving_grams) = 1.0
      expect(result!.calories).toBeCloseTo(400, 0);
    });

    it('scales portion correctly based on recipe total weight', async () => {
      // searchDish finds the dish
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });
      // getCanonicalRecipe returns recipe with 500g total
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_RECIPE] });
      // ingredients
      mockExecute.mockResolvedValueOnce({ rows: MOCK_INGREDIENTS });

      // Request 500g (full recipe)
      const result = await service.calculateDishNutrition('pad thai', 500);

      expect(result).not.toBeNull();
      // Full recipe: no scaling down
      // Total recipe protein: (200*3.4/100) + (150*20.1/100) + (30*25.8/100) + (30*0/100)
      //                     = 6.8 + 30.15 + 7.74 + 0 = 44.69
      expect(result!.protein).toBeCloseTo(44.69, 0);
    });
  });

  describe('getDishAverages', () => {
    it('returns dish-level avg macros when available', async () => {
      // searchDish finds the dish
      mockExecute.mockResolvedValueOnce({ rows: [MOCK_DISH] });

      const result = await service.getDishAverages('pad thai');

      expect(result).not.toBeNull();
      expect(result!.avgCaloriesPerServing).toBe(400);
      expect(result!.avgProteinPerServing).toBe(15);
      expect(result!.avgCarbsPerServing).toBe(50);
      expect(result!.avgFatPerServing).toBe(14);
      expect(result!.defaultServingGrams).toBe(300);
    });

    it('returns null for unknown dish', async () => {
      // Both FTS5 miss
      mockExecute.mockResolvedValueOnce({ rows: [] });
      mockExecute.mockResolvedValueOnce({ rows: [] });
      const symspellInstance = (SymSpellIndex as jest.Mock).mock.results[0]
        ?.value;
      if (symspellInstance) {
        symspellInstance.lookup.mockReturnValue([]);
      }

      const result = await service.getDishAverages('unknown dish');

      expect(result).toBeNull();
    });
  });

  describe('close', () => {
    it('closes the database connection', () => {
      service.close();
      expect(mockClose).toHaveBeenCalled();
    });
  });
});
