/**
 * Tests for NutritionService.
 *
 * Mocks op-sqlite connection via the openNutritionDb function from db/client.
 */

// ── Mock db/client ──
const mockExecute = jest.fn();
const mockConnection = {
  execute: mockExecute,
  close: jest.fn(),
};

jest.mock('../../../../db/client', () => ({
  openNutritionDb: jest.fn(() => mockConnection),
}));

import { NutritionService } from '../nutritionService';

describe('NutritionService', () => {
  let service: NutritionService;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new NutritionService();
    service.open('/mock/path/usda-core.db');
  });

  describe('searchFoods', () => {
    it('returns foods matching a query string via FTS5', async () => {
      const mockFoods = [
        {
          fdc_id: 100,
          description: 'Chicken breast raw',
          food_category: 'Poultry Products',
          data_type: 'foundation_food',
          rank: -5.2,
        },
        {
          fdc_id: 500,
          description: 'Chicken thigh raw',
          food_category: 'Poultry Products',
          data_type: 'foundation_food',
          rank: -4.8,
        },
      ];
      mockExecute.mockReturnValue({ rows: mockFoods });

      const results = await service.searchFoods('chicken');

      expect(results).toHaveLength(2);
      expect(results[0].description).toBe('Chicken breast raw');
      expect(mockExecute).toHaveBeenCalledWith(
        expect.stringContaining('foods_fts'),
        expect.arrayContaining(['chicken*'])
      );
    });

    it('returns empty array for no matches', async () => {
      mockExecute.mockReturnValue({ rows: [] });

      const results = await service.searchFoods('xyznonexistent');
      expect(results).toHaveLength(0);
    });

    it('respects limit parameter', async () => {
      mockExecute.mockReturnValue({ rows: [] });

      await service.searchFoods('apple', 5);

      expect(mockExecute).toHaveBeenCalledWith(
        expect.stringContaining('LIMIT'),
        expect.arrayContaining([5])
      );
    });
  });

  describe('getFoodById', () => {
    it('returns a food by fdc_id', async () => {
      const mockFood = {
        fdc_id: 100,
        description: 'Chicken breast raw',
        food_category: 'Poultry Products',
        data_type: 'foundation_food',
        publication_date: null,
      };
      mockExecute.mockReturnValue({ rows: [mockFood] });

      const result = await service.getFoodById(100);

      expect(result).not.toBeNull();
      expect(result!.fdcId).toBe(100);
      expect(result!.description).toBe('Chicken breast raw');
    });

    it('returns null for non-existent food', async () => {
      mockExecute.mockReturnValue({ rows: [] });

      const result = await service.getFoodById(999999);
      expect(result).toBeNull();
    });
  });

  describe('getFoodNutrients', () => {
    it('returns nutrient breakdown for a given fdc_id', async () => {
      const mockNutrients = [
        { nutrient_id: 1008, name: 'Energy', amount: 165.0, unit: 'kcal' },
        { nutrient_id: 1003, name: 'Protein', amount: 31.0, unit: 'g' },
        { nutrient_id: 1005, name: 'Carbohydrate, by difference', amount: 0.0, unit: 'g' },
        { nutrient_id: 1004, name: 'Total lipid (fat)', amount: 3.6, unit: 'g' },
      ];
      mockExecute.mockReturnValue({ rows: mockNutrients });

      const results = await service.getFoodNutrients(100);

      expect(results).toHaveLength(4);
      expect(results[0].name).toBe('Energy');
      expect(results[0].amount).toBe(165.0);
      expect(results[0].unit).toBe('kcal');
    });
  });

  describe('getFoodPortions', () => {
    it('returns serving size options for a food', async () => {
      const mockPortions = [
        {
          id: 1,
          food_id: 100,
          portion_description: '1 breast, bone and skin removed',
          gram_weight: 172.0,
          modifier: 'breast',
        },
      ];
      mockExecute.mockReturnValue({ rows: mockPortions });

      const results = await service.getFoodPortions(100);

      expect(results).toHaveLength(1);
      expect(results[0].portionDescription).toBe('1 breast, bone and skin removed');
      expect(results[0].gramWeight).toBe(172.0);
    });
  });

  describe('getMacros', () => {
    it('calculates macros for a given weight based on per-100g values', async () => {
      // Mock getNutrients to return per-100g values
      const mockNutrients = [
        { nutrient_id: 1008, name: 'Energy', amount: 165.0, unit: 'kcal' },
        { nutrient_id: 1003, name: 'Protein', amount: 31.0, unit: 'g' },
        { nutrient_id: 1005, name: 'Carbohydrate, by difference', amount: 0.0, unit: 'g' },
        { nutrient_id: 1004, name: 'Total lipid (fat)', amount: 3.6, unit: 'g' },
      ];
      mockExecute.mockReturnValue({ rows: mockNutrients });

      const result = await service.getMacros(100, 200); // 200g of chicken

      expect(result.weightGrams).toBe(200);
      // 165 kcal per 100g * 2 = 330
      expect(result.calories).toBeCloseTo(330, 1);
      // 31g protein per 100g * 2 = 62
      expect(result.protein).toBeCloseTo(62, 1);
      // 0g carbs per 100g * 2 = 0
      expect(result.carbs).toBeCloseTo(0, 1);
      // 3.6g fat per 100g * 2 = 7.2
      expect(result.fat).toBeCloseTo(7.2, 1);
    });
  });
});
