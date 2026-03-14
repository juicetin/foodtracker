import {
  VLM_TIER_CONFIG,
  FOOD_IDENTIFICATION_SCHEMA,
  type VlmTier,
  type VlmTierConfig,
  type VlmFoodResult,
  type VlmDish,
} from '../vlmTypes';

describe('vlmTypes', () => {
  describe('VLM_TIER_CONFIG', () => {
    it('has budget, mid, and high tiers', () => {
      expect(Object.keys(VLM_TIER_CONFIG)).toEqual(
        expect.arrayContaining(['budget', 'mid', 'high'])
      );
      expect(Object.keys(VLM_TIER_CONFIG)).toHaveLength(3);
    });

    it.each(['budget', 'mid', 'high'] as const)(
      '%s tier has all required fields',
      (tier) => {
        const config = VLM_TIER_CONFIG[tier];
        expect(config.modelId).toEqual(expect.any(String));
        expect(config.modelFile).toMatch(/\.gguf$/);
        expect(config.mmprojFile).toMatch(/\.gguf$/);
        expect(config.modelSize).toBeGreaterThan(0);
        expect(config.mmprojSize).toBeGreaterThan(0);
        expect(config.totalDownload).toBeGreaterThan(0);
        expect(config.runtimeRam).toBeGreaterThan(0);
      }
    );

    it('budget tier has SmolVLM-256M model', () => {
      expect(VLM_TIER_CONFIG.budget.modelFile).toBe(
        'SmolVLM-256M-Instruct-Q8_0.gguf'
      );
      expect(VLM_TIER_CONFIG.budget.modelSize).toBe(175_000_000);
    });

    it('mid tier has SmolVLM-500M model', () => {
      expect(VLM_TIER_CONFIG.mid.modelFile).toBe(
        'SmolVLM-500M-Instruct-Q8_0.gguf'
      );
      expect(VLM_TIER_CONFIG.mid.modelSize).toBe(437_000_000);
    });

    it('high tier has SmolVLM2-2.2B model', () => {
      expect(VLM_TIER_CONFIG.high.modelFile).toBe(
        'SmolVLM2-2.2B-Instruct-Q4_K_M.gguf'
      );
      expect(VLM_TIER_CONFIG.high.modelSize).toBe(1_110_000_000);
    });
  });

  describe('FOOD_IDENTIFICATION_SCHEMA', () => {
    it('has name and strict fields', () => {
      expect(FOOD_IDENTIFICATION_SCHEMA.name).toBe('food_identification');
      expect(FOOD_IDENTIFICATION_SCHEMA.strict).toBe(true);
    });

    it('schema requires dishes array', () => {
      expect(FOOD_IDENTIFICATION_SCHEMA.schema.type).toBe('object');
      expect(FOOD_IDENTIFICATION_SCHEMA.schema.required).toContain('dishes');
      expect(FOOD_IDENTIFICATION_SCHEMA.schema.properties.dishes.type).toBe(
        'array'
      );
    });

    it('dish items require name, cuisine, ingredients', () => {
      const dishSchema =
        FOOD_IDENTIFICATION_SCHEMA.schema.properties.dishes.items;
      expect(dishSchema.required).toEqual(
        expect.arrayContaining(['name', 'cuisine', 'ingredients'])
      );
      expect(dishSchema.properties.name.type).toBe('string');
      expect(dishSchema.properties.cuisine.type).toBe('string');
      expect(dishSchema.properties.ingredients.type).toBe('array');
    });

    it('portion_hint is optional (not in required)', () => {
      const dishSchema =
        FOOD_IDENTIFICATION_SCHEMA.schema.properties.dishes.items;
      expect(dishSchema.properties.portion_hint).toBeDefined();
      expect(dishSchema.required).not.toContain('portion_hint');
    });
  });

  describe('Type contracts', () => {
    it('VlmTier includes none', () => {
      // Type-level test: ensure 'none' is a valid VlmTier
      const tier: VlmTier = 'none';
      expect(['budget', 'mid', 'high', 'none']).toContain(tier);
    });

    it('VlmFoodResult has dishes array', () => {
      const result: VlmFoodResult = {
        dishes: [
          {
            name: 'pad thai',
            cuisine: 'Thai',
            ingredients: ['noodles', 'shrimp', 'peanuts'],
            portion_hint: 'large plate',
          },
        ],
      };
      expect(result.dishes).toHaveLength(1);
      expect(result.dishes[0].name).toBe('pad thai');
    });

    it('VlmDish portion_hint is optional', () => {
      const dish: VlmDish = {
        name: 'sushi',
        cuisine: 'Japanese',
        ingredients: ['rice', 'fish'],
      };
      expect(dish.portion_hint).toBeUndefined();
    });
  });
});
