/**
 * PortionBridge test suite -- validates TypeScript port matches Python PortionEstimator
 * reference outputs within 10%.
 */
import {
  PortionEstimator,
  estimatePortion,
  FOOD_DENSITY_TABLE,
  REFERENCE_OBJECTS,
  STANDARD_SERVINGS,
  FLAT_FOODS,
  DEFAULT_DEPTH_RATIO,
  BBOX_FILL_FACTOR,
} from '../portionBridge';

// ---------- Constants ----------

describe('portionBridge constants', () => {
  it('FOOD_DENSITY_TABLE has all 55 entries matching Python version', () => {
    // Python version has exactly 55 entries (counted from source)
    expect(Object.keys(FOOD_DENSITY_TABLE).length).toBe(55);
  });

  it('REFERENCE_OBJECTS has all 15 entries matching Python version', () => {
    expect(Object.keys(REFERENCE_OBJECTS).length).toBe(15);
  });

  it('STANDARD_SERVINGS has 60+ entries matching Python version', () => {
    expect(Object.keys(STANDARD_SERVINGS).length).toBeGreaterThanOrEqual(60);
  });

  it('DEFAULT_DEPTH_RATIO is 0.25', () => {
    expect(DEFAULT_DEPTH_RATIO).toBe(0.25);
  });

  it('BBOX_FILL_FACTOR is 0.55', () => {
    expect(BBOX_FILL_FACTOR).toBe(0.55);
  });

  it('FLAT_FOODS contains expected entries', () => {
    expect(FLAT_FOODS.has('pizza')).toBe(true);
    expect(FLAT_FOODS.has('pancake')).toBe(true);
    expect(FLAT_FOODS.has('waffle')).toBe(true);
    expect(FLAT_FOODS.has('tortilla')).toBe(true);
    expect(FLAT_FOODS.has('bread')).toBe(true);
    expect(FLAT_FOODS.has('cookie')).toBe(true);
    expect(FLAT_FOODS.has('cracker')).toBe(true);
  });

  // Spot-check specific density values from Python source
  it('density table has correct values for key entries', () => {
    expect(FOOD_DENSITY_TABLE['rice']).toBe(0.90);
    expect(FOOD_DENSITY_TABLE['fried rice']).toBe(0.85);
    expect(FOOD_DENSITY_TABLE['chicken']).toBe(1.04);
    expect(FOOD_DENSITY_TABLE['salad']).toBe(0.25);
    expect(FOOD_DENSITY_TABLE['pizza']).toBe(0.65);
    expect(FOOD_DENSITY_TABLE['sushi']).toBe(0.95);
    expect(FOOD_DENSITY_TABLE['popcorn']).toBe(0.10);
    expect(FOOD_DENSITY_TABLE['default']).toBe(0.70);
  });

  // Spot-check reference objects
  it('reference objects have correct dimensions', () => {
    expect(REFERENCE_OBJECTS['plate']).toEqual({
      type: 'circle',
      diameterCm: 26.0,
    });
    expect(REFERENCE_OBJECTS['credit_card']).toEqual({
      type: 'rectangle',
      widthCm: 8.56,
      heightCm: 5.4,
    });
    expect(REFERENCE_OBJECTS['bowl']).toEqual({
      type: 'circle',
      diameterCm: 16.0,
    });
  });

  // Spot-check standard servings
  it('standard servings have correct values for key entries', () => {
    expect(STANDARD_SERVINGS['fried rice']).toBe(250.0);
    expect(STANDARD_SERVINGS['sushi']).toBe(250.0);
    expect(STANDARD_SERVINGS['ramen']).toBe(400.0);
    expect(STANDARD_SERVINGS['burger']).toBe(250.0);
    expect(STANDARD_SERVINGS['pizza']).toBe(200.0);
  });
});

// ---------- PortionEstimator ----------

describe('PortionEstimator', () => {
  let estimator: PortionEstimator;

  beforeEach(() => {
    estimator = new PortionEstimator();
  });

  // --- Level 3: USDA Default fallback ---

  describe('USDA default fallback', () => {
    it('fried rice returns 250g', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'fried rice',
      );
      expect(result.weightG).toBe(250.0);
      expect(result.confidence).toBe('low');
      expect(result.method).toBe('usda_default');
      expect(result.suggestReference).toBe(true);
    });

    it('sushi returns 250g', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'sushi',
      );
      expect(result.weightG).toBe(250.0);
      expect(result.confidence).toBe('low');
      expect(result.method).toBe('usda_default');
    });

    it('unknown-dish returns 250g generic default', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'unknown-dish',
      );
      expect(result.weightG).toBe(250.0);
      expect(result.confidence).toBe('low');
      expect(result.method).toBe('usda_default');
      expect(result.suggestReference).toBe(true);
    });

    it('ramen returns 400g', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'ramen',
      );
      expect(result.weightG).toBe(400.0);
    });

    it('burger returns 250g', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'burger',
      );
      expect(result.weightG).toBe(250.0);
    });
  });

  // --- Level 1: Geometry estimation ---

  describe('geometry estimation with plate reference', () => {
    it('produces weight within 10% of Python reference (244.6g)', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'fried rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.confidence).toBe('high');
      expect(result.method).toBe('geometry');
      expect(result.suggestReference).toBe(false);
      // Python gives 244.6g -- within 10% = 220.1 to 269.1
      expect(result.weightG).toBeGreaterThan(220);
      expect(result.weightG).toBeLessThan(270);
    });

    it('returns null/fallback for zero-size bbox', () => {
      // Zero-size food bbox -> geometry returns null -> falls through to USDA default
      const result = estimator.estimate(
        { x: 200, y: 200, w: 0, h: 0 },
        { width: 640, height: 640 },
        'rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      // Should fall through to USDA default since geometry can't compute from zero bbox
      expect(result.method).toBe('usda_default');
    });
  });

  // --- Level 2: User history estimation ---

  describe('user history estimation', () => {
    it('weighted average of past servings (~301.3g)', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'fried rice',
        [], // no reference objects
        [
          { dish: 'fried rice', weightG: 280 },
          { dish: 'fried rice', weightG: 320 },
          { dish: 'fried rice', weightG: 300 },
        ],
      );
      expect(result.confidence).toBe('medium');
      expect(result.method).toBe('user_history');
      expect(result.suggestReference).toBe(false);
      // Python gives 301.3g -- within 10% = 271.2 to 331.4
      expect(result.weightG).toBeGreaterThan(271);
      expect(result.weightG).toBeLessThan(332);
    });

    it('single history entry returns that weight', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'sushi',
        [],
        [{ dish: 'sushi', weightG: 350 }],
      );
      expect(result.confidence).toBe('medium');
      expect(result.method).toBe('user_history');
      expect(result.weightG).toBeCloseTo(350, 0);
    });

    it('ignores non-matching history entries', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'sushi',
        [],
        [
          { dish: 'pizza', weightG: 200 },
          { dish: 'burger', weightG: 300 },
        ],
      );
      // No matching history -> falls through to USDA default
      expect(result.method).toBe('usda_default');
    });
  });

  // --- Flat food detection ---

  describe('flat food detection', () => {
    it('pizza gets reduced depth (within 10% of 76.2g)', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'pizza',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.confidence).toBe('high');
      expect(result.method).toBe('geometry');
      // Python gives 76.2g -- within 10% = 68.6 to 83.8
      expect(result.weightG).toBeGreaterThan(68);
      expect(result.weightG).toBeLessThan(84);
      // Should be marked as flat food
      expect(result.details.isFlatFood).toBe(true);
    });
  });

  // --- Density lookup ---

  describe('density lookup', () => {
    it('exact match: "rice" returns 0.90', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.details.densityGPerCm3).toBe(0.90);
    });

    it('exact match: "fried rice" returns 0.85', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'fried rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.details.densityGPerCm3).toBe(0.85);
    });

    it('unknown dish falls back to default density (0.70)', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'alien food',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.details.densityGPerCm3).toBe(0.70);
    });
  });

  // --- Reference object scaling ---

  describe('reference object scaling', () => {
    it('plate (26cm diameter) provides correct cm-per-pixel conversion', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      // Python: scale = 26.0 / 540 = 0.048148
      expect(result.details.scaleCmPerPx).toBeCloseTo(0.048148, 4);
    });

    it('credit card provides correct scaling', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 200, h: 200 },
        { width: 640, height: 640 },
        'rice',
        [{ type: 'credit_card', bbox: { x: 400, y: 400, w: 100, h: 60 } }],
      );
      expect(result.confidence).toBe('high');
      expect(result.method).toBe('geometry');
      // scale = ((8.56/100) + (5.4/60)) / 2 = (0.0856 + 0.09) / 2 = 0.0878
      expect(result.details.scaleCmPerPx).toBeCloseTo(0.0878, 3);
    });
  });

  // --- Weight clamping ---

  describe('weight clamping', () => {
    it('weight is clamped to 5-5000g range', () => {
      // Very small food with plate = should clamp to >= 5g
      const result = estimator.estimate(
        { x: 300, y: 300, w: 5, h: 5 },
        { width: 640, height: 640 },
        'rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.weightG).toBeGreaterThanOrEqual(5);
      expect(result.weightG).toBeLessThanOrEqual(5000);
    });
  });

  // --- suggestReference ---

  describe('suggestReference flag', () => {
    it('is true when confidence is low', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'fried rice',
      );
      expect(result.confidence).toBe('low');
      expect(result.suggestReference).toBe(true);
    });

    it('is false when confidence is high (geometry)', () => {
      const result = estimator.estimate(
        { x: 150, y: 150, w: 300, h: 250 },
        { width: 640, height: 640 },
        'fried rice',
        [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      );
      expect(result.confidence).toBe('high');
      expect(result.suggestReference).toBe(false);
    });

    it('is false when confidence is medium (history)', () => {
      const result = estimator.estimate(
        { x: 100, y: 100, w: 300, h: 300 },
        { width: 640, height: 640 },
        'fried rice',
        [],
        [{ dish: 'fried rice', weightG: 300 }],
      );
      expect(result.confidence).toBe('medium');
      expect(result.suggestReference).toBe(false);
    });
  });
});

// ---------- Convenience function ----------

describe('estimatePortion convenience function', () => {
  it('delegates to PortionEstimator', () => {
    const result = estimatePortion(
      { x: 100, y: 100, w: 300, h: 300 },
      { width: 640, height: 640 },
      'fried rice',
    );
    expect(result.weightG).toBe(250.0);
    expect(result.confidence).toBe('low');
    expect(result.method).toBe('usda_default');
  });
});

// ---------- Three-tier fallback priority ----------

describe('three-tier fallback priority', () => {
  it('geometry takes priority over history and USDA default', () => {
    const result = estimatePortion(
      { x: 150, y: 150, w: 300, h: 250 },
      { width: 640, height: 640 },
      'fried rice',
      [{ type: 'plate', bbox: { x: 50, y: 50, w: 540, h: 540 } }],
      [{ dish: 'fried rice', weightG: 500 }],
    );
    expect(result.method).toBe('geometry');
  });

  it('history takes priority over USDA default when no reference objects', () => {
    const result = estimatePortion(
      { x: 100, y: 100, w: 300, h: 300 },
      { width: 640, height: 640 },
      'fried rice',
      [],
      [{ dish: 'fried rice', weightG: 300 }],
    );
    expect(result.method).toBe('user_history');
  });

  it('falls through to USDA default when no reference objects or history', () => {
    const result = estimatePortion(
      { x: 100, y: 100, w: 300, h: 300 },
      { width: 640, height: 640 },
      'fried rice',
    );
    expect(result.method).toBe('usda_default');
  });
});
