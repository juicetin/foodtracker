/**
 * PortionBridge -- TypeScript port of training/portion_estimator.py
 *
 * Estimates food weight (grams) from visual cues with a three-tier fallback:
 *   Level 1: Reference object geometry (HIGH confidence)
 *   Level 2: User history extrapolation (MEDIUM confidence)
 *   Level 3: USDA standard serving size (LOW confidence)
 *
 * Per locked decision: Target +/-10% when reference objects present.
 * When confidence is LOW, suggests reference objects for next time.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ImageSize {
  width: number;
  height: number;
}

export interface ReferenceObject {
  type: string;
  bbox: BoundingBox;
}

export interface UserHistoryEntry {
  dish: string;
  weightG: number;
  timestamp?: string;
}

export interface PortionEstimate {
  weightG: number;
  confidence: 'high' | 'medium' | 'low';
  method: 'geometry' | 'user_history' | 'usda_default';
  suggestReference: boolean;
  details: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Food density table (g/cm^3) -- faithful port of Python FOOD_DENSITY_TABLE
// ---------------------------------------------------------------------------

export const FOOD_DENSITY_TABLE: Record<string, number> = {
  // Grains & starches
  'rice': 0.90,
  'fried rice': 0.85,
  'noodle': 0.80,
  'pasta': 0.85,
  'bread': 0.35,
  'cereal': 0.30,
  'oatmeal': 0.95,
  'pancake': 0.70,
  'waffle': 0.55,
  'pizza': 0.65,
  'tortilla': 0.70,
  'couscous': 0.85,
  'quinoa': 0.88,

  // Proteins
  'meat': 1.05,
  'chicken': 1.04,
  'beef': 1.08,
  'pork': 1.06,
  'fish': 1.02,
  'shrimp': 1.00,
  'egg': 1.03,
  'tofu': 0.95,
  'tempeh': 0.98,

  // Vegetables
  'vegetable': 0.60,
  'salad': 0.25,
  'broccoli': 0.55,
  'carrot': 0.65,
  'potato': 0.75,
  'sweet potato': 0.72,
  'corn': 0.70,
  'beans': 0.80,
  'peas': 0.75,
  'mushroom': 0.45,
  'spinach': 0.30,
  'cabbage': 0.35,
  'lettuce': 0.20,
  'tomato': 0.65,
  'onion': 0.60,
  'pepper': 0.55,
  'eggplant': 0.50,
  'zucchini': 0.55,

  // Fruits
  'fruit': 0.70,
  'apple': 0.80,
  'banana': 0.75,
  'orange': 0.85,
  'berry': 0.60,
  'mango': 0.78,
  'watermelon': 0.60,
  'grape': 0.85,

  // Dairy
  'cheese': 0.95,
  'yogurt': 1.05,
  'milk': 1.03,
  'ice cream': 0.65,
  'cream': 0.98,
  'butter': 0.91,

  // Soups & liquids
  'soup': 1.00,
  'broth': 1.00,
  'curry': 0.95,
  'stew': 0.98,
  'sauce': 1.05,
  'gravy': 1.02,

  // Mixed dishes
  'stir fry': 0.75,
  'casserole': 0.85,
  'burrito': 0.80,
  'sandwich': 0.55,
  'burger': 0.70,
  'wrap': 0.65,
  'sushi': 0.95,
  'dumpling': 0.90,
  'spring roll': 0.70,
  'pie': 0.75,

  // Snacks & desserts
  'cookie': 0.60,
  'cake': 0.50,
  'brownie': 0.65,
  'chips': 0.25,
  'nuts': 0.65,
  'popcorn': 0.10,
  'chocolate': 0.95,
  'candy': 0.80,

  // Default categories
  'liquid': 1.00,
  'solid': 0.75,
  'default': 0.70,
};

// ---------------------------------------------------------------------------
// Reference object dimensions (cm) -- faithful port of Python REFERENCE_OBJECTS
// ---------------------------------------------------------------------------

interface CircleRef {
  type: 'circle';
  diameterCm: number;
}

interface RectangleRef {
  type: 'rectangle';
  widthCm: number;
  heightCm: number;
}

type RefObjectInfo = CircleRef | RectangleRef;

export const REFERENCE_OBJECTS: Record<string, RefObjectInfo> = {
  'plate_dinner': { type: 'circle', diameterCm: 26.0 },
  'plate_side': { type: 'circle', diameterCm: 20.0 },
  'plate': { type: 'circle', diameterCm: 26.0 },
  'bowl': { type: 'circle', diameterCm: 16.0 },
  'credit_card': { type: 'rectangle', widthCm: 8.56, heightCm: 5.4 },
  'coin_quarter': { type: 'circle', diameterCm: 2.426 },
  'coin_dollar': { type: 'circle', diameterCm: 2.67 },
  'coin_50c_aud': { type: 'circle', diameterCm: 3.17 },
  'hand': { type: 'rectangle', widthCm: 8.5, heightCm: 19.0 },
  'fork': { type: 'rectangle', widthCm: 2.5, heightCm: 19.0 },
  'knife': { type: 'rectangle', widthCm: 2.0, heightCm: 23.0 },
  'spoon': { type: 'rectangle', widthCm: 4.0, heightCm: 18.0 },
  'chopstick': { type: 'rectangle', widthCm: 0.8, heightCm: 24.0 },
  'can_330ml': { type: 'circle', diameterCm: 6.6 },
  'phone': { type: 'rectangle', widthCm: 7.15, heightCm: 14.67 },
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default assumed food depth as fraction of bounding box shorter side */
export const DEFAULT_DEPTH_RATIO = 0.25;

/** Packing factor: how much of the bounding box is actually filled with food */
export const BBOX_FILL_FACTOR = 0.55;

/** Foods that are flat (reduced depth calculation) */
export const FLAT_FOODS = new Set([
  'pizza',
  'pancake',
  'waffle',
  'tortilla',
  'bread',
  'cookie',
  'cracker',
]);

// ---------------------------------------------------------------------------
// Standard serving sizes -- faithful port of Python _builtin_serving_size
// ---------------------------------------------------------------------------

export const STANDARD_SERVINGS: Record<string, number> = {
  // Rice dishes
  'rice': 200.0,
  'fried rice': 250.0,
  'risotto': 250.0,
  'biryani': 300.0,

  // Noodle dishes
  'noodle': 250.0,
  'pasta': 220.0,
  'ramen': 400.0,
  'pho': 450.0,
  'udon': 350.0,
  'pad thai': 300.0,
  'chow mein': 280.0,

  // Protein dishes
  'steak': 200.0,
  'chicken': 150.0,
  'fish': 150.0,
  'pork': 150.0,
  'tofu': 150.0,
  'shrimp': 120.0,

  // Soups
  'soup': 350.0,
  'stew': 350.0,
  'curry': 300.0,
  'chili': 300.0,

  // Sandwiches & wraps
  'sandwich': 200.0,
  'burger': 250.0,
  'burrito': 300.0,
  'wrap': 250.0,
  'banh mi': 300.0,

  // Salads & vegetables
  'salad': 150.0,
  'coleslaw': 100.0,

  // Pizza & flatbreads
  'pizza': 200.0,

  // Asian dishes
  'sushi': 250.0,
  'dim sum': 200.0,
  'dumpling': 180.0,
  'spring roll': 120.0,
  'bibimbap': 400.0,
  'kimchi': 80.0,
  'tempura': 150.0,
  'tonkatsu': 200.0,
  'gyoza': 150.0,
  'bulgogi': 200.0,
  'tteokbokki': 250.0,

  // Breakfast
  'omelette': 180.0,
  'pancake': 150.0,
  'waffle': 120.0,
  'cereal': 60.0,
  'oatmeal': 250.0,

  // Desserts
  'cake': 120.0,
  'ice cream': 150.0,
  'cookie': 40.0,
  'brownie': 60.0,
  'mochi': 80.0,

  // Drinks
  'smoothie': 350.0,
  'juice': 250.0,
};

/** Category-based fallback for partial matching */
const CATEGORY_DEFAULTS: Record<string, number> = {
  'soup': 350.0,
  'curry': 300.0,
  'rice': 250.0,
  'noodle': 250.0,
  'meat': 150.0,
  'fish': 150.0,
  'salad': 150.0,
  'dessert': 120.0,
};

/** Generic fallback weight */
const GENERIC_DEFAULT_G = 250.0;

// ---------------------------------------------------------------------------
// PortionEstimator class
// ---------------------------------------------------------------------------

/**
 * Estimates food portion weight from visual cues using a three-tier fallback chain.
 *
 * Fallback chain:
 *   1. Reference object geometry (HIGH confidence)
 *   2. User history extrapolation (MEDIUM confidence)
 *   3. USDA standard serving / built-in lookup (LOW confidence)
 */
export class PortionEstimator {
  /**
   * Estimate portion weight using the smart fallback chain.
   */
  estimate(
    bbox: BoundingBox,
    imageSize: ImageSize,
    dishName: string,
    referenceObjects: ReferenceObject[] = [],
    userHistory: UserHistoryEntry[] = [],
  ): PortionEstimate {
    // Level 1: Reference object geometry
    if (referenceObjects.length > 0) {
      const result = this._estimateFromGeometry(
        bbox,
        imageSize,
        dishName,
        referenceObjects,
      );
      if (result !== null) {
        return result;
      }
    }

    // Level 2: User history extrapolation
    if (userHistory.length > 0) {
      const result = this._estimateFromHistory(dishName, userHistory);
      if (result !== null) {
        return result;
      }
    }

    // Level 3: USDA standard serving
    return this._estimateFromUsdaDefault(dishName);
  }

  /**
   * Level 1: Estimate portion using reference object for pixel-to-cm conversion.
   */
  private _estimateFromGeometry(
    bbox: BoundingBox,
    imageSize: ImageSize,
    dishName: string,
    referenceObjects: ReferenceObject[],
  ): PortionEstimate | null {
    const foodPxW = bbox.w;
    const foodPxH = bbox.h;

    if (foodPxW <= 0 || foodPxH <= 0) {
      return null;
    }

    // Find best reference object
    let bestRef: ReferenceObject | null = null;
    let bestScale: number | null = null;
    let bestArea = 0;

    for (const refObj of referenceObjects) {
      const refType = refObj.type.toLowerCase().replace(/ /g, '_');
      const refInfo = REFERENCE_OBJECTS[refType];

      if (!refInfo) {
        continue;
      }

      const refPxW = refObj.bbox.w;
      const refPxH = refObj.bbox.h;

      if (refPxW <= 0 || refPxH <= 0) {
        continue;
      }

      // Calculate cm/pixel from reference object
      let scale: number;
      if (refInfo.type === 'circle') {
        const refPxDiameter = Math.max(refPxW, refPxH);
        scale = refInfo.diameterCm / refPxDiameter;
      } else {
        const scaleW = refInfo.widthCm / refPxW;
        const scaleH = refInfo.heightCm / refPxH;
        scale = (scaleW + scaleH) / 2;
      }

      const area = refPxW * refPxH;
      if (bestScale === null || area > bestArea) {
        bestRef = refObj;
        bestScale = scale;
        bestArea = area;
      }
    }

    if (bestScale === null || bestRef === null) {
      return null;
    }

    // Convert food bounding box to real dimensions
    const foodCmW = foodPxW * bestScale;
    const foodCmH = foodPxH * bestScale;

    // Estimate depth
    const shorterSide = Math.min(foodCmW, foodCmH);

    // Check if flat food
    const dishLower = dishName.toLowerCase().replace(/-/g, ' ');
    const isFlat = Array.from(FLAT_FOODS).some((f) => dishLower.includes(f));

    let foodCmDepth: number;
    let volumeCm3: number;

    if (isFlat) {
      // Flat foods: reduced depth (0.08x, clamped 0.5-2.0cm)
      foodCmDepth = Math.max(0.5, Math.min(shorterSide * 0.08, 2.0));
      // Flat foods have a higher fill factor (0.70)
      volumeCm3 = foodCmW * foodCmH * foodCmDepth * 0.70;
    } else {
      // Normal depth (0.25x, clamped 1.0-5.0cm)
      foodCmDepth = shorterSide * DEFAULT_DEPTH_RATIO;
      foodCmDepth = Math.max(1.0, Math.min(foodCmDepth, 5.0));
      // Standard fill factor (0.55)
      volumeCm3 = foodCmW * foodCmH * foodCmDepth * BBOX_FILL_FACTOR;
    }

    // Look up food density
    const density = this._getFoodDensity(dishName);

    // Weight = volume * density
    let weightG = volumeCm3 * density;

    // Clamp to reasonable range
    weightG = Math.max(5.0, Math.min(weightG, 5000.0));

    return {
      weightG: Math.round(weightG * 10) / 10,
      confidence: 'high',
      method: 'geometry',
      suggestReference: false,
      details: {
        foodCmW: Math.round(foodCmW * 100) / 100,
        foodCmH: Math.round(foodCmH * 100) / 100,
        foodCmDepth: Math.round(foodCmDepth * 100) / 100,
        volumeCm3: Math.round(volumeCm3 * 100) / 100,
        densityGPerCm3: density,
        referenceType: bestRef.type,
        scaleCmPerPx: Math.round(bestScale * 1000000) / 1000000,
        isFlatFood: isFlat,
      },
    };
  }

  /**
   * Level 2: Estimate from user's previous portion sizes, weighted by recency.
   */
  private _estimateFromHistory(
    dishName: string,
    userHistory: UserHistoryEntry[],
  ): PortionEstimate | null {
    const dishLower = dishName
      .toLowerCase()
      .replace(/-/g, ' ')
      .replace(/_/g, ' ');

    // Exact match
    let matching = userHistory.filter(
      (h) =>
        h.dish
          .toLowerCase()
          .replace(/-/g, ' ')
          .replace(/_/g, ' ') === dishLower,
    );

    // Fuzzy match: name contains or is contained
    if (matching.length === 0) {
      matching = userHistory.filter((h) => {
        const hDish = h.dish
          .toLowerCase()
          .replace(/-/g, ' ')
          .replace(/_/g, ' ');
        return dishLower.includes(hDish) || hDish.includes(dishLower);
      });
    }

    if (matching.length === 0) {
      return null;
    }

    // Weight by recency (more recent = higher weight)
    const weights: number[] = [];
    for (let i = 0; i < matching.length; i++) {
      const recencyWeight =
        1.0 + 0.5 * (i / Math.max(matching.length - 1, 1));
      weights.push(recencyWeight);
    }

    // Weighted average
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    const weightedSum = matching.reduce(
      (sum, entry, i) => sum + entry.weightG * weights[i],
      0,
    );
    const avgWeight = weightedSum / totalWeight;

    // Calculate coefficient of variation
    const values = matching.map((e) => e.weightG);
    let cv: number;
    if (values.length > 1) {
      const stdDev = Math.sqrt(
        values.reduce((sum, v) => sum + (v - avgWeight) ** 2, 0) /
          values.length,
      );
      cv = avgWeight > 0 ? stdDev / avgWeight : 1.0;
    } else {
      cv = 0.3;
    }

    return {
      weightG: Math.round(avgWeight * 10) / 10,
      confidence: 'medium',
      method: 'user_history',
      suggestReference: false,
      details: {
        matchingEntries: matching.length,
        valuesG: matching.map((e) => Math.round(e.weightG * 10) / 10),
        coefficientOfVariation: Math.round(cv * 1000) / 1000,
        weightedAvgG: Math.round(avgWeight * 10) / 10,
      },
    };
  }

  /**
   * Level 3: Fall back to USDA standard serving size.
   */
  private _estimateFromUsdaDefault(dishName: string): PortionEstimate {
    const { weightG, source } = this._builtinServingSize(dishName);

    return {
      weightG: Math.round(weightG * 10) / 10,
      confidence: 'low',
      method: 'usda_default',
      suggestReference: true,
      details: {
        source,
        dishQueried: dishName,
        note: 'Based on standard serving size. For more accuracy, include a reference object (coin, credit card) next to your food.',
      },
    };
  }

  /**
   * Built-in standard serving sizes.
   */
  private _builtinServingSize(
    dishName: string,
  ): { weightG: number; source: string } {
    const dishLower = dishName
      .toLowerCase()
      .replace(/-/g, ' ')
      .replace(/_/g, ' ');

    // Exact match
    if (dishLower in STANDARD_SERVINGS) {
      return {
        weightG: STANDARD_SERVINGS[dishLower],
        source: `builtin_standard (${dishLower})`,
      };
    }

    // Partial match
    for (const [key, weight] of Object.entries(STANDARD_SERVINGS)) {
      if (key.includes(dishLower) || dishLower.includes(key)) {
        return {
          weightG: weight,
          source: `builtin_standard (partial: ${key})`,
        };
      }
    }

    // Category-based fallback
    for (const [cat, weight] of Object.entries(CATEGORY_DEFAULTS)) {
      if (dishLower.includes(cat)) {
        return {
          weightG: weight,
          source: `builtin_category (${cat})`,
        };
      }
    }

    // Ultimate fallback: generic portion
    return {
      weightG: GENERIC_DEFAULT_G,
      source: 'builtin_generic (250g default)',
    };
  }

  /**
   * Look up food density from the density table.
   * Returns density in g/cm^3.
   */
  private _getFoodDensity(dishName: string): number {
    const dishLower = dishName
      .toLowerCase()
      .replace(/-/g, ' ')
      .replace(/_/g, ' ');

    // Exact match
    if (dishLower in FOOD_DENSITY_TABLE) {
      return FOOD_DENSITY_TABLE[dishLower];
    }

    // Partial match
    for (const [key, density] of Object.entries(FOOD_DENSITY_TABLE)) {
      if (key.includes(dishLower) || dishLower.includes(key)) {
        return density;
      }
    }

    // Default
    return FOOD_DENSITY_TABLE['default'];
  }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/**
 * Convenience function for quick portion estimation.
 * See PortionEstimator.estimate() for full documentation.
 */
export function estimatePortion(
  bbox: BoundingBox,
  imageSize: ImageSize,
  dishName: string,
  referenceObjects?: ReferenceObject[],
  userHistory?: UserHistoryEntry[],
): PortionEstimate {
  const estimator = new PortionEstimator();
  return estimator.estimate(
    bbox,
    imageSize,
    dishName,
    referenceObjects,
    userHistory,
  );
}
