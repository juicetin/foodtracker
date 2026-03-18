/**
 * Mock scan result for devices where Gemini Nano is unavailable
 * (Pixel 7 and older, needs_update, etc.).
 *
 * Returns a realistic chicken rice bowl so all downstream flows
 * (KG lookup, dish cards, editing) work correctly on any device.
 */

import type { VlmFoodResult } from './vlmTypes';

export function getMockScanResult(): VlmFoodResult {
  return {
    dishes: [
      {
        name: 'Grilled Chicken Rice Bowl',
        cuisine: 'Asian',
        ingredients: [
          { name: 'steamed jasmine rice', amount_g: 180 },
          { name: 'grilled chicken breast', amount_g: 150 },
          { name: 'steamed broccoli', amount_g: 80 },
          { name: 'soy sauce', amount_g: 15 },
          { name: 'sesame oil', amount_g: 5 },
        ],
      },
    ],
  };
}
