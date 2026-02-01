// Food Composition Database Configuration

export interface FoodDatabase {
  id: string;
  name: string;
  region: string;
  apiEndpoint?: string;
  priority: number;
}

export const FOOD_DATABASES: FoodDatabase[] = [
  {
    id: 'AFCD',
    name: 'Australian Food Composition Database',
    region: 'AU',
    priority: 1,
  },
  {
    id: 'NUTTAB',
    name: 'NUTTAB (Australian)',
    region: 'AU',
    priority: 2,
  },
  {
    id: 'USDA',
    name: 'USDA FoodData Central',
    region: 'US',
    apiEndpoint: 'https://api.nal.usda.gov/fdc/v1',
    priority: 1,
  },
  {
    id: 'CNF',
    name: 'Canadian Nutrient File',
    region: 'CA',
    priority: 1,
  },
  {
    id: 'CoFID',
    name: 'Composition of Foods Integrated Dataset (UK)',
    region: 'UK',
    priority: 1,
  },
  {
    id: 'CIQUAL',
    name: 'CIQUAL (France)',
    region: 'FR',
    priority: 1,
  },
  {
    id: 'OpenFoodFacts',
    name: 'Open Food Facts',
    region: 'global',
    apiEndpoint: 'https://world.openfoodfacts.org/api/v0',
    priority: 10, // Lowest priority - fallback
  },
];

export function getDatabaseForRegion(region: string): FoodDatabase[] {
  return FOOD_DATABASES
    .filter(db => db.region === region || db.region === 'global')
    .sort((a, b) => a.priority - b.priority);
}

// Food density lookup table (g/cm³)
export const FOOD_DENSITY_TABLE: Record<string, number> = {
  // Proteins
  'chicken breast': 1.04,
  'beef': 1.05,
  'salmon': 1.02,
  'egg': 1.03,
  'tofu': 0.95,

  // Carbs
  'rice (cooked)': 0.81,
  'pasta (cooked)': 1.00,
  'bread': 0.27,
  'potato': 1.08,
  'oats (cooked)': 0.84,

  // Vegetables
  'broccoli': 0.35,
  'carrot': 0.64,
  'tomato': 0.62,
  'lettuce': 0.36,
  'cucumber': 0.96,

  // Fruits
  'apple': 0.64,
  'banana': 0.94,
  'orange': 0.87,
  'avocado': 0.92,
  'berries': 0.65,

  // Fats & oils
  'olive oil': 0.91,
  'butter': 0.91,
  'cheese': 1.00,

  // Nuts & seeds
  'almonds': 0.60,
  'peanuts': 0.64,
  'walnuts': 0.52,
};
