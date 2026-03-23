// Navigation types
export type RootStackParamList = {
  Main: undefined;
  EntryDetail: { entryId: string };
  Detection: undefined;
  FoodSearch: undefined;
  BarcodeScan: undefined;
  Recipes: undefined;
  GeminiNanoTest: undefined;
  QuickAdd: undefined;
  ReidentifyMerge: { entryId: string };
  SyncSettings: undefined;
  GalleryScan: undefined;
  ScaleInput: { photoUri?: string; onResult?: (netWeight: number) => void };
  WeightTrend: undefined;
  AddFood: { mealType?: string };
};

export type MainTabParamList = {
  Today: undefined;
  Add: undefined;
  Insights: undefined;
  Profile: undefined;
};

// Food entry types
export interface Photo {
  id: string;
  uri: string;
  localPath?: string;
  timestamp: Date;
  metadata?: {
    width: number;
    height: number;
    location?: {
      latitude: number;
      longitude: number;
    };
  };
}

export interface Ingredient {
  id: string;
  name: string;
  quantity: number;
  unit: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  sourceSegment?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  aiConfidence?: number;
  userModified: boolean;
  databaseSource:
    | 'AFCD'
    | 'USDA'
    | 'CNF'
    | 'CoFID'
    | 'CIQUAL'
    | 'OpenFoodFacts'
    | 'branded';
  originalQuantity?: number;
}

export interface FoodEntry {
  id: string;
  createdAt: string;
  entryDate: string;
  mealType: 'breakfast' | 'lunch' | 'dinner' | 'snack';
  photos: Photo[];
  ingredients: Ingredient[];
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  notes?: string;
  updatedAt: string;
  isSynced: boolean;
  isDeleted: boolean;
  modificationHistory?: ModificationEvent[];
}

export interface ModificationEvent {
  timestamp: Date;
  type: 'add' | 'remove' | 'modify';
  ingredientId: string;
  oldValue?: any;
  newValue?: any;
}

// AI processing types
export interface AIProcessingResult {
  photos: Photo[];
  detectedItems: DetectedItem[];
  scaleWeight?: ScaleReading;
}

export interface DetectedItem {
  name: string;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  estimatedVolume?: number;
  estimatedWeight?: number;
}

export interface ScaleReading {
  value: number;
  unit: 'g' | 'kg' | 'oz' | 'lb';
  confidence: number;
  photoId: string;
}

// ---------------------------------------------------------------------------
// Scan result types (Gemini Nano pipeline output)
// ---------------------------------------------------------------------------

/** A single ingredient as identified by Gemini Nano, enriched with KG nutrition. */
export interface ScannedIngredient {
  id: string;
  name: string;
  /** Current weight in grams (may be scaled or manually edited). */
  amount_g: number;
  /** Locked original Gemini Nano estimate — used as scaling baseline. */
  originalAmount_g: number;
  /** Nutrition values are for originalAmount_g. Scale by amount_g/originalAmount_g to display. */
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sodium: number;
  nutritionSource: 'kg' | 'proxy';
  userModified: boolean;
}

/** A dish identified in a single photo scan. */
export interface ScannedDish {
  id: string;
  name: string;
  cuisine: string | null;
  /** URI of the source photo this dish was identified from. */
  photoUri: string;
  ingredients: ScannedIngredient[];
  /** Multiplicative scale applied to all non-userModified ingredients (1.0 = Gemini estimate). */
  portionScale: number;
  /** Names of ingredients inferred from KG (not VLM). Present when KG fills in missing ingredients. */
  kgInferredIngredients?: string[];
}

/** Full result of a single food scan (photo + all dishes). */
export interface ScanResult {
  photoUri: string;
  dishes: ScannedDish[];
  /** True when Gemini Nano is unavailable and mock data is used. */
  isMock: boolean;
}

// UX modes for recipe/food logging workflow
export type UxMode = 'zero-effort' | 'confirm-only' | 'guided-edit';

// Theme preference
export type ThemePreference = 'system' | 'light' | 'dark';

// User preferences
export interface UserPreferences {
  region: 'AU' | 'US' | 'CA' | 'UK' | 'FR' | 'global';
  units: 'metric' | 'imperial';
  nutritionGoals: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  themePreference: ThemePreference;
  uxMode: UxMode;
}
