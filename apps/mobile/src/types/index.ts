// Navigation types
export type RootStackParamList = {
  Main: undefined;
  EntryDetail: { entryId: string };
};

export type MainTabParamList = {
  Home: undefined;
  Diary: undefined;
  Profile: undefined;
};

// Food entry types
export interface Photo {
  id: string;
  uri: string;
  gcsUrl?: string;
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
  sourceSegment?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  aiConfidence?: number;
  userModified: boolean;
  databaseSource: 'AFCD' | 'USDA' | 'CNF' | 'CoFID' | 'CIQUAL' | 'OpenFoodFacts';
}

export interface FoodEntry {
  id: string;
  userId: string;
  createdAt: Date;
  mealType: 'breakfast' | 'lunch' | 'dinner' | 'snack';
  photos: Photo[];
  ingredients: Ingredient[];
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
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
  branches?: HypothesisBranch[];
  selectedBranch?: string;
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

export interface HypothesisBranch {
  id: string;
  name: string;
  description: string;
  foodWeight: number;
  error: number;
  ingredients: Ingredient[];
}

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
  darkMode: boolean;
}

// API response types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
