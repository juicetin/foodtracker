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
