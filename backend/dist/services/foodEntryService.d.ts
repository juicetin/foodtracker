export interface CreateFoodEntryParams {
    userId: string;
    mealType: 'breakfast' | 'lunch' | 'dinner' | 'snack';
    entryDate: Date;
    photos: Array<{
        uri: string;
        gcsUrl?: string;
        width?: number;
        height?: number;
        latitude?: number;
        longitude?: number;
    }>;
    ingredients: Array<{
        name: string;
        quantity: number;
        unit: string;
        calories: number;
        protein?: number;
        carbs?: number;
        fat?: number;
        fiber?: number;
        sugar?: number;
        aiConfidence?: number;
        boundingBox?: {
            x: number;
            y: number;
            width: number;
            height: number;
        };
        databaseSource?: string;
        databaseId?: string;
    }>;
    notes?: string;
}
export interface FoodEntry {
    id: string;
    userId: string;
    mealType: string;
    entryDate: Date;
    totalCalories: number;
    totalProtein: number;
    totalCarbs: number;
    totalFat: number;
    notes?: string;
    createdAt: Date;
    updatedAt: Date;
    photos?: Array<any>;
    ingredients?: Array<any>;
}
export interface Ingredient {
    id: string;
    entryId: string;
    name: string;
    quantity: number;
    unit: string;
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber?: number;
    sugar?: number;
    userModified: boolean;
    originalQuantity?: number;
    databaseSource?: string;
    createdAt: Date;
    updatedAt: Date;
}
/**
 * Create a food entry with photos and ingredients
 */
export declare function createFoodEntry(params: CreateFoodEntryParams): Promise<FoodEntry>;
/**
 * Get food entries for a user
 */
export declare function getFoodEntries(userId: string, options?: {
    startDate?: Date;
    endDate?: Date;
    limit?: number;
}): Promise<FoodEntry[]>;
/**
 * Get a single food entry by ID
 */
export declare function getFoodEntry(entryId: string): Promise<FoodEntry | null>;
/**
 * Update an ingredient (for retrospective editing)
 */
export declare function updateIngredient(ingredientId: string, updates: Partial<{
    name: string;
    quantity: number;
    unit: string;
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
}>): Promise<Ingredient>;
/**
 * Get modification history for an ingredient
 */
export declare function getModificationHistory(ingredientId: string): Promise<{
    id: any;
    ingredientId: any;
    fieldName: any;
    oldValue: any;
    newValue: any;
    modifiedAt: any;
    modifiedBy: any;
}[]>;
/**
 * Delete a food entry
 */
export declare function deleteFoodEntry(entryId: string): Promise<void>;
