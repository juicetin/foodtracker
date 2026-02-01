export interface CustomRecipe {
    id: string;
    userId: string;
    name: string;
    description?: string;
    sourceEntryId?: string;
    totalCalories: number;
    totalProtein: number;
    totalCarbs: number;
    totalFat: number;
    timesUsed: number;
    lastUsedAt?: Date;
    createdAt: Date;
    updatedAt: Date;
    ingredients: Array<{
        id: string;
        name: string;
        quantity: number;
        unit: string;
        calories: number;
        protein: number;
        carbs: number;
        fat: number;
    }>;
    photos?: Array<{
        id: string;
        gcsUrl: string;
        isPrimary: boolean;
    }>;
}
/**
 * Create a custom recipe from an existing food entry
 */
export declare function createRecipeFromEntry(userId: string, entryId: string, name: string, description?: string): Promise<CustomRecipe>;
/**
 * Get all recipes for a user
 */
export declare function getRecipes(userId: string): Promise<CustomRecipe[]>;
/**
 * Get a single recipe by ID
 */
export declare function getRecipe(recipeId: string): Promise<CustomRecipe | null>;
/**
 * Use a recipe to create a new food entry
 */
export declare function useRecipe(recipeId: string, userId: string, options?: {
    mealType?: 'breakfast' | 'lunch' | 'dinner' | 'snack';
    entryDate?: Date;
}): Promise<import("./foodEntryService.js").FoodEntry>;
/**
 * Search recipes by name
 */
export declare function searchRecipes(userId: string, query: string): Promise<CustomRecipe[]>;
/**
 * Delete a recipe
 */
export declare function deleteRecipe(recipeId: string): Promise<void>;
