import * as foodEntryService from './foodEntryService.js';
/**
 * AI Service - Interfaces with the AI agent service (Google ADK)
 */
interface Photo {
    id: string;
    uri: string;
    gcsUrl?: string;
    timestamp: Date;
}
interface ProcessPhotosRequest {
    photos: Photo[];
    userId: string;
    userRegion: 'AU' | 'US' | 'CA' | 'UK' | 'FR' | 'global';
    mealType: 'breakfast' | 'lunch' | 'dinner' | 'snack';
}
/**
 * Process food photos using the AI agent service
 */
export declare function processFoodPhotos(request: ProcessPhotosRequest): Promise<{
    entry: foodEntryService.FoodEntry;
    processingTime: number;
}>;
/**
 * Get food logs for a user
 */
export declare function getFoodLogs(userId: string, options?: {
    limit?: number;
    startDate?: Date;
    endDate?: Date;
}): Promise<foodEntryService.FoodEntry[]>;
export {};
