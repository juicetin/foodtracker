import { apiClient } from './client';
import { FoodEntry, Photo, APIResponse } from '../../types';

export interface ProcessPhotosRequest {
  photos: Photo[];
  userId: string;
  userRegion: 'AU' | 'US' | 'CA' | 'UK' | 'FR' | 'global';
  mealType?: 'breakfast' | 'lunch' | 'dinner' | 'snack';
}

export interface ProcessPhotosResponse {
  entry: FoodEntry;
  processingTime: number;
}

/**
 * Upload photos and process them with AI to create a food entry
 */
export async function processPhotos(
  request: ProcessPhotosRequest
): Promise<APIResponse<ProcessPhotosResponse>> {
  try {
    // First, upload photos to get GCS URLs
    const uploadPromises = request.photos.map(async (photo) => {
      // TODO: Implement actual photo upload
      // For now, return the local URI
      return {
        ...photo,
        gcsUrl: photo.uri, // Will be replaced with actual GCS URL
      };
    });

    const uploadedPhotos = await Promise.all(uploadPromises);

    // Then, send to AI service for processing
    const response = await apiClient.post<ProcessPhotosResponse>(
      '/food-logs/process',
      {
        photos: uploadedPhotos,
        userId: request.userId,
        userRegion: request.userRegion,
        mealType: request.mealType,
      }
    );

    return {
      success: true,
      data: response,
    };
  } catch (error) {
    console.error('Error processing photos:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to process photos',
    };
  }
}

/**
 * Get all food entries for a user
 */
export async function getFoodEntries(userId: string): Promise<APIResponse<FoodEntry[]>> {
  try {
    const entries = await apiClient.get<FoodEntry[]>(`/food-logs?userId=${userId}`);
    return {
      success: true,
      data: entries,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to fetch entries',
    };
  }
}

/**
 * Update a food entry (for retrospective editing)
 */
export async function updateFoodEntry(
  entryId: string,
  updates: Partial<FoodEntry>
): Promise<APIResponse<FoodEntry>> {
  try {
    const entry = await apiClient.put<FoodEntry>(`/food-logs/${entryId}`, updates);
    return {
      success: true,
      data: entry,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to update entry',
    };
  }
}

/**
 * Delete a food entry
 */
export async function deleteFoodEntry(entryId: string): Promise<APIResponse<void>> {
  try {
    await apiClient.delete(`/food-logs/${entryId}`);
    return {
      success: true,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to delete entry',
    };
  }
}
