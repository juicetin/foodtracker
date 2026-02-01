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
export async function processFoodPhotos(request: ProcessPhotosRequest) {
  const startTime = Date.now();

  // TODO: Call the actual AI agent service
  // For now, return mock data
  console.log(`Processing ${request.photos.length} photos...`);
  console.log(`User region: ${request.userRegion}`);

  // Simulate AI processing delay
  await new Promise(resolve => setTimeout(resolve, 1500));

  // Mock AI response - detect ingredients
  const mockIngredients = [
    {
      name: 'Grilled Chicken Breast',
      quantity: 150,
      unit: 'g',
      calories: 165,
      protein: 31,
      carbs: 0,
      fat: 3.6,
      databaseSource: request.userRegion === 'AU' ? 'AFCD' : 'USDA',
    },
    {
      name: 'Steamed Broccoli',
      quantity: 100,
      unit: 'g',
      calories: 35,
      protein: 2.8,
      carbs: 7,
      fat: 0.4,
      databaseSource: request.userRegion === 'AU' ? 'AFCD' : 'USDA',
    },
  ];

  // Create actual food entry in database
  const entry = await foodEntryService.createFoodEntry({
    userId: request.userId,
    mealType: request.mealType,
    entryDate: new Date(),
    photos: request.photos.map(p => ({
      uri: p.uri,
      gcsUrl: p.gcsUrl,
      width: 1920,
      height: 1080,
    })),
    ingredients: mockIngredients,
  });

  const processingTime = Date.now() - startTime;

  return {
    entry,
    processingTime,
  };
}

/**
 * Get food logs for a user
 */
export async function getFoodLogs(userId: string, options?: { limit?: number; startDate?: Date; endDate?: Date }) {
  return foodEntryService.getFoodEntries(userId, options);
}
