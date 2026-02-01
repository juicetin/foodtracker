import 'dotenv/config';
import {
  processFoodImage,
  processBatchImages,
  type FoodAnalysisRequest,
} from './agents/coordinatorAgent.js';

// Export main functions
export { processFoodImage, processBatchImages };

// Export types
export type { FoodAnalysisRequest } from './agents/coordinatorAgent.js';
export type { VisionAnalysis, DetectedFoodItem } from './agents/visionAgent.js';
export type { ScaleReading } from './agents/scaleAgent.js';

// Simple test function
async function test() {
  console.log('🍎 Food Tracker AI Agent Service');
  console.log('================================\n');

  const testRequest: FoodAnalysisRequest = {
    imageUrl: 'https://example.com/food.jpg',
    userRegion: 'AU',
    userId: 'test-user',
  };

  console.log('Processing test image...');
  const result = await processFoodImage(testRequest);
  console.log('Result:', JSON.stringify(result, null, 2));
}

// Run test if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  test().catch(console.error);
}
