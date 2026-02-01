import 'dotenv/config';
import { processFoodImage, processBatchImages } from './agents/coordinatorAgent.js';
export { processFoodImage, processBatchImages };
export type { FoodAnalysisRequest } from './agents/coordinatorAgent.js';
export type { VisionAnalysis, DetectedFoodItem } from './agents/visionAgent.js';
export type { ScaleReading } from './agents/scaleAgent.js';
