import { LlmAgent } from '@google/adk';
import { VisionAnalysis } from './visionAgent.js';
import { ScaleReading } from './scaleAgent.js';
/**
 * CoordinatorAgent: Root orchestrator for food image processing
 *
 * Orchestrates:
 * 1. Vision detection (parallel for batch)
 * 2. Scale reading (if scale detected)
 * 3. Volume estimation
 * 4. Hypothesis branching (for scale weights)
 * 5. Database lookup
 * 6. Final ingredient compilation
 */
export interface FoodAnalysisRequest {
    imageUrl: string;
    imageBase64?: string;
    userRegion: 'AU' | 'US' | 'CA' | 'UK' | 'FR' | 'global';
    userId: string;
}
export interface FoodAnalysisResult {
    imageUrl: string;
    visionAnalysis: VisionAnalysis;
    scaleReading?: ScaleReading;
    ingredients: Array<{
        name: string;
        quantity: number;
        unit: string;
        calories: number;
        protein: number;
        carbs: number;
        fat: number;
    }>;
    totalNutrition: {
        calories: number;
        protein: number;
        carbs: number;
        fat: number;
    };
    processingTime: number;
}
/**
 * Coordinator agent that orchestrates the entire food analysis pipeline
 */
export declare const coordinatorAgent: LlmAgent;
/**
 * Process a single food image
 */
export declare function processFoodImage(request: FoodAnalysisRequest): Promise<FoodAnalysisResult>;
/**
 * Process multiple images in parallel (batch processing)
 */
export declare function processBatchImages(requests: FoodAnalysisRequest[]): Promise<FoodAnalysisResult[]>;
