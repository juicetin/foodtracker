import { LlmAgent } from '@google/adk';
/**
 * VisionAgent: Analyzes food images to detect and identify food items
 *
 * Responsibilities:
 * - Image segmentation and food item detection
 * - Identify bounding boxes for each food item
 * - Detect container type (plate, bowl, scale)
 * - Provide confidence scores
 */
export interface DetectedFoodItem {
    name: string;
    boundingBox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    confidence: number;
}
export interface VisionAnalysis {
    items: DetectedFoodItem[];
    containerType: 'plate' | 'bowl' | 'scale' | 'other';
    hasScale: boolean;
    imageQuality: 'good' | 'fair' | 'poor';
}
export declare const visionAgent: LlmAgent;
