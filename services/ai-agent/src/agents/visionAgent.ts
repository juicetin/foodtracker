import { LlmAgent, Gemini } from '@google/adk';

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

const VISION_PROMPT = `You are a food detection expert. Analyze the provided image and identify all food items visible.

For each food item, provide:
1. The name of the food (be specific: "grilled chicken breast" not just "chicken")
2. Approximate bounding box (x, y, width, height as percentages of image dimensions)
3. Confidence score (0-1)

Also identify:
- The type of container (plate, bowl, scale, or other)
- Whether a kitchen scale is visible in the image
- The overall image quality for food detection

Return your analysis as a JSON object with this structure:
{
  "items": [
    {
      "name": "food name",
      "boundingBox": { "x": 0, "y": 0, "width": 0, "height": 0 },
      "confidence": 0.0
    }
  ],
  "containerType": "plate" | "bowl" | "scale" | "other",
  "hasScale": true | false,
  "imageQuality": "good" | "fair" | "poor"
}

Be thorough but only include items you can clearly identify. If the image quality is poor, note that.`;

export const visionAgent = new LlmAgent({
  name: 'VisionAgent',
  description: 'Analyzes food images to detect and identify food items',
  model: new Gemini({ model: 'gemini-2.0-flash-exp' }),
  instruction: VISION_PROMPT,
});
