/**
 * Single-stage inference pipeline router: bbox-only detection.
 *
 * Detects food bounding boxes using GGCD YOLO (241 food-specific classes).
 * Each detection is labelled "Food Region" and flagged as isRefining=true,
 * pending VLM identification (handled by a separate pipeline).
 *
 * Accepts a single detectBuffer (640x640) for the detection stage.
 * Classification stage has been removed (Phase 7).
 */

import { getModelSet } from './modelLoader';
import { decodeYoloOutput } from './postProcess';
import type {
  InferenceResult,
  DetectedItem,
  PipelineStage,
  PortionEstimate,
} from './types';

/** Counter for generating unique detection IDs within a session. */
let detectionCounter = 0;

/**
 * Generate a unique ID for a detected item.
 * Uses a monotonic counter since crypto.randomUUID may not be available
 * in all React Native runtimes.
 */
function generateDetectionId(): string {
  detectionCounter += 1;
  return `det_${Date.now()}_${detectionCounter}`;
}

/**
 * Default placeholder portion estimate.
 * Plan 03's portionBridge will fill in real estimates after inference.
 */
function defaultPortionEstimate(): PortionEstimate {
  return {
    weightG: 0,
    confidence: 'low',
    method: 'pending',
    suggestReference: false,
    details: {},
  };
}

/**
 * Format a food label into a readable title-case string.
 * Handles hyphens, underscores, and spaces in GGCD and classifier names.
 *
 * Examples:
 *   "airan-katyk"                 -> "Airan Katyk"
 *   "pad_thai"                    -> "Pad Thai"
 *   "vegetable based cooked food" -> "Vegetable Based Cooked Food"
 *   "grilled-cheese_sandwich"     -> "Grilled Cheese Sandwich"
 */
export function formatFoodLabel(rawLabel: string): string {
  return rawLabel
    .split(/[-_\s]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Run the bbox-only detection pipeline on a preprocessed image buffer.
 *
 * @param detectBuffer   - Float32Array preprocessed at 640x640 for detection stage
 * @param imageWidth     - Width of the detection image (e.g. 640)
 * @param imageHeight    - Height of the detection image (e.g. 640)
 * @param classNames     - Array of 241 GGCD food class labels for detection output decoding
 * @returns InferenceResult with detected items and timing metrics
 * @throws If models are not loaded (call loadModelSet() first)
 */
export async function runBboxDetection(
  detectBuffer: Float32Array,
  imageWidth: number,
  imageHeight: number,
  classNames: string[],
): Promise<InferenceResult> {
  const models = getModelSet();
  if (!models) {
    throw new Error(
      'Models not loaded. Call loadModelSet() before running the pipeline.',
    );
  }

  const pipelineStart = performance.now();
  const pipelineStages: PipelineStage[] = [];

  // -- Stage 1: Detection --
  // Uses detectBuffer (640x640) since YOLO expects that input size.
  const detectStart = performance.now();
  const detectOutput = await models.detect.run([detectBuffer]);
  const detectTimeMs = performance.now() - detectStart;
  pipelineStages.push({ stage: 'detect', timeMs: detectTimeMs });

  // Decode YOLO output tensor into raw detections
  // model.run() returns TypedArray[] -- convert to Float32Array for uniform access.
  const detectTensor = detectOutput[0] instanceof Float32Array
    ? detectOutput[0]
    : new Float32Array(detectOutput[0] as ArrayBuffer);

  // Determine number of predictions from output shape
  // YOLO output shape: [1, 4+nc, numPredictions]
  // Total elements = (4 + numClasses) * numPredictions
  const numClasses = classNames.length;
  const stride = 4 + numClasses;
  const numPredictions = stride > 0 ? Math.floor(detectTensor.length / stride) : 0;

  const rawDetections = decodeYoloOutput(
    detectTensor,
    numClasses,
    numPredictions,
    classNames,
    imageWidth,
    imageHeight,
  );

  // All 241 GGCD classes are food -- no COCO filtering needed.
  const foodDetections = rawDetections;

  // Log YOLO labels in dev mode for debugging
  if (__DEV__ && foodDetections.length > 0) {
    for (const det of foodDetections) {
      console.log(`[Detection] YOLO bbox: ${formatFoodLabel(det.className)} (${(det.confidence * 100).toFixed(1)}%)`);
    }
  }

  // -- Build DetectedItem array --
  // All items labelled "Food Region" with isRefining=true.
  // VLM will identify the actual food name asynchronously.
  const items: DetectedItem[] = foodDetections.map((det) => ({
    id: generateDetectionId(),
    className: 'Food Region',
    confidence: det.confidence,
    bbox: {
      x: det.x,
      y: det.y,
      w: det.w,
      h: det.h,
    },
    portionEstimate: defaultPortionEstimate(),
    portionMultiplier: 1,
    isRemoved: false,
    isRefining: true,
  }));

  return {
    items,
    inferenceTimeMs: performance.now() - pipelineStart,
    pipelineStages,
  };
}
