/**
 * Two-stage inference pipeline router: detect -> classify.
 *
 * Orchestrates the detection pipeline sequentially:
 * 1. Detection: where are the food items? (GGCD YOLO 241 food-specific classes)
 * 2. Classification: what food is each item? (EfficientNet-Lite0 905-class labels)
 *
 * All 241 YOLO classes are food-specific (trained on GGCD dataset), so no
 * COCO food-class filtering is needed. Each detected item carries its own
 * YOLO-assigned food class name as the primary label.
 *
 * Accepts two buffers: detectBuffer (640x640) for the detection stage,
 * classifyBuffer (224x224, ImageNet-normalized) for the classify stage.
 */

import { getModelSet } from './modelLoader';
import { decodeYoloOutput } from './postProcess';
import {
  CLASSIFY_CLASS_NAMES,
} from './constants';
import type {
  InferenceResult,
  DetectedItem,
  PipelineStage,
  PortionEstimate,
} from './types';

/**
 * Classify confidence threshold. Below this, the classifier result is treated
 * as "not confident enough". Since YOLO now provides meaningful per-box food
 * labels, this threshold only affects whether the classifier's secondary
 * label is logged for debugging.
 */
const CLASSIFY_CONFIDENCE_THRESHOLD = 0.15;

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
 * Run the detection pipeline on preprocessed image buffers.
 *
 * @param detectBuffer   - Float32Array preprocessed at 640x640 for detection stage
 * @param classifyBuffer - Float32Array preprocessed at 224x224 with ImageNet normalization
 * @param imageWidth     - Width of the detection image (e.g. 640)
 * @param imageHeight    - Height of the detection image (e.g. 640)
 * @param classNames     - Array of 241 GGCD food class labels for detection output decoding
 * @returns InferenceResult with detected items and timing metrics
 * @throws If models are not loaded (call loadModelSet() first)
 */
export async function runDetectionPipeline(
  detectBuffer: Float32Array,
  classifyBuffer: Float32Array,
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

  // ── Stage 1: Detection ──
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
  );

  // All 241 GGCD classes are food -- no COCO filtering needed.
  // Every detection is a food item.
  const foodDetections = rawDetections;

  // ── Stage 2: Classification ──
  // Uses EfficientNet-Lite0 (224x224 with ImageNet normalization) for a
  // secondary food label. With GGCD YOLO providing meaningful per-box
  // food names, the classifier serves as confirmation/refinement.
  const classifyStart = performance.now();
  if (foodDetections.length > 0) {
    const classifyOutput = await models.classify.run([classifyBuffer]);
    const classifyScores = classifyOutput[0] instanceof Float32Array
      ? classifyOutput[0]
      : new Float32Array(classifyOutput[0] as ArrayBuffer);

    // Find top class from EfficientNet-Lite0's output logits.
    let topConf = 0;
    let topIdx = 0;
    for (let i = 0; i < classifyScores.length; i++) {
      if (classifyScores[i] > topConf) {
        topConf = classifyScores[i];
        topIdx = i;
      }
    }

    // Log classifier result for debugging -- YOLO label is primary.
    if (__DEV__) {
      if (topConf >= CLASSIFY_CONFIDENCE_THRESHOLD && topIdx < CLASSIFY_CLASS_NAMES.length) {
        const classifyLabel = formatFoodLabel(CLASSIFY_CLASS_NAMES[topIdx]);
        console.log(`[Detection] Classifier secondary label: ${classifyLabel} (${(topConf * 100).toFixed(1)}%)`);
      }
    }
  }
  const classifyTimeMs = performance.now() - classifyStart;
  pipelineStages.push({ stage: 'classify', timeMs: classifyTimeMs });

  // ── Build DetectedItem array ──
  // Each detection uses its own YOLO-assigned food class name (from GGCD).
  // The per-box YOLO label is the primary className for each item.
  const items: DetectedItem[] = foodDetections.map((det) => ({
    id: generateDetectionId(),
    className: formatFoodLabel(det.className),
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
  }));

  return {
    items,
    inferenceTimeMs: performance.now() - pipelineStart,
    pipelineStages,
  };
}
