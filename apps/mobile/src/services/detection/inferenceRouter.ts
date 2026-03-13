/**
 * Two-stage inference pipeline router: detect -> classify.
 *
 * Orchestrates the detection pipeline sequentially:
 * 1. Detection: where are the food items? (YOLO bounding boxes, filtered to COCO food classes)
 * 2. Classification: what food is each item? (EfficientNet-Lite0 335-class labels)
 *
 * The binary gate is no longer needed because EfficientNet-Lite0 is trained
 * exclusively on food images -- a confidence threshold on its output serves
 * the same purpose. Food-101 fallback is unnecessary because its 101 classes
 * are a strict subset of the new 335 classes.
 *
 * Accepts two buffers: detectBuffer (640x640) for the detection stage,
 * classifyBuffer (224x224, ImageNet-normalized) for the classify stage.
 */

import { getModelSet } from './modelLoader';
import { decodeYoloOutput } from './postProcess';
import {
  CLASSIFY_CLASS_NAMES,
  COCO_FOOD_CLASS_IDS,
} from './constants';
import type {
  InferenceResult,
  DetectedItem,
  PipelineStage,
  PortionEstimate,
} from './types';

/**
 * Classify confidence threshold. Below this, the classifier result is treated
 * as "not confident enough" and items get a generic "Food Item" fallback label.
 * This is NOT a binary gate -- all COCO food detections are returned regardless.
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
 * Format a snake_case class name into a readable title-case label.
 * e.g. "pad_thai" -> "Pad Thai", "ramen" -> "Ramen",
 *      "grilled_cheese_sandwich" -> "Grilled Cheese Sandwich"
 */
function formatClassLabel(rawLabel: string): string {
  return rawLabel
    .split('_')
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
 * @param classNames     - Array of class labels for detection output decoding
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
  // model.run() returns TypedArray[] — convert to Float32Array for uniform access.
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

  // Filter to COCO food class detections only (remove person, car, chair, etc.)
  const foodDetections = rawDetections.filter((det) =>
    COCO_FOOD_CLASS_IDS.has(det.classId),
  );

  // ── Stage 2: Classification ──
  // Uses EfficientNet-Lite0 (224x224 with ImageNet normalization) to get
  // a proper food label for detected items.
  // COCO only has 10 food classes (banana, apple, pizza, etc.) which produce
  // misleading labels. EfficientNet-Lite0 has 335 food-specific classes.
  const classifyStart = performance.now();
  let topFoodLabel = 'Food Item';
  if (foodDetections.length > 0) {
    const classifyOutput = await models.classify.run([classifyBuffer]);
    const classifyScores = classifyOutput[0] instanceof Float32Array
      ? classifyOutput[0]
      : new Float32Array(classifyOutput[0] as ArrayBuffer);

    // Find top class from EfficientNet-Lite0's 335 output logits.
    let topConf = 0;
    let topIdx = 0;
    for (let i = 0; i < classifyScores.length; i++) {
      if (classifyScores[i] > topConf) {
        topConf = classifyScores[i];
        topIdx = i;
      }
    }

    if (topConf >= CLASSIFY_CONFIDENCE_THRESHOLD && topIdx < CLASSIFY_CLASS_NAMES.length) {
      topFoodLabel = formatClassLabel(CLASSIFY_CLASS_NAMES[topIdx]);
    }
  }
  const classifyTimeMs = performance.now() - classifyStart;
  pipelineStages.push({ stage: 'classify', timeMs: classifyTimeMs });

  // ── Build DetectedItem array ──
  // All COCO food detections are returned with the classify label.
  // If multiple food boxes are detected, they all share the same classify
  // label (whole-image classification, same as before).
  const items: DetectedItem[] = foodDetections.map((det) => ({
    id: generateDetectionId(),
    className: topFoodLabel,
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
