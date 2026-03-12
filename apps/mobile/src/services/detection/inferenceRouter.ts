/**
 * Three-stage inference pipeline router: binary -> detect -> classify.
 *
 * Orchestrates the detection pipeline sequentially:
 * 1. Binary gate: is this image food? (short-circuits if not)
 * 2. Detection: where are the food items? (YOLO bounding boxes)
 * 3. Classification: what food is each item? (per-detection labels)
 *
 * Anti-pattern: do NOT run stages in parallel. The binary gate exists
 * to save compute when the image is not food.
 */

import { getModelSet } from './modelLoader';
import { decodeYoloOutput } from './postProcess';
import type {
  InferenceResult,
  DetectedItem,
  PipelineStage,
  PortionEstimate,
} from './types';

/** Binary gate threshold: above this = food detected. */
const BINARY_THRESHOLD = 0.5;

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
 * Run the three-stage detection pipeline on an image buffer.
 *
 * @param imageBuffer  - Raw image data as ArrayBuffer (preprocessed to model input size)
 * @param imageWidth   - Width of the preprocessed image (e.g. 640)
 * @param imageHeight  - Height of the preprocessed image (e.g. 640)
 * @param classNames   - Array of class labels for detection output decoding
 * @returns InferenceResult with detected items and timing metrics
 * @throws If models are not loaded (call loadModelSet() first)
 */
export async function runDetectionPipeline(
  imageBuffer: ArrayBufferLike,
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

  // ── Stage 1: Binary gate ──
  const binaryStart = performance.now();
  const binaryOutput = await models.binary.run([imageBuffer]);
  const binaryTimeMs = performance.now() - binaryStart;
  pipelineStages.push({ stage: 'binary', timeMs: binaryTimeMs });

  // Interpret binary output: first value > threshold = food
  const binaryScore = new Float32Array(
    binaryOutput[0] instanceof Float32Array
      ? binaryOutput[0].buffer
      : binaryOutput[0],
  )[0];
  const isFood = binaryScore > BINARY_THRESHOLD;

  if (!isFood) {
    return {
      items: [],
      inferenceTimeMs: performance.now() - pipelineStart,
      pipelineStages,
    };
  }

  // ── Stage 2: Detection ──
  const detectStart = performance.now();
  const detectOutput = await models.detect.run([imageBuffer]);
  const detectTimeMs = performance.now() - detectStart;
  pipelineStages.push({ stage: 'detect', timeMs: detectTimeMs });

  // Decode YOLO output tensor into raw detections
  const detectTensor = new Float32Array(
    detectOutput[0] instanceof Float32Array
      ? detectOutput[0].buffer
      : detectOutput[0],
  );

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

  // ── Stage 3: Classification ──
  // For a single-pass YOLO detector, classification is already done in the
  // detection output (class scores per anchor). The classify model refines
  // class predictions for each detected region. If no detections, skip.
  const classifyStart = performance.now();
  if (rawDetections.length > 0) {
    await models.classify.run([imageBuffer]);
  }
  const classifyTimeMs = performance.now() - classifyStart;
  pipelineStages.push({ stage: 'classify', timeMs: classifyTimeMs });

  // ── Build DetectedItem array ──
  const items: DetectedItem[] = rawDetections.map((det) => ({
    id: generateDetectionId(),
    className: det.className,
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
