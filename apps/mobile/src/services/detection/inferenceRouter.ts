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
 *
 * Binary gate uses AIY Food V1 which outputs 2024 class probabilities.
 * The max confidence across all classes is used as the food score.
 * Uses manual loop (not Math.max(...array)) to avoid stack overflow
 * on 2024-element arrays.
 *
 * Accepts dual buffers: detectBuffer (640x640) for the detection stage,
 * and classifyBuffer (192x192) for the binary gate and classify stages.
 */

import { getModelSet } from './modelLoader';
import { decodeYoloOutput } from './postProcess';
import { FOOD_V1_CLASS_NAMES, COCO_FOOD_CLASS_IDS } from './constants';
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
 * Run the three-stage detection pipeline on preprocessed image buffers.
 *
 * @param detectBuffer   - Float32Array preprocessed at 640x640 for detection stage
 * @param classifyBuffer - Float32Array preprocessed at 192x192 for binary gate + classify
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

  // ── Stage 1: Binary gate ──
  // Uses classifyBuffer (192x192) since AIY Food V1 expects that input size.
  const binaryStart = performance.now();
  const binaryOutput = await models.binary.run([classifyBuffer]);
  const binaryTimeMs = performance.now() - binaryStart;
  pipelineStages.push({ stage: 'binary', timeMs: binaryTimeMs });

  // Interpret binary output: AIY Food V1 outputs 2024 class probabilities.
  // Max confidence across all classes = food score.
  // Manual loop avoids stack overflow on 2024-element array (no Math.max(...spread)).
  // model.run() returns TypedArray[] — convert to Float32Array for uniform access.
  const binaryScores = binaryOutput[0] instanceof Float32Array
    ? binaryOutput[0]
    : new Float32Array(binaryOutput[0] as ArrayBuffer);
  let binaryScore = 0;
  for (let i = 0; i < binaryScores.length; i++) {
    if (binaryScores[i] > binaryScore) binaryScore = binaryScores[i];
  }
  const isFood = binaryScore > BINARY_THRESHOLD;

  if (!isFood) {
    return {
      items: [],
      inferenceTimeMs: performance.now() - pipelineStart,
      pipelineStages,
    };
  }

  // ── Stage 2: Detection ──
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

  // ── Stage 3: Classification ──
  // Uses AIY Food V1 (192x192) to get a proper food label for detected items.
  // COCO only has 10 food classes (banana, apple, pizza, etc.) which produce
  // misleading labels. AIY Food V1 has 2024 food-specific classes.
  const classifyStart = performance.now();
  let topFoodLabel = 'Food item';
  if (foodDetections.length > 0) {
    const classifyOutput = await models.classify.run([classifyBuffer]);
    const classifyScores = classifyOutput[0] instanceof Float32Array
      ? classifyOutput[0]
      : new Float32Array(classifyOutput[0] as ArrayBuffer);

    // Find top valid food label from AIY Food V1's 2024 classes.
    // Skip index 0 (__background__) and Google KG IDs (start with '/').
    let topConf = 0;
    let topIdx = 0;
    for (let i = 1; i < classifyScores.length; i++) {
      if (classifyScores[i] > topConf) {
        const label = FOOD_V1_CLASS_NAMES[i];
        if (label && !label.startsWith('/')) {
          topConf = classifyScores[i];
          topIdx = i;
        }
      }
    }
    if (topIdx > 0 && topConf > 0.01) {
      topFoodLabel = FOOD_V1_CLASS_NAMES[topIdx];
    }
  }
  const classifyTimeMs = performance.now() - classifyStart;
  pipelineStages.push({ stage: 'classify', timeMs: classifyTimeMs });

  // ── Build DetectedItem array ──
  // Uses AIY Food V1 label instead of COCO class names for food items.
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
