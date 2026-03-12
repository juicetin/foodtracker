/**
 * Detection pipeline type contracts.
 *
 * All detection-related modules import from this file. Types cover the full
 * pipeline: raw YOLO output -> enriched DetectedItem -> session state.
 *
 * Confidence thresholds are per locked decision (ADR-005):
 *   green  >= 80%  -> 'high'
 *   yellow  50-79% -> 'medium'
 *   red    < 50%   -> 'low'
 */

// ---------------------------------------------------------------------------
// Bounding box (normalised 0-1 coordinates)
// ---------------------------------------------------------------------------

/** Bounding box in normalised coordinates (0-1). */
export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

/** Confidence bucket per locked thresholds. */
export type ConfidenceLevel = 'high' | 'medium' | 'low';

/** Map a raw confidence score (0-1) to a bucket. */
export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.8) return 'high';
  if (confidence >= 0.5) return 'medium';
  return 'low';
}

/** Tailwind-friendly colour tokens for each confidence level. */
export const CONFIDENCE_COLORS: Record<ConfidenceLevel, string> = {
  high: '#22C55E', // green-500
  medium: '#EAB308', // yellow-500
  low: '#EF4444', // red-500
};

// ---------------------------------------------------------------------------
// Portion estimation
// ---------------------------------------------------------------------------

/** Portion estimation result (mirrors Python PortionEstimate). */
export interface PortionEstimate {
  weightG: number;
  confidence: string; // 'high' | 'medium' | 'low'
  method: string; // 'geometry' | 'user_history' | 'usda_default'
  suggestReference: boolean;
  details: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Detected item
// ---------------------------------------------------------------------------

/** Single detected food item after enrichment. */
export interface DetectedItem {
  id: string;
  className: string;
  confidence: number; // 0-1
  bbox: BoundingBox;
  portionEstimate: PortionEstimate;
  portionMultiplier: number; // 0.5-3.0 slider value
  isRemoved: boolean; // soft-delete for undo
  removedAt?: number; // timestamp for undo timeout
  correctedFrom?: string; // original class name if user corrected
}

// ---------------------------------------------------------------------------
// Inference pipeline
// ---------------------------------------------------------------------------

/** Stage timing for the three-stage pipeline. */
export interface PipelineStage {
  stage: 'binary' | 'detect' | 'classify';
  timeMs: number;
}

/** Full inference pipeline result. */
export interface InferenceResult {
  items: DetectedItem[];
  inferenceTimeMs: number;
  pipelineStages: PipelineStage[];
}

// ---------------------------------------------------------------------------
// Raw detection (pre-enrichment)
// ---------------------------------------------------------------------------

/** Raw YOLO detection before enrichment with portion/nutrient data. */
export interface RawDetection {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
  classId: number;
  className: string;
}

// ---------------------------------------------------------------------------
// Model handles
// ---------------------------------------------------------------------------

/**
 * Opaque model handle from react-native-fast-tflite.
 *
 * Matches the TensorflowModel interface returned by loadTensorflowModel().
 * Both sync and async run methods accept an array of typed-array buffers
 * and return an array of typed-array buffers.
 */
export interface TFLiteModel {
  run: (input: ArrayBufferLike[]) => Promise<ArrayBufferLike[]>;
  runSync: (input: ArrayBufferLike[]) => ArrayBufferLike[];
}

/** Model set for the three-stage detection pipeline. */
export interface ModelSet {
  binary: TFLiteModel;
  detect: TFLiteModel;
  classify: TFLiteModel;
}

// ---------------------------------------------------------------------------
// Meal type
// ---------------------------------------------------------------------------

/** Meal type auto-detection per locked decision. */
export type MealType = 'breakfast' | 'lunch' | 'snack' | 'dinner';

/** Auto-detect meal type from current wall-clock hour. */
export function autoDetectMealType(): MealType {
  const hour = new Date().getHours();
  if (hour < 10) return 'breakfast';
  if (hour < 14) return 'lunch';
  if (hour < 17) return 'snack';
  if (hour < 21) return 'dinner';
  return 'snack';
}

// ---------------------------------------------------------------------------
// Correction history
// ---------------------------------------------------------------------------

/** Record of a user correction (original -> corrected class name). */
export interface CorrectionRecord {
  id: string;
  originalClassName: string;
  correctedClassName: string;
  confidence: number;
  correctedAt: string; // ISO 8601
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/** Detection session state shape (used by Zustand store). */
export interface DetectionSessionState {
  photoUri: string | null;
  photoWidth: number;
  photoHeight: number;
  items: DetectedItem[];
  isDetecting: boolean;
  mealType: MealType;
  selectedItemId: string | null;
}
