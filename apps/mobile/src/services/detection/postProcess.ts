/**
 * YOLO output post-processing: tensor decoding and Non-Max Suppression.
 *
 * YOLO detection output shape: [1, 4+nc, numPredictions] stored row-major.
 * The tensor is TRANSPOSED compared to the intuitive [predictions, channels]
 * layout -- access as output[row * numPredictions + col].
 *
 * First 4 rows: cx, cy, w, h (normalised 0-1 coordinates)
 * Remaining rows: per-class confidence scores
 */

import type { RawDetection } from './types';

/**
 * Decode raw YOLO output tensor into RawDetection array.
 *
 * CRITICAL: Uses transposed access pattern output[row * numPredictions + col]
 * as documented in RESEARCH.md (Pitfall #1). The alternative
 * output[predIdx * stride + channel] produces garbage bounding boxes.
 *
 * @param output       - Flat Float32Array from model.run(), shape [1, 4+nc, numPredictions]
 * @param numClasses   - Number of object classes (nc)
 * @param numPredictions - Number of predictions (typically 8400 for 640x640 input)
 * @param classNames   - Human-readable class labels, indexed by classId
 * @param confThreshold - Minimum confidence to keep a detection (default 0.25)
 * @returns Filtered and NMS-processed detections
 */
export function decodeYoloOutput(
  output: Float32Array,
  numClasses: number,
  numPredictions: number,
  classNames: string[],
  confThreshold: number = 0.25,
): RawDetection[] {
  if (numPredictions === 0) return [];

  const detections: RawDetection[] = [];

  for (let i = 0; i < numPredictions; i++) {
    // Transposed access: output[row * numPredictions + col]
    const cx = output[0 * numPredictions + i];
    const cy = output[1 * numPredictions + i];
    const w = output[2 * numPredictions + i];
    const h = output[3 * numPredictions + i];

    // Find class with maximum confidence
    let maxConf = 0;
    let maxClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const conf = output[(4 + c) * numPredictions + i];
      if (conf > maxConf) {
        maxConf = conf;
        maxClassId = c;
      }
    }

    // Filter by confidence threshold
    if (maxConf >= confThreshold) {
      detections.push({
        // Convert center-format to corner-format
        x: cx - w / 2,
        y: cy - h / 2,
        w,
        h,
        confidence: maxConf,
        classId: maxClassId,
        className: classNames[maxClassId] ?? `class_${maxClassId}`,
      });
    }
  }

  return nonMaxSuppression(detections, 0.45);
}

/**
 * Greedy Non-Max Suppression.
 *
 * Sorts detections by confidence descending, then iteratively keeps
 * detections whose IoU with all previously kept detections is below
 * the threshold. This removes duplicate/overlapping bounding boxes,
 * keeping only the highest-confidence detection in each cluster.
 *
 * @param detections   - Array of raw detections (unsorted)
 * @param iouThreshold - Maximum IoU before a detection is suppressed (default 0.45)
 * @returns Filtered detections with overlaps removed
 */
export function nonMaxSuppression(
  detections: RawDetection[],
  iouThreshold: number = 0.45,
): RawDetection[] {
  if (detections.length === 0) return [];

  // Sort by confidence descending
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: RawDetection[] = [];

  for (const det of sorted) {
    let dominated = false;
    for (const keptDet of kept) {
      if (iou(det, keptDet) > iouThreshold) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      kept.push(det);
    }
  }

  return kept;
}

/**
 * Intersection over Union of two bounding boxes.
 *
 * Both boxes are in corner-format: (x, y, w, h) where (x, y) is the
 * top-left corner and (w, h) is width/height.
 *
 * @returns IoU value between 0.0 (no overlap) and 1.0 (perfect overlap)
 */
export function iou(a: RawDetection, b: RawDetection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  const unionArea = areaA + areaB - interArea;

  if (unionArea === 0) return 0;
  return interArea / unionArea;
}
