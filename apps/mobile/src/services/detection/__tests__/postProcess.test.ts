/**
 * Tests for YOLO output post-processing: decodeYoloOutput and nonMaxSuppression.
 *
 * Verifies the transposed tensor access pattern, confidence thresholding,
 * NMS overlap filtering, and IoU calculation edge cases.
 */
import { decodeYoloOutput, nonMaxSuppression, iou } from '../postProcess';
import type { RawDetection } from '../types';

// ---------------------------------------------------------------------------
// Helper: build a Float32Array mimicking YOLO transposed output
// Shape: [1, 4 + numClasses, numPredictions] stored row-major as a flat array
// Access pattern: output[row * numPredictions + col]
// ---------------------------------------------------------------------------
function buildYoloOutput(
  predictions: { cx: number; cy: number; w: number; h: number; classScores: number[] }[],
  numClasses: number,
): Float32Array {
  const numPredictions = predictions.length;
  const numRows = 4 + numClasses;
  const data = new Float32Array(numRows * numPredictions);

  for (let i = 0; i < numPredictions; i++) {
    const p = predictions[i];
    data[0 * numPredictions + i] = p.cx; // row 0: cx
    data[1 * numPredictions + i] = p.cy; // row 1: cy
    data[2 * numPredictions + i] = p.w;  // row 2: w
    data[3 * numPredictions + i] = p.h;  // row 3: h
    for (let c = 0; c < numClasses; c++) {
      data[(4 + c) * numPredictions + i] = p.classScores[c];
    }
  }
  return data;
}

describe('decodeYoloOutput', () => {
  const classNames = ['apple', 'banana'];

  it('decodes transposed YOLO output into bounding boxes with correct coordinates', () => {
    const output = buildYoloOutput(
      [
        { cx: 0.5, cy: 0.5, w: 0.3, h: 0.3, classScores: [0.9, 0.1] },
      ],
      2,
    );

    const results = decodeYoloOutput(output, 2, 1, classNames, 0.25);

    expect(results).toHaveLength(1);
    // cx=0.5, w=0.3 -> x = 0.5 - 0.3/2 = 0.35
    // cy=0.5, h=0.3 -> y = 0.5 - 0.3/2 = 0.35
    expect(results[0].x).toBeCloseTo(0.35);
    expect(results[0].y).toBeCloseTo(0.35);
    expect(results[0].w).toBeCloseTo(0.3);
    expect(results[0].h).toBeCloseTo(0.3);
    expect(results[0].confidence).toBeCloseTo(0.9);
    expect(results[0].classId).toBe(0);
    expect(results[0].className).toBe('apple');
  });

  it('filters detections below confThreshold', () => {
    const output = buildYoloOutput(
      [
        { cx: 0.5, cy: 0.5, w: 0.3, h: 0.3, classScores: [0.1, 0.15] },
      ],
      2,
    );

    const results = decodeYoloOutput(output, 2, 1, classNames, 0.25);
    expect(results).toHaveLength(0);
  });

  it('removes overlapping detections via NMS', () => {
    // Two overlapping predictions + one far away
    const output = buildYoloOutput(
      [
        { cx: 0.5, cy: 0.5, w: 0.3, h: 0.3, classScores: [0.9, 0.1] },   // kept (highest conf)
        { cx: 0.51, cy: 0.51, w: 0.3, h: 0.3, classScores: [0.8, 0.05] }, // removed by NMS
        { cx: 0.2, cy: 0.8, w: 0.1, h: 0.1, classScores: [0.1, 0.6] },   // kept (no overlap)
      ],
      2,
    );

    const results = decodeYoloOutput(output, 2, 3, classNames, 0.25);

    expect(results).toHaveLength(2);
    // First kept: apple with conf 0.9
    expect(results[0].confidence).toBeCloseTo(0.9);
    expect(results[0].className).toBe('apple');
    // Second kept: banana with conf 0.6
    expect(results[1].confidence).toBeCloseTo(0.6);
    expect(results[1].className).toBe('banana');
  });

  it('returns empty array for empty tensor', () => {
    const output = new Float32Array(0);
    const results = decodeYoloOutput(output, 2, 0, classNames);
    expect(results).toEqual([]);
  });

  it('keeps a single detection with high confidence', () => {
    const output = buildYoloOutput(
      [{ cx: 0.4, cy: 0.6, w: 0.2, h: 0.15, classScores: [0.3, 0.85] }],
      2,
    );

    const results = decodeYoloOutput(output, 2, 1, classNames);
    expect(results).toHaveLength(1);
    expect(results[0].className).toBe('banana');
    expect(results[0].classId).toBe(1);
    expect(results[0].confidence).toBeCloseTo(0.85);
    // x = 0.4 - 0.2/2 = 0.3, y = 0.6 - 0.15/2 = 0.525
    expect(results[0].x).toBeCloseTo(0.3);
    expect(results[0].y).toBeCloseTo(0.525);
  });

  it('keeps multiple non-overlapping detections', () => {
    const output = buildYoloOutput(
      [
        { cx: 0.1, cy: 0.1, w: 0.05, h: 0.05, classScores: [0.7, 0.2] },
        { cx: 0.9, cy: 0.9, w: 0.05, h: 0.05, classScores: [0.2, 0.8] },
      ],
      2,
    );

    const results = decodeYoloOutput(output, 2, 2, classNames, 0.25);
    expect(results).toHaveLength(2);
  });

  it('uses default confThreshold of 0.25 when not specified', () => {
    const output = buildYoloOutput(
      [
        { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1, classScores: [0.24, 0.1] }, // below default
        { cx: 0.3, cy: 0.3, w: 0.1, h: 0.1, classScores: [0.26, 0.1] }, // above default
      ],
      2,
    );

    const results = decodeYoloOutput(output, 2, 2, classNames);
    expect(results).toHaveLength(1);
    expect(results[0].confidence).toBeCloseTo(0.26);
  });

  it('falls back to class_N when classNames array is too short', () => {
    const output = buildYoloOutput(
      [{ cx: 0.5, cy: 0.5, w: 0.1, h: 0.1, classScores: [0.1, 0.2, 0.9] }],
      3,
    );

    const results = decodeYoloOutput(output, 3, 1, classNames, 0.25);
    expect(results).toHaveLength(1);
    expect(results[0].className).toBe('class_2');
    expect(results[0].classId).toBe(2);
  });
});

describe('nonMaxSuppression', () => {
  it('keeps the higher-confidence detection when two overlap', () => {
    const detections: RawDetection[] = [
      { x: 0.0, y: 0.0, w: 0.5, h: 0.5, confidence: 0.6, classId: 0, className: 'apple' },
      { x: 0.05, y: 0.05, w: 0.5, h: 0.5, confidence: 0.9, classId: 0, className: 'apple' },
    ];

    const result = nonMaxSuppression(detections, 0.45);
    expect(result).toHaveLength(1);
    expect(result[0].confidence).toBeCloseTo(0.9);
  });

  it('keeps all detections when there is no overlap', () => {
    const detections: RawDetection[] = [
      { x: 0.0, y: 0.0, w: 0.1, h: 0.1, confidence: 0.9, classId: 0, className: 'apple' },
      { x: 0.5, y: 0.5, w: 0.1, h: 0.1, confidence: 0.8, classId: 1, className: 'banana' },
    ];

    const result = nonMaxSuppression(detections, 0.45);
    expect(result).toHaveLength(2);
  });

  it('returns empty array for empty input', () => {
    const result = nonMaxSuppression([], 0.45);
    expect(result).toEqual([]);
  });

  it('returns single detection unchanged', () => {
    const detections: RawDetection[] = [
      { x: 0.1, y: 0.2, w: 0.3, h: 0.4, confidence: 0.75, classId: 0, className: 'apple' },
    ];

    const result = nonMaxSuppression(detections, 0.45);
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual(detections[0]);
  });
});

describe('iou', () => {
  it('returns 1.0 for perfect overlap', () => {
    const a: RawDetection = { x: 0.1, y: 0.1, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'a' };
    const b: RawDetection = { x: 0.1, y: 0.1, w: 0.3, h: 0.3, confidence: 0.8, classId: 0, className: 'a' };

    expect(iou(a, b)).toBeCloseTo(1.0);
  });

  it('returns 0.0 for no overlap', () => {
    const a: RawDetection = { x: 0.0, y: 0.0, w: 0.1, h: 0.1, confidence: 0.9, classId: 0, className: 'a' };
    const b: RawDetection = { x: 0.5, y: 0.5, w: 0.1, h: 0.1, confidence: 0.8, classId: 0, className: 'a' };

    expect(iou(a, b)).toBeCloseTo(0.0);
  });

  it('returns correct IoU for partial overlap', () => {
    // Box A: (0, 0) to (0.4, 0.4) -> area 0.16
    // Box B: (0.2, 0.2) to (0.6, 0.6) -> area 0.16
    // Intersection: (0.2, 0.2) to (0.4, 0.4) -> area 0.04
    // Union: 0.16 + 0.16 - 0.04 = 0.28
    // IoU: 0.04 / 0.28 = ~0.1429
    const a: RawDetection = { x: 0.0, y: 0.0, w: 0.4, h: 0.4, confidence: 0.9, classId: 0, className: 'a' };
    const b: RawDetection = { x: 0.2, y: 0.2, w: 0.4, h: 0.4, confidence: 0.8, classId: 0, className: 'a' };

    expect(iou(a, b)).toBeCloseTo(0.04 / 0.28);
  });

  it('handles touching boxes (no area overlap)', () => {
    const a: RawDetection = { x: 0.0, y: 0.0, w: 0.5, h: 0.5, confidence: 0.9, classId: 0, className: 'a' };
    const b: RawDetection = { x: 0.5, y: 0.0, w: 0.5, h: 0.5, confidence: 0.8, classId: 0, className: 'a' };

    expect(iou(a, b)).toBeCloseTo(0.0);
  });
});
