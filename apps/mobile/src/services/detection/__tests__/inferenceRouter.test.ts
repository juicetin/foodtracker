/**
 * Tests for inference router: three-stage pipeline orchestration
 * (binary gate -> detection -> classification -> Food-101 fallback).
 *
 * Updated for triple-buffer signature (detectBuffer + classifyBuffer + food101Buffer)
 * and AIY Food V1 binary gate (max over 2024 class probabilities).
 */

// ── Mock modelLoader ──
const mockGetModelSet = jest.fn();
jest.mock('../modelLoader', () => ({
  getModelSet: () => mockGetModelSet(),
}));

// ── Mock postProcess ──
const mockDecodeYoloOutput = jest.fn();
jest.mock('../postProcess', () => ({
  decodeYoloOutput: (...args: unknown[]) => mockDecodeYoloOutput(...args),
}));

import { runDetectionPipeline } from '../inferenceRouter';
import type { RawDetection } from '../types';

// Helper: build a mock model with controllable run() output
function createMockModel(output: Float32Array[]) {
  return {
    run: jest.fn().mockResolvedValue(output),
    runSync: jest.fn().mockReturnValue(output),
    inputs: [],
    outputs: [],
    delegate: 'default' as const,
  };
}

describe('inferenceRouter', () => {
  const classNames = ['apple', 'banana', 'rice'];
  // Triple buffers: detect at 640x640, classify at 192x192, food101 at 224x224
  const detectBuffer = new Float32Array([1, 2, 3]);
  const classifyBuffer = new Float32Array([4, 5, 6]);
  const food101Buffer = new Float32Array([7, 8, 9]);
  const imageWidth = 640;
  const imageHeight = 640;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('runDetectionPipeline - triple buffer signature', () => {
    it('returns empty items when binary gate says not food', async () => {
      // Binary gate output: 2024-element array with all values below 0.5
      const binaryOutput = new Float32Array(2024).fill(0.2);
      const binaryModel = createMockModel([binaryOutput]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(0);
      // Binary model was called with classifyBuffer (192x192)
      expect(binaryModel.run).toHaveBeenCalledTimes(1);
      expect(binaryModel.run).toHaveBeenCalledWith([classifyBuffer]);
      // Detect and classify should NOT be called (short-circuited)
      expect(detectModel.run).toHaveBeenCalledTimes(0);
      expect(classifyModel.run).toHaveBeenCalledTimes(0);
    });

    it('returns detected items when food is present', async () => {
      // Binary gate: AIY-like output with 2024 values, high confidence food
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[500] = 0.95; // High confidence at class index 500
      const binaryModel = createMockModel([binaryOutput]);

      // Detection model: returns fake tensor
      const detectOutput = new Float32Array(6 * 2);
      const detectModel = createMockModel([detectOutput]);
      // Classify model: high confidence so Food-101 fallback is NOT triggered
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.85; // High confidence at index 1 (a valid food class)
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      // Use COCO food class IDs (46=banana, 47=apple) to pass the food filter
      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 47, className: 'apple' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 46, className: 'banana' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(2);
      // Both items get the AIY V1 label (not COCO label)
      expect(result.items[0].confidence).toBe(0.9);
      expect(result.items[0].bbox).toEqual({ x: 0.1, y: 0.2, w: 0.3, h: 0.3 });
      expect(result.items[1].confidence).toBe(0.7);
      // Each item should have an id
      expect(result.items[0].id).toBeDefined();
      expect(result.items[1].id).toBeDefined();
      expect(result.items[0].id).not.toBe(result.items[1].id);
      // Portion estimates should be placeholder
      expect(result.items[0].portionEstimate.method).toBe('pending');
      expect(result.items[0].portionMultiplier).toBe(1);
      expect(result.items[0].isRemoved).toBe(false);
    });

    it('passes detectBuffer to detect stage and classifyBuffer to binary/classify', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.9;
      const binaryModel = createMockModel([binaryOutput]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.8;
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Binary gate gets classifyBuffer (192x192 input)
      expect(binaryModel.run).toHaveBeenCalledWith([classifyBuffer]);
      // Detection gets detectBuffer (640x640 input)
      expect(detectModel.run).toHaveBeenCalledWith([detectBuffer]);
      // Classify gets classifyBuffer (192x192 input)
      expect(classifyModel.run).toHaveBeenCalledWith([classifyBuffer]);
    });

    it('records timing for each stage', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[0] = 0.9;
      const binaryModel = createMockModel([binaryOutput]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.pipelineStages).toBeDefined();
      expect(result.pipelineStages.length).toBeGreaterThanOrEqual(2);

      const binaryStage = result.pipelineStages.find(s => s.stage === 'binary');
      expect(binaryStage).toBeDefined();
      expect(typeof binaryStage!.timeMs).toBe('number');
      expect(binaryStage!.timeMs).toBeGreaterThanOrEqual(0);

      const detectStage = result.pipelineStages.find(s => s.stage === 'detect');
      expect(detectStage).toBeDefined();

      expect(typeof result.inferenceTimeMs).toBe('number');
      expect(result.inferenceTimeMs).toBeGreaterThanOrEqual(0);
    });

    it('runs pipeline sequentially: binary -> detect -> classify', async () => {
      const callOrder: string[] = [];

      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[0] = 0.9;

      const binaryModel = {
        run: jest.fn().mockImplementation(async () => {
          callOrder.push('binary');
          return [binaryOutput];
        }),
        runSync: jest.fn(),
        inputs: [],
        outputs: [],
        delegate: 'default' as const,
      };
      const detectModel = {
        run: jest.fn().mockImplementation(async () => {
          callOrder.push('detect');
          return [new Float32Array(0)];
        }),
        runSync: jest.fn(),
        inputs: [],
        outputs: [],
        delegate: 'default' as const,
      };
      const classifyModel = {
        run: jest.fn().mockImplementation(async () => {
          callOrder.push('classify');
          return [new Float32Array(2024).fill(0.8)];
        }),
        runSync: jest.fn(),
        inputs: [],
        outputs: [],
        delegate: 'default' as const,
      };

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(callOrder[0]).toBe('binary');
      expect(callOrder[1]).toBe('detect');
      expect(callOrder.indexOf('binary')).toBeLessThan(callOrder.indexOf('detect'));
    });

    it('throws if model set is not loaded', async () => {
      mockGetModelSet.mockReturnValue(null);

      await expect(
        runDetectionPipeline(detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames),
      ).rejects.toThrow();
    });
  });

  describe('binary gate - AIY Food V1 max-over-classes', () => {
    it('takes MAX of all 2024 class scores (not just index 0)', async () => {
      // All scores are low EXCEPT one at a non-zero index
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[500] = 0.85; // High confidence at index 500 (some food class)
      const binaryModel = createMockModel([binaryOutput]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Binary gate should pass (max=0.85 > 0.5), so detect stage runs
      expect(detectModel.run).toHaveBeenCalledTimes(1);
    });

    it('returns items=[] when all 2024 class scores are below threshold', async () => {
      // All 2024 values below 0.5
      const binaryOutput = new Float32Array(2024).fill(0.3);
      const binaryModel = createMockModel([binaryOutput]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(0);
      // Should NOT call detect or classify stages
      expect(detectModel.run).toHaveBeenCalledTimes(0);
      expect(classifyModel.run).toHaveBeenCalledTimes(0);
    });

    it('handles 2024-element Float32Array correctly for binary output', async () => {
      // Verify no stack overflow or issues with large array
      const binaryOutput = new Float32Array(2024);
      // Set a single high value at the last index
      binaryOutput[2023] = 0.92;
      const binaryModel = createMockModel([binaryOutput]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Binary gate passes (max=0.92 at index 2023 > 0.5)
      expect(detectModel.run).toHaveBeenCalledTimes(1);
    });
  });

  describe('Food-101 fallback classifier', () => {
    it('triggers Food-101 fallback when AIY V1 confidence < 60%', async () => {
      // Binary gate passes
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);

      // AIY V1 classify: low confidence (below 0.6 threshold)
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.3; // 30% confidence -- triggers fallback
      const classifyModel = createMockModel([classifyOutput]);

      // Food-101: high confidence for "ramen" (index 81 in the 101-class list)
      const food101Output = new Float32Array(101).fill(0.0);
      food101Output[81] = 0.75; // "ramen" at index 81
      const food101Model = createMockModel([food101Output]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
        food101: food101Model,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Food-101 model should have been called with food101Buffer
      expect(food101Model.run).toHaveBeenCalledTimes(1);
      expect(food101Model.run).toHaveBeenCalledWith([food101Buffer]);

      // Label should be Food-101's "Ramen" (formatted from "ramen")
      expect(result.items).toHaveLength(1);
      expect(result.items[0].className).toBe('Ramen');

      // Should have food101-fallback stage in timing
      const fallbackStage = result.pipelineStages.find(s => s.stage === 'food101-fallback');
      expect(fallbackStage).toBeDefined();
    });

    it('does NOT trigger fallback when AIY V1 confidence >= 60%', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);

      // AIY V1 classify: high confidence (above 0.6 threshold)
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.75; // 75% confidence -- no fallback
      const classifyModel = createMockModel([classifyOutput]);

      const food101Model = createMockModel([new Float32Array(101).fill(0.0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
        food101: food101Model,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Food-101 model should NOT have been called
      expect(food101Model.run).toHaveBeenCalledTimes(0);

      // Should NOT have food101-fallback stage
      const fallbackStage = result.pipelineStages.find(s => s.stage === 'food101-fallback');
      expect(fallbackStage).toBeUndefined();
    });

    it('keeps AIY V1 label when Food-101 confidence is lower', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);

      // AIY V1: low confidence
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.4; // 40% -- triggers fallback
      const classifyModel = createMockModel([classifyOutput]);

      // Food-101: even lower confidence
      const food101Output = new Float32Array(101).fill(0.0);
      food101Output[0] = 0.2; // 20% -- lower than AIY V1
      const food101Model = createMockModel([food101Output]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
        food101: food101Model,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Food-101 was called but its label was NOT used (lower confidence)
      expect(food101Model.run).toHaveBeenCalledTimes(1);
      // Should still have AIY V1's label (from index 1 of FOOD_V1_CLASS_NAMES)
      expect(result.items[0].className).not.toBe('Apple Pie'); // index 0 of Food-101
    });

    it('formats Food-101 labels with title case (underscores to spaces)', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);

      // AIY V1: low confidence to trigger fallback
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.2;
      const classifyModel = createMockModel([classifyOutput]);

      // Food-101: high confidence for "pad_thai" (index 70)
      const food101Output = new Float32Array(101).fill(0.0);
      food101Output[70] = 0.85; // "pad_thai"
      const food101Model = createMockModel([food101Output]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
        food101: food101Model,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items[0].className).toBe('Pad Thai');
    });

    it('gracefully handles Food-101 model failure', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);

      // AIY V1: low confidence to trigger fallback
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.3;
      const classifyModel = createMockModel([classifyOutput]);

      // Food-101: throws an error
      const food101Model = {
        run: jest.fn().mockRejectedValue(new Error('Model inference failed')),
        runSync: jest.fn(),
      };

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
        food101: food101Model,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      // Should NOT throw -- falls back to AIY V1 label
      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(1);
      // Label comes from AIY V1, not Food-101
      expect(food101Model.run).toHaveBeenCalledTimes(1);
    });

    it('works when Food-101 model is not loaded (undefined)', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);

      // AIY V1: low confidence to trigger fallback
      const classifyOutput = new Float32Array(2024).fill(0.0);
      classifyOutput[1] = 0.3;
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
        // food101 is undefined
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      // Should NOT throw or crash -- just uses AIY V1 label
      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(1);
      // No food101-fallback stage
      const fallbackStage = result.pipelineStages.find(s => s.stage === 'food101-fallback');
      expect(fallbackStage).toBeUndefined();
    });

    it('filters non-food COCO detections before classification', async () => {
      const binaryOutput = new Float32Array(2024).fill(0.01);
      binaryOutput[100] = 0.8;
      const binaryModel = createMockModel([binaryOutput]);

      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(2024).fill(0.0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      // Include non-food COCO classes (person=0, car=2) alongside food (banana=46)
      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.95, classId: 0, className: 'person' },
        { x: 0.4, y: 0.5, w: 0.2, h: 0.2, confidence: 0.8, classId: 2, className: 'car' },
        { x: 0.6, y: 0.7, w: 0.15, h: 0.15, confidence: 0.7, classId: 46, className: 'banana' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, food101Buffer, imageWidth, imageHeight, classNames,
      );

      // Only the banana detection (classId 46) should pass through
      expect(result.items).toHaveLength(1);
    });
  });
});
