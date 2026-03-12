/**
 * Tests for inference router: three-stage pipeline orchestration
 * (binary gate -> detection -> classification).
 *
 * Updated for dual-buffer signature (detectBuffer + classifyBuffer)
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
  // Dual buffers: detect at 640x640, classify at 192x192
  const detectBuffer = new Float32Array([1, 2, 3]).buffer;
  const classifyBuffer = new Float32Array([4, 5, 6]).buffer;
  const imageWidth = 640;
  const imageHeight = 640;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('runDetectionPipeline - dual buffer signature', () => {
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
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
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
      // Classify model
      const classifyModel = createMockModel([new Float32Array([0.85, 0.1, 0.05])]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'apple' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 1, className: 'banana' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(2);
      expect(result.items[0].className).toBe('apple');
      expect(result.items[0].confidence).toBe(0.9);
      expect(result.items[0].bbox).toEqual({ x: 0.1, y: 0.2, w: 0.3, h: 0.3 });
      expect(result.items[1].className).toBe('banana');
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
      const classifyModel = createMockModel([new Float32Array([0.8])]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 0, className: 'apple' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
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
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 0, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
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
          return [new Float32Array([0.8, 0.1, 0.1])];
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
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 0, className: 'apple' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(callOrder[0]).toBe('binary');
      expect(callOrder[1]).toBe('detect');
      expect(callOrder.indexOf('binary')).toBeLessThan(callOrder.indexOf('detect'));
    });

    it('throws if model set is not loaded', async () => {
      mockGetModelSet.mockReturnValue(null);

      await expect(
        runDetectionPipeline(detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames),
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
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
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
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
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
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Binary gate passes (max=0.92 at index 2023 > 0.5)
      expect(detectModel.run).toHaveBeenCalledTimes(1);
    });
  });
});
