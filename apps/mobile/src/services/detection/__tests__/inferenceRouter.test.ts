/**
 * Tests for inference router: three-stage pipeline orchestration
 * (binary gate -> detection -> classification).
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
  const imageBuffer = new Float32Array([1, 2, 3]).buffer;
  const imageWidth = 640;
  const imageHeight = 640;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('runDetectionPipeline', () => {
    it('returns empty items when binary gate says not food', async () => {
      // Binary gate output: single value below 0.5 = not food
      const binaryModel = createMockModel([new Float32Array([0.2])]);
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      const result = await runDetectionPipeline(imageBuffer, imageWidth, imageHeight, classNames);

      expect(result.items).toHaveLength(0);
      // Binary model was called
      expect(binaryModel.run).toHaveBeenCalledTimes(1);
      // Detect and classify should NOT be called
      expect(detectModel.run).toHaveBeenCalledTimes(0);
      expect(classifyModel.run).toHaveBeenCalledTimes(0);
    });

    it('returns detected items when food is present', async () => {
      // Binary gate: food detected (>0.5)
      const binaryModel = createMockModel([new Float32Array([0.95])]);
      // Detection model: returns fake tensor (will be decoded by mocked postProcess)
      const detectOutput = new Float32Array(6 * 2); // 6 rows (4 bbox + 2 classes) x 2 predictions
      const detectModel = createMockModel([detectOutput]);
      // Classify model: returns class scores for each detection
      const classifyModel = createMockModel([new Float32Array([0.85, 0.1, 0.05])]);

      mockGetModelSet.mockReturnValue({
        binary: binaryModel,
        detect: detectModel,
        classify: classifyModel,
      });

      // Mock postProcess to return 2 raw detections
      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'apple' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 1, className: 'banana' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runDetectionPipeline(imageBuffer, imageWidth, imageHeight, classNames);

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

    it('records timing for each stage', async () => {
      const binaryModel = createMockModel([new Float32Array([0.9])]);
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

      const result = await runDetectionPipeline(imageBuffer, imageWidth, imageHeight, classNames);

      expect(result.pipelineStages).toBeDefined();
      expect(result.pipelineStages.length).toBeGreaterThanOrEqual(2);

      // Binary stage should always be present
      const binaryStage = result.pipelineStages.find(s => s.stage === 'binary');
      expect(binaryStage).toBeDefined();
      expect(typeof binaryStage!.timeMs).toBe('number');
      expect(binaryStage!.timeMs).toBeGreaterThanOrEqual(0);

      // Detect stage should be present when food detected
      const detectStage = result.pipelineStages.find(s => s.stage === 'detect');
      expect(detectStage).toBeDefined();

      // Total inference time should be a number
      expect(typeof result.inferenceTimeMs).toBe('number');
      expect(result.inferenceTimeMs).toBeGreaterThanOrEqual(0);
    });

    it('runs pipeline sequentially: binary -> detect -> classify', async () => {
      const callOrder: string[] = [];

      const binaryModel = {
        run: jest.fn().mockImplementation(async () => {
          callOrder.push('binary');
          return [new Float32Array([0.9])];
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

      await runDetectionPipeline(imageBuffer, imageWidth, imageHeight, classNames);

      // Verify sequential order
      expect(callOrder[0]).toBe('binary');
      expect(callOrder[1]).toBe('detect');
      // classify may or may not be called depending on implementation
      // but binary must be before detect
      expect(callOrder.indexOf('binary')).toBeLessThan(callOrder.indexOf('detect'));
    });

    it('throws if model set is not loaded', async () => {
      mockGetModelSet.mockReturnValue(null);

      await expect(
        runDetectionPipeline(imageBuffer, imageWidth, imageHeight, classNames),
      ).rejects.toThrow();
    });
  });
});
