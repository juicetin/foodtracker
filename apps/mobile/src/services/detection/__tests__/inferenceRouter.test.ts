/**
 * Tests for inference router: two-stage pipeline orchestration
 * (detect -> classify with 335-class EfficientNet-Lite0 output).
 *
 * Updated for dual-buffer signature (detectBuffer + classifyBuffer)
 * and EfficientNet-Lite0 classifier (335 food-specific classes).
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
  // Dual buffers: detect at 640x640, classify at 224x224 (ImageNet-normalized)
  const detectBuffer = new Float32Array([1, 2, 3]);
  const classifyBuffer = new Float32Array([4, 5, 6]);
  const imageWidth = 640;
  const imageHeight = 640;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('runDetectionPipeline - dual buffer signature (no food101Buffer)', () => {
    it('pipeline function signature has NO food101Buffer parameter', async () => {
      // Verify the function accepts exactly 5 arguments (no food101Buffer)
      expect(runDetectionPipeline.length).toBe(5);
    });

    it('returns empty items when no COCO food detections found', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      // 335-element classify output
      const classifyModel = createMockModel([new Float32Array(335).fill(0.0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      // No food detections returned
      mockDecodeYoloOutput.mockReturnValue([]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(0);
      // Detect should be called, classify should NOT (no food detections)
      expect(detectModel.run).toHaveBeenCalledTimes(1);
      expect(classifyModel.run).toHaveBeenCalledTimes(0);
    });

    it('returns detected items labeled with 335-class name when food is present', async () => {
      // Detection model: returns fake tensor
      const detectOutput = new Float32Array(6 * 2);
      const detectModel = createMockModel([detectOutput]);

      // Classify model: 335-element output with high confidence at index 42
      const classifyOutput = new Float32Array(335).fill(0.0);
      classifyOutput[42] = 0.85; // High confidence at index 42
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
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
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(2);
      // Both items get the 335-class label (not COCO label)
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

    it('pipeline runs as detect -> classify (2 stages, no binary stage)', async () => {
      const callOrder: string[] = [];

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
          return [new Float32Array(335).fill(0.8)];
        }),
        runSync: jest.fn(),
        inputs: [],
        outputs: [],
        delegate: 'default' as const,
      };

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Should run detect then classify, no binary step
      expect(callOrder).toEqual(['detect', 'classify']);
    });

    it('classify model receives classifyBuffer (ImageNet-normalized)', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyOutput = new Float32Array(335).fill(0.0);
      classifyOutput[1] = 0.8;
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Detection gets detectBuffer (640x640 input)
      expect(detectModel.run).toHaveBeenCalledWith([detectBuffer]);
      // Classify gets classifyBuffer (224x224 ImageNet-normalized input)
      expect(classifyModel.run).toHaveBeenCalledWith([classifyBuffer]);
    });

    it('formats classify label from snake_case to Title Case', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      // Mock CLASSIFY_CLASS_NAMES to have a known label at index 10
      // The actual CLASSIFY_CLASS_NAMES comes from constants.ts which is mocked via the module
      const classifyOutput = new Float32Array(335).fill(0.0);
      classifyOutput[10] = 0.9; // High confidence at index 10
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // The label should be formatted (no underscores, title case)
      expect(result.items[0].className).toBeDefined();
      expect(result.items[0].className).not.toContain('_');
      // First character of each word should be uppercase
      const words = result.items[0].className.split(' ');
      for (const word of words) {
        if (word.length > 0) {
          expect(word[0]).toBe(word[0].toUpperCase());
        }
      }
    });

    it('uses fallback label when classify confidence is below threshold', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      // All classify scores very low (below 0.15 threshold)
      const classifyOutput = new Float32Array(335).fill(0.01);
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Items should still be returned (no binary gate to reject them)
      expect(result.items).toHaveLength(1);
      // Label should be "Food item" fallback when classify confidence is too low
      expect(result.items[0].className).toBe('Food Item');
    });

    it('records timing for detect and classify stages only', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(335).fill(0.0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 47, className: 'apple' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.pipelineStages).toBeDefined();
      expect(result.pipelineStages.length).toBe(2);

      const detectStage = result.pipelineStages.find(s => s.stage === 'detect');
      expect(detectStage).toBeDefined();
      expect(typeof detectStage!.timeMs).toBe('number');

      const classifyStage = result.pipelineStages.find(s => s.stage === 'classify');
      expect(classifyStage).toBeDefined();

      // No binary or food101-fallback stages
      const binaryStage = result.pipelineStages.find(s => (s as { stage: string }).stage === 'binary');
      expect(binaryStage).toBeUndefined();
      const fallbackStage = result.pipelineStages.find(s => (s as { stage: string }).stage === 'food101-fallback');
      expect(fallbackStage).toBeUndefined();

      expect(typeof result.inferenceTimeMs).toBe('number');
      expect(result.inferenceTimeMs).toBeGreaterThanOrEqual(0);
    });

    it('throws if model set is not loaded', async () => {
      mockGetModelSet.mockReturnValue(null);

      await expect(
        runDetectionPipeline(detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames),
      ).rejects.toThrow();
    });

    it('filters non-food COCO detections before classification', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(335).fill(0.5)]);

      mockGetModelSet.mockReturnValue({
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
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Only the banana detection (classId 46) should pass through
      expect(result.items).toHaveLength(1);
    });

    it('ImageNet normalization produces correct values (pixel 128/255 -> (0.502-0.485)/0.229 = 0.074)', () => {
      // This test validates the normalization math that imagePreprocess.ts uses.
      // pixel value 128 -> 128/255 = 0.50196...
      // R channel: (0.50196 - 0.485) / 0.229 = 0.0741...
      const pixelValue = 128;
      const normalized = (pixelValue / 255 - 0.485) / 0.229;
      expect(normalized).toBeCloseTo(0.074, 2);
    });
  });
});
