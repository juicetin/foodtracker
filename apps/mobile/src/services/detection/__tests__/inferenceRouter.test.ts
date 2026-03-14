/**
 * Tests for inference router: two-stage pipeline orchestration
 * (detect -> classify with 241-class GGCD YOLO + 905-class EfficientNet-Lite0).
 *
 * Updated for 241 GGCD food-specific detection classes.
 * All YOLO detections are food -- no COCO food-class filtering.
 * Classifier uses 905-class merged_v2 model (14-dataset global cuisine merge).
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

import { runDetectionPipeline, formatFoodLabel } from '../inferenceRouter';
import { CLASSIFY_CLASS_NAMES } from '../constants';
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
  // 241-element GGCD class names array for tests
  const classNames = Array.from({ length: 241 }, (_, i) => `food_class_${i}`);
  // Dual buffers: detect at 640x640, classify at 224x224 (ImageNet-normalized)
  const detectBuffer = new Float32Array([1, 2, 3]);
  const classifyBuffer = new Float32Array([4, 5, 6]);
  const imageWidth = 640;
  const imageHeight = 640;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('formatFoodLabel', () => {
    it('handles hyphens in GGCD names', () => {
      expect(formatFoodLabel('airan-katyk')).toBe('Airan Katyk');
    });

    it('handles underscores in class names', () => {
      expect(formatFoodLabel('pad_thai')).toBe('Pad Thai');
    });

    it('handles spaces in class names', () => {
      expect(formatFoodLabel('vegetable based cooked food')).toBe('Vegetable Based Cooked Food');
    });

    it('handles mixed separators', () => {
      expect(formatFoodLabel('grilled-cheese_sandwich')).toBe('Grilled Cheese Sandwich');
    });

    it('handles single word', () => {
      expect(formatFoodLabel('rice')).toBe('Rice');
    });

    it('preserves already-capitalized words', () => {
      // First char becomes uppercase, rest stays as-is
      expect(formatFoodLabel('BBQ_sauce')).toBe('BBQ Sauce');
    });
  });

  describe('CLASSIFY_CLASS_NAMES from labels_classify.json', () => {
    it('has 905 class entries from merged_v2 training', () => {
      expect(CLASSIFY_CLASS_NAMES.length).toBe(905);
    });

    it('contains global cuisine food names', () => {
      // Spot-check representative classes from different cuisine regions
      expect(CLASSIFY_CLASS_NAMES).toContain('biryani');       // Indian
      expect(CLASSIFY_CLASS_NAMES).toContain('pad_thai');       // Thai
      expect(CLASSIFY_CLASS_NAMES).toContain('pho');            // Vietnamese
      expect(CLASSIFY_CLASS_NAMES).toContain('dosa');           // South Indian
      expect(CLASSIFY_CLASS_NAMES).toContain('doro_wat');       // Ethiopian
      expect(CLASSIFY_CLASS_NAMES).toContain('rendang');        // Indonesian
    });
  });

  describe('runDetectionPipeline - 241-class GGCD detection', () => {
    it('pipeline function signature has 5 arguments', async () => {
      expect(runDetectionPipeline.length).toBe(5);
    });

    it('returns empty items when no detections found', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(0);
      // Detect should be called, classify should NOT (no detections)
      expect(detectModel.run).toHaveBeenCalledTimes(1);
      expect(classifyModel.run).toHaveBeenCalledTimes(0);
    });

    it('all YOLO detections pass through without food-class filtering', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.5)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      // 3 detections with different class IDs -- ALL should pass through (no COCO filtering)
      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'achichuk' },
        { x: 0.4, y: 0.5, w: 0.2, h: 0.2, confidence: 0.8, classId: 100, className: 'mango' },
        { x: 0.6, y: 0.7, w: 0.15, h: 0.15, confidence: 0.7, classId: 200, className: 'vegetable_soup' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // All 3 detections pass -- no filtering
      expect(result.items).toHaveLength(3);
    });

    it('each detection uses its own YOLO class name', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.5)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      // 2 detections with different food classes
      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'achichuk' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 100, className: 'mango' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(2);
      // Each item should have its OWN YOLO class name, not a shared classifier label
      expect(result.items[0].className).toBe('Achichuk');
      expect(result.items[1].className).toBe('Mango');
    });

    it('pipeline correctly handles 241-class YOLO output with stride 245', async () => {
      // Stride = 4 + 241 = 245
      const stride = 4 + 241;
      const numPredictions = 2;
      const detectOutput = new Float32Array(stride * numPredictions);
      const detectModel = createMockModel([detectOutput]);
      const classifyModel = createMockModel([new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.5)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 10, className: 'rice' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // decodeYoloOutput should be called with 241 classes
      expect(mockDecodeYoloOutput).toHaveBeenCalledWith(
        expect.any(Float32Array),
        241,
        expect.any(Number),
        classNames,
      );
      expect(result.items).toHaveLength(1);
      expect(result.items[0].className).toBe('Rice');
    });

    it('pipeline runs as detect -> classify (2 stages)', async () => {
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
          return [new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.8)];
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
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 10, className: 'rice' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(callOrder).toEqual(['detect', 'classify']);
    });

    it('classify model receives classifyBuffer (ImageNet-normalized)', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyOutput = new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.0);
      classifyOutput[1] = 0.8;
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 10, className: 'rice' },
      ]);

      await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Detection gets detectBuffer
      expect(detectModel.run).toHaveBeenCalledWith([detectBuffer]);
      // Classify gets classifyBuffer
      expect(classifyModel.run).toHaveBeenCalledWith([classifyBuffer]);
    });

    it('uses YOLO label even when classify confidence is below threshold', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      // All classify scores very low (below 0.15 threshold)
      const classifyOutput = new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.01);
      const classifyModel = createMockModel([classifyOutput]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 10, className: 'rice' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      // Items should use their YOLO label, not "Food Item" fallback
      expect(result.items).toHaveLength(1);
      expect(result.items[0].className).toBe('Rice');
    });

    it('records timing for detect and classify stages only', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 10, className: 'rice' },
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

      expect(typeof result.inferenceTimeMs).toBe('number');
      expect(result.inferenceTimeMs).toBeGreaterThanOrEqual(0);
    });

    it('throws if model set is not loaded', async () => {
      mockGetModelSet.mockReturnValue(null);

      await expect(
        runDetectionPipeline(detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames),
      ).rejects.toThrow();
    });

    it('preserves portion estimate and metadata on each item', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);
      const classifyModel = createMockModel([new Float32Array(CLASSIFY_CLASS_NAMES.length).fill(0.5)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
        classify: classifyModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 10, className: 'curry' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 20, className: 'naan' },
      ]);

      const result = await runDetectionPipeline(
        detectBuffer, classifyBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(2);
      // Each item has unique ID
      expect(result.items[0].id).toBeDefined();
      expect(result.items[1].id).toBeDefined();
      expect(result.items[0].id).not.toBe(result.items[1].id);
      // Confidence from YOLO detection
      expect(result.items[0].confidence).toBe(0.9);
      expect(result.items[1].confidence).toBe(0.7);
      // Bbox preserved
      expect(result.items[0].bbox).toEqual({ x: 0.1, y: 0.2, w: 0.3, h: 0.3 });
      // Portion estimates placeholder
      expect(result.items[0].portionEstimate.method).toBe('pending');
      expect(result.items[0].portionMultiplier).toBe(1);
      expect(result.items[0].isRemoved).toBe(false);
    });

    it('ImageNet normalization produces correct values (pixel 128/255 -> (0.502-0.485)/0.229 = 0.074)', () => {
      const pixelValue = 128;
      const normalized = (pixelValue / 255 - 0.485) / 0.229;
      expect(normalized).toBeCloseTo(0.074, 2);
    });
  });
});
