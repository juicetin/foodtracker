/**
 * Tests for inference router: single-stage bbox-only pipeline.
 *
 * YOLO detects bounding boxes with generic "Food Region" labels.
 * VLM identifies food items post-detection (separate pipeline).
 * Classification stage has been removed (Phase 7).
 */

// -- Mock modelLoader --
const mockGetModelSet = jest.fn();
jest.mock('../modelLoader', () => ({
  getModelSet: () => mockGetModelSet(),
}));

// -- Mock postProcess --
const mockDecodeYoloOutput = jest.fn();
jest.mock('../postProcess', () => ({
  decodeYoloOutput: (...args: unknown[]) => mockDecodeYoloOutput(...args),
}));

import { runBboxDetection, formatFoodLabel } from '../inferenceRouter';
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
  // Single buffer: detect at 640x640 (no classify buffer)
  const detectBuffer = new Float32Array([1, 2, 3]);
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
      expect(formatFoodLabel('BBQ_sauce')).toBe('BBQ Sauce');
    });
  });

  describe('runBboxDetection - bbox-only pipeline', () => {
    it('function has 4 arguments (no classifyBuffer)', () => {
      expect(runBboxDetection.length).toBe(4);
    });

    it('returns empty items when no detections found', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      mockDecodeYoloOutput.mockReturnValue([]);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(0);
      expect(detectModel.run).toHaveBeenCalledTimes(1);
    });

    it('all YOLO detections pass through without food-class filtering', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'achichuk' },
        { x: 0.4, y: 0.5, w: 0.2, h: 0.2, confidence: 0.8, classId: 100, className: 'mango' },
        { x: 0.6, y: 0.7, w: 0.15, h: 0.15, confidence: 0.7, classId: 200, className: 'vegetable_soup' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(3);
    });

    it('returns items with className="Food Region" (not YOLO label)', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      const rawDetections: RawDetection[] = [
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 0, className: 'achichuk' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 100, className: 'mango' },
      ];
      mockDecodeYoloOutput.mockReturnValue(rawDetections);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(2);
      // All items should have generic "Food Region" label, not the YOLO class name
      expect(result.items[0].className).toBe('Food Region');
      expect(result.items[1].className).toBe('Food Region');
    });

    it('returns items with isRefining=true (shimmer state until VLM identifies)', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 10, className: 'rice' },
      ]);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.items).toHaveLength(1);
      expect(result.items[0].isRefining).toBe(true);
    });

    it('pipelineStages has only detect (no classify)', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.8, classId: 10, className: 'rice' },
      ]);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      expect(result.pipelineStages).toBeDefined();
      expect(result.pipelineStages.length).toBe(1);

      const detectStage = result.pipelineStages.find(s => s.stage === 'detect');
      expect(detectStage).toBeDefined();
      expect(typeof detectStage!.timeMs).toBe('number');

      // No classify stage
      const classifyStage = result.pipelineStages.find(s => (s as { stage: string }).stage === 'classify');
      expect(classifyStage).toBeUndefined();

      expect(typeof result.inferenceTimeMs).toBe('number');
      expect(result.inferenceTimeMs).toBeGreaterThanOrEqual(0);
    });

    it('classify model is never called', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 10, className: 'rice' },
      ]);

      await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      // Only detect model should be called
      expect(detectModel.run).toHaveBeenCalledTimes(1);
      expect(detectModel.run).toHaveBeenCalledWith([detectBuffer]);
    });

    it('pipeline correctly handles 241-class YOLO output with stride 245', async () => {
      const stride = 4 + 241;
      const numPredictions = 2;
      const detectOutput = new Float32Array(stride * numPredictions);
      const detectModel = createMockModel([detectOutput]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 10, className: 'rice' },
      ]);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
      );

      expect(mockDecodeYoloOutput).toHaveBeenCalledWith(
        expect.any(Float32Array),
        241,
        expect.any(Number),
        classNames,
      );
      expect(result.items).toHaveLength(1);
      expect(result.items[0].className).toBe('Food Region');
    });

    it('throws if model set is not loaded', async () => {
      mockGetModelSet.mockReturnValue(null);

      await expect(
        runBboxDetection(detectBuffer, imageWidth, imageHeight, classNames),
      ).rejects.toThrow();
    });

    it('preserves portion estimate and metadata on each item', async () => {
      const detectModel = createMockModel([new Float32Array(0)]);

      mockGetModelSet.mockReturnValue({
        detect: detectModel,
      });

      mockDecodeYoloOutput.mockReturnValue([
        { x: 0.1, y: 0.2, w: 0.3, h: 0.3, confidence: 0.9, classId: 10, className: 'curry' },
        { x: 0.5, y: 0.6, w: 0.2, h: 0.2, confidence: 0.7, classId: 20, className: 'naan' },
      ]);

      const result = await runBboxDetection(
        detectBuffer, imageWidth, imageHeight, classNames,
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
      // isRefining should be true on all items
      expect(result.items[0].isRefining).toBe(true);
      expect(result.items[1].isRefining).toBe(true);
    });
  });
});
