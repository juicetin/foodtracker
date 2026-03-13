/**
 * Detection pipeline constants.
 *
 * COCO 80 class names at correct indices, food-specific class IDs,
 * EfficientNet-Lite0 class names (335 entries), and model input sizes
 * for the two-stage pipeline (detect + classify).
 *
 * EfficientNet-Lite0 input: 224x224 float32 with ImageNet normalization.
 * YOLO11n COCO input: 640x640 float32.
 */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const classifyLabels = require('../../../assets/models/labels_classify.json');

/**
 * Full 80-class COCO names array (indices 0-79).
 * Source: Ultralytics COCO dataset docs.
 */
export const COCO_CLASS_NAMES: string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
  'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
  'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
  'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
  'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

/**
 * COCO class IDs that represent food items (0-indexed).
 * banana=46, apple=47, sandwich=48, orange=49, broccoli=50,
 * carrot=51, hot dog=52, pizza=53, donut=54, cake=55.
 */
export const COCO_FOOD_CLASS_IDS: Set<number> = new Set([
  46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
]);

/**
 * Input size for the detection model (YOLO11n COCO).
 */
export const DETECT_INPUT_SIZE = 640;

/**
 * Input size for the classification model (EfficientNet-Lite0).
 * 224x224 float32 with ImageNet normalization.
 */
export const CLASSIFY_INPUT_SIZE = 224;

/**
 * EfficientNet-Lite0 class names (335 food-specific entries).
 * Index-aligned with model output logits.
 * Trained on Food-101 + UEC-256 merged dataset.
 */
export const CLASSIFY_CLASS_NAMES: string[] = classifyLabels.labels;

/** ImageNet normalization mean (RGB channels). */
export const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
/** ImageNet normalization std (RGB channels). */
export const IMAGENET_STD: [number, number, number] = [0.229, 0.224, 0.225];
