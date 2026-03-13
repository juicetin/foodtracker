/**
 * Detection pipeline constants.
 *
 * COCO 80 class names at correct indices, food-specific class IDs,
 * AIY Food V1 class names (2024 entries), and model input sizes
 * for the three-stage pipeline.
 *
 * AIY Food V1 actual input: 192x192 uint8 (discovered in Plan 01).
 * YOLO11n COCO input: 640x640 float32.
 */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const foodV1Labels = require('../../../assets/models/labels_food_v1.json');

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
 * Input size for the binary gate and classify models (AIY Food V1).
 * Actual model requires 192x192 (not 224x224 as originally planned).
 */
export const BINARY_INPUT_SIZE = 192;

/**
 * Input size for the detection model (YOLO11n COCO).
 */
export const DETECT_INPUT_SIZE = 640;

/**
 * Input size for the classification model (AIY Food V1).
 * Same model as binary gate, same input size.
 */
export const CLASSIFY_INPUT_SIZE = 192;

/**
 * AIY Food V1 class names (2024 entries).
 * Index 0 is '__background__'. Some entries are Google KG IDs (start with '/').
 * Used by the classify stage to label detected food items.
 */
export const FOOD_V1_CLASS_NAMES: string[] = foodV1Labels.classNames;
