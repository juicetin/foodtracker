/**
 * Detection pipeline constants.
 *
 * GGCD YOLOv8n detection class names (241 food-specific entries),
 * EfficientNet-Lite0 class names (905 entries), and model input sizes
 * for the two-stage pipeline (detect + classify).
 *
 * EfficientNet-Lite0 input: 224x224 float32 with ImageNet normalization.
 * GGCD YOLOv8n input: 640x640 float32.
 */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const detectLabels = require('../../../assets/models/labels_detect.json');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const classifyLabels = require('../../../assets/models/labels_classify.json');

/**
 * GGCD YOLOv8n detection class names (241 food-specific entries).
 * Index-aligned with model output class indices.
 * Trained on Global Gastronomic Culinary Dataset.
 */
export const DETECT_CLASS_NAMES: string[] = detectLabels.classNames;

/**
 * Input size for the detection model (GGCD YOLOv8n).
 */
export const DETECT_INPUT_SIZE = 640;

/**
 * Input size for the classification model (EfficientNet-Lite0).
 * 224x224 float32 with ImageNet normalization.
 */
export const CLASSIFY_INPUT_SIZE = 224;

/**
 * EfficientNet-Lite0 class names (905 food-specific entries).
 * Index-aligned with model output logits.
 * Trained on merged_v2 dataset (14-dataset global cuisine merge).
 */
export const CLASSIFY_CLASS_NAMES: string[] = classifyLabels.labels;

/** ImageNet normalization mean (RGB channels). */
export const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
/** ImageNet normalization std (RGB channels). */
export const IMAGENET_STD: [number, number, number] = [0.229, 0.224, 0.225];
