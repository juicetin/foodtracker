/**
 * Detection pipeline constants.
 *
 * GGCD YOLOv8n detection class names (241 food-specific entries)
 * and model input size for the single-stage bbox-only pipeline.
 *
 * GGCD YOLOv8n input: 640x640 float32 with zero_one normalization.
 */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const detectLabels = require('../../../assets/models/labels_detect.json');

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
