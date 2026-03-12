/**
 * Mock for .tflite asset files in Jest.
 *
 * In React Native with Metro bundler, require('path/to/model.tflite')
 * resolves to a numeric asset ID. This mock returns a numeric value
 * to simulate that behavior in tests.
 */
module.exports = 1;
