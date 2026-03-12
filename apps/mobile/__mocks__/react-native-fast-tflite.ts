/**
 * Jest mock for react-native-fast-tflite.
 *
 * Provides a mock TensorflowModel with both sync and async run methods
 * that return empty Float32Array buffers. Tests can override via
 * jest.mocked(loadTensorflowModel).mockResolvedValueOnce(...).
 */

const mockModel = {
  run: jest.fn().mockResolvedValue([new Float32Array(0)]),
  runSync: jest.fn().mockReturnValue([new Float32Array(0)]),
};

export const loadTensorflowModel = jest.fn().mockResolvedValue(mockModel);
