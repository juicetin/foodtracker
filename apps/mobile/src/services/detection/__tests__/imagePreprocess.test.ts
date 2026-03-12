/**
 * Tests for image preprocessing: resize + pixel extraction + normalization.
 *
 * Verifies the contract of preprocessImageForModel:
 * - Returns Float32Array of correct length (size * size * 3)
 * - Pixel values normalized to 0-1 range
 * - Throws on invalid/missing URI
 */

// ── Mock expo-image-manipulator ──
const mockManipulateAsync = jest.fn();
jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: (...args: unknown[]) => mockManipulateAsync(...args),
  SaveFormat: { JPEG: 'jpeg', PNG: 'png', WEBP: 'webp' },
}));

// ── Mock expo-file-system ──
const mockReadAsStringAsync = jest.fn();
jest.mock('expo-file-system', () => ({
  readAsStringAsync: (...args: unknown[]) => mockReadAsStringAsync(...args),
  EncodingType: { Base64: 'base64' },
}));

import { preprocessImageForModel } from '../imagePreprocess';

// Helper: create a fake base64 string that represents raw RGBA pixel data
// For testing, we simulate the scenario where the decoder extracts known pixel values.
function createFakeBase64(pixelCount: number, value: number): string {
  // Create RGBA bytes: each pixel has R, G, B, A channels
  const bytes = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    bytes[i * 4] = value;       // R
    bytes[i * 4 + 1] = value;   // G
    bytes[i * 4 + 2] = value;   // B
    bytes[i * 4 + 3] = 255;     // A (fully opaque)
  }
  // Convert to base64 using Buffer (available in Jest/Node environment)
  return Buffer.from(bytes).toString('base64');
}

describe('imagePreprocess', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('preprocessImageForModel', () => {
    it('returns Float32Array of correct length (size * size * 3)', async () => {
      const size = 4; // Small size for testing
      const fakeBase64 = createFakeBase64(size * size, 128);

      mockManipulateAsync.mockResolvedValue({
        uri: 'file:///tmp/resized.jpg',
        width: size,
        height: size,
        base64: fakeBase64,
      });

      const result = await preprocessImageForModel('file:///test/photo.jpg', size);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(size * size * 3);
    });

    it('normalizes pixel values to 0-1 range', async () => {
      const size = 2; // 2x2 image = 4 pixels
      const pixelValue = 128;
      const fakeBase64 = createFakeBase64(size * size, pixelValue);

      mockManipulateAsync.mockResolvedValue({
        uri: 'file:///tmp/resized.jpg',
        width: size,
        height: size,
        base64: fakeBase64,
      });

      const result = await preprocessImageForModel('file:///test/photo.jpg', size);

      // All values should be in 0-1 range
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBeGreaterThanOrEqual(0);
        expect(result[i]).toBeLessThanOrEqual(1);
      }

      // Verify normalization: 128/255 ~= 0.502
      const expected = pixelValue / 255;
      expect(result[0]).toBeCloseTo(expected, 2);
    });

    it('throws on empty/missing URI', async () => {
      await expect(preprocessImageForModel('', 640)).rejects.toThrow();
    });

    it('throws on invalid URI (manipulateAsync fails)', async () => {
      mockManipulateAsync.mockRejectedValue(new Error('File not found'));

      await expect(
        preprocessImageForModel('file:///nonexistent.jpg', 640),
      ).rejects.toThrow('File not found');
    });

    it('calls manipulateAsync with correct resize action', async () => {
      const size = 4;
      const fakeBase64 = createFakeBase64(size * size, 100);

      mockManipulateAsync.mockResolvedValue({
        uri: 'file:///tmp/resized.jpg',
        width: size,
        height: size,
        base64: fakeBase64,
      });

      await preprocessImageForModel('file:///test/photo.jpg', size);

      expect(mockManipulateAsync).toHaveBeenCalledWith(
        'file:///test/photo.jpg',
        [{ resize: { width: size, height: size } }],
        expect.objectContaining({ base64: true }),
      );
    });
  });
});
