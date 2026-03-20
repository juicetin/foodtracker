import { importPhoto } from '../photoImporter';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockManipulateAsync = jest.fn();
jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: (...args: unknown[]) => mockManipulateAsync(...args),
  SaveFormat: { JPEG: 'jpeg' },
}));

const mockGetInfoAsync = jest.fn();
jest.mock('expo-file-system', () => ({
  getInfoAsync: (...args: unknown[]) => mockGetInfoAsync(...args),
  documentDirectory: 'file:///data/user/0/com.tastimate/files/',
  makeDirectoryAsync: jest.fn().mockResolvedValue(undefined),
  moveAsync: jest.fn().mockResolvedValue(undefined),
  copyAsync: jest.fn().mockResolvedValue(undefined),
}));

// Also mock expo-file-system/next since photoImporter uses Paths from there
jest.mock('expo-file-system/next', () => ({
  Paths: {
    document: { uri: 'file:///data/user/0/com.tastimate/files' },
  },
}));

jest.mock('expo-crypto', () => ({
  randomUUID: jest.fn(() => 'test-uuid-1234'),
}));

describe('photoImporter', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('downscales photo with longest edge > 1024px and returns persistent path', async () => {
    mockGetInfoAsync.mockResolvedValue({ exists: false });
    mockManipulateAsync.mockResolvedValue({
      uri: 'file:///cache/resized.jpg',
      width: 1024,
      height: 768,
    });

    const result = await importPhoto('file:///gallery/big-photo.jpg', 'asset-123', {
      width: 4032,
      height: 3024,
    });

    // Should have called manipulateAsync with resize
    expect(mockManipulateAsync).toHaveBeenCalledWith(
      'file:///gallery/big-photo.jpg',
      [{ resize: { width: 1024, height: 768 } }],
      expect.objectContaining({ compress: 0.8 }),
    );

    // Should return a path in gallery-imports/
    expect(result).toContain('gallery-imports');
    expect(result).toContain('.jpg');
  });

  it('does not resize photo already <= 1024px on longest edge', async () => {
    mockGetInfoAsync.mockResolvedValue({ exists: false });
    mockManipulateAsync.mockResolvedValue({
      uri: 'file:///cache/compressed.jpg',
      width: 800,
      height: 600,
    });

    const result = await importPhoto('file:///gallery/small-photo.jpg', 'asset-456', {
      width: 800,
      height: 600,
    });

    // Should call manipulateAsync but WITHOUT resize action
    expect(mockManipulateAsync).toHaveBeenCalledWith(
      'file:///gallery/small-photo.jpg',
      [], // no resize actions
      expect.objectContaining({ compress: 0.8 }),
    );

    expect(result).toContain('gallery-imports');
  });

  it('handles portrait photos (height > width)', async () => {
    mockGetInfoAsync.mockResolvedValue({ exists: false });
    mockManipulateAsync.mockResolvedValue({
      uri: 'file:///cache/resized-portrait.jpg',
      width: 768,
      height: 1024,
    });

    await importPhoto('file:///gallery/portrait.jpg', 'asset-789', {
      width: 3024,
      height: 4032,
    });

    // Longest edge is height (4032), scale to 1024
    expect(mockManipulateAsync).toHaveBeenCalledWith(
      'file:///gallery/portrait.jpg',
      [{ resize: { width: 768, height: 1024 } }],
      expect.objectContaining({ compress: 0.8 }),
    );
  });
});
