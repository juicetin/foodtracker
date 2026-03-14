import { VLM_TIER_CONFIG } from '../vlmTypes';

const GB = 1024 ** 3;

let mockTotalMemory: number | null = null;

// Mock expo-device with a getter so mutations are visible to the imported module
jest.mock('expo-device', () => ({
  __esModule: true,
  get totalMemory() {
    return mockTotalMemory;
  },
}));

// Import after mock is set up
import { detectVlmTier, getVlmTierConfig } from '../ramDetector';

describe('ramDetector', () => {
  describe('detectVlmTier', () => {
    afterEach(() => {
      mockTotalMemory = null;
    });

    it('returns "none" when Device.totalMemory is null', () => {
      mockTotalMemory = null;
      expect(detectVlmTier()).toBe('none');
    });

    it('returns "none" for less than 4GB RAM (3GB)', () => {
      mockTotalMemory = 3 * GB;
      expect(detectVlmTier()).toBe('none');
    });

    it('returns "budget" for 4GB RAM', () => {
      mockTotalMemory = 4 * GB;
      expect(detectVlmTier()).toBe('budget');
    });

    it('returns "budget" for 5.99GB RAM', () => {
      mockTotalMemory = 5.99 * GB;
      expect(detectVlmTier()).toBe('budget');
    });

    it('returns "mid" for 6GB RAM', () => {
      mockTotalMemory = 6 * GB;
      expect(detectVlmTier()).toBe('mid');
    });

    it('returns "mid" for 7.99GB RAM', () => {
      mockTotalMemory = 7.99 * GB;
      expect(detectVlmTier()).toBe('mid');
    });

    it('returns "high" for 8GB RAM', () => {
      mockTotalMemory = 8 * GB;
      expect(detectVlmTier()).toBe('high');
    });

    it('returns "high" for 12GB RAM', () => {
      mockTotalMemory = 12 * GB;
      expect(detectVlmTier()).toBe('high');
    });
  });

  describe('getVlmTierConfig', () => {
    afterEach(() => {
      mockTotalMemory = null;
    });

    it('returns null for "none" tier', () => {
      mockTotalMemory = null;
      expect(getVlmTierConfig()).toBeNull();
    });

    it('returns null for low RAM (2GB)', () => {
      mockTotalMemory = 2 * GB;
      expect(getVlmTierConfig()).toBeNull();
    });

    it('returns budget config with correct modelFile for 4GB', () => {
      mockTotalMemory = 4 * GB;
      const config = getVlmTierConfig();
      expect(config).not.toBeNull();
      expect(config!.modelFile).toBe(VLM_TIER_CONFIG.budget.modelFile);
      expect(config!.modelId).toBe('smolvlm-256m-q8');
    });

    it('returns mid config for 6GB', () => {
      mockTotalMemory = 6 * GB;
      const config = getVlmTierConfig();
      expect(config).not.toBeNull();
      expect(config!.modelFile).toBe(VLM_TIER_CONFIG.mid.modelFile);
    });

    it('returns high config for 8GB', () => {
      mockTotalMemory = 8 * GB;
      const config = getVlmTierConfig();
      expect(config).not.toBeNull();
      expect(config!.modelFile).toBe(VLM_TIER_CONFIG.high.modelFile);
    });
  });
});
