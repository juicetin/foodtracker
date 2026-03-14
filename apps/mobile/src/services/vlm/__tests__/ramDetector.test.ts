import { detectVlmTier, getVlmTierConfig } from '../ramDetector';
import { VLM_TIER_CONFIG } from '../vlmTypes';

// Mock expo-device module
jest.mock('expo-device', () => ({
  totalMemory: null as number | null,
}));

// Get reference to the mocked module for per-test mutation
const MockDevice = jest.requireMock('expo-device') as { totalMemory: number | null };

const GB = 1024 ** 3;

describe('ramDetector', () => {
  describe('detectVlmTier', () => {
    afterEach(() => {
      // Reset to null after each test
      MockDevice.totalMemory = null;
    });

    it('returns "none" when Device.totalMemory is null', () => {
      MockDevice.totalMemory = null;
      expect(detectVlmTier()).toBe('none');
    });

    it('returns "none" for less than 4GB RAM (3GB)', () => {
      MockDevice.totalMemory = 3 * GB;
      expect(detectVlmTier()).toBe('none');
    });

    it('returns "budget" for 4GB RAM', () => {
      MockDevice.totalMemory = 4 * GB;
      expect(detectVlmTier()).toBe('budget');
    });

    it('returns "budget" for 5.99GB RAM', () => {
      MockDevice.totalMemory = 5.99 * GB;
      expect(detectVlmTier()).toBe('budget');
    });

    it('returns "mid" for 6GB RAM', () => {
      MockDevice.totalMemory = 6 * GB;
      expect(detectVlmTier()).toBe('mid');
    });

    it('returns "mid" for 7.99GB RAM', () => {
      MockDevice.totalMemory = 7.99 * GB;
      expect(detectVlmTier()).toBe('mid');
    });

    it('returns "high" for 8GB RAM', () => {
      MockDevice.totalMemory = 8 * GB;
      expect(detectVlmTier()).toBe('high');
    });

    it('returns "high" for 12GB RAM', () => {
      MockDevice.totalMemory = 12 * GB;
      expect(detectVlmTier()).toBe('high');
    });
  });

  describe('getVlmTierConfig', () => {
    afterEach(() => {
      MockDevice.totalMemory = null;
    });

    it('returns null for "none" tier', () => {
      MockDevice.totalMemory = null;
      expect(getVlmTierConfig()).toBeNull();
    });

    it('returns null for low RAM (2GB)', () => {
      MockDevice.totalMemory = 2 * GB;
      expect(getVlmTierConfig()).toBeNull();
    });

    it('returns budget config with correct modelFile for 4GB', () => {
      MockDevice.totalMemory = 4 * GB;
      const config = getVlmTierConfig();
      expect(config).not.toBeNull();
      expect(config!.modelFile).toBe(VLM_TIER_CONFIG.budget.modelFile);
      expect(config!.modelId).toBe('smolvlm-256m-q8');
    });

    it('returns mid config for 6GB', () => {
      MockDevice.totalMemory = 6 * GB;
      const config = getVlmTierConfig();
      expect(config).not.toBeNull();
      expect(config!.modelFile).toBe(VLM_TIER_CONFIG.mid.modelFile);
    });

    it('returns high config for 8GB', () => {
      MockDevice.totalMemory = 8 * GB;
      const config = getVlmTierConfig();
      expect(config).not.toBeNull();
      expect(config!.modelFile).toBe(VLM_TIER_CONFIG.high.modelFile);
    });
  });
});
