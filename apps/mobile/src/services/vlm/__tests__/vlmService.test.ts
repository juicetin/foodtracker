/**
 * Tests for VlmService singleton.
 *
 * Verifies lifecycle management (init/identify/release),
 * idempotent initialization, grammar-constrained output,
 * inactivity timeout, and error handling.
 */

import { initLlama } from 'llama.rn';
import type { LlamaContext } from 'llama.rn';

// Must import after mock is set up by jest.config moduleNameMapper
import { vlmService } from '../vlmService';

// Grab the mock for type-safe assertions
const mockInitLlama = initLlama as jest.Mock;

function getMockContext(): LlamaContext {
  // initLlama resolves to the mock context
  return mockInitLlama.mock.results[0]?.value as unknown as LlamaContext;
}

beforeEach(async () => {
  jest.clearAllMocks();
  // Ensure clean state between tests -- release any lingering context
  await vlmService.release();
});

describe('vlmService', () => {
  describe('isReady', () => {
    it('is false before init', () => {
      expect(vlmService.isReady).toBe(false);
    });

    it('is true after init', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      expect(vlmService.isReady).toBe(true);
    });
  });

  describe('init', () => {
    it('calls initLlama with correct config', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      expect(mockInitLlama).toHaveBeenCalledWith(
        expect.objectContaining({
          model: '/path/model.gguf',
          n_ctx: 2048,
          n_gpu_layers: 99,
          use_mlock: true,
          ctx_shift: false,
        }),
      );
    });

    it('calls initMultimodal on context', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      const ctx = await mockInitLlama.mock.results[0].value;
      expect(ctx.initMultimodal).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/path/mmproj.gguf',
          use_gpu: true,
        }),
      );
    });

    it('is idempotent -- calling twice does not create two contexts', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      expect(mockInitLlama).toHaveBeenCalledTimes(1);
    });

    it('cleans up state on init failure', async () => {
      mockInitLlama.mockRejectedValueOnce(new Error('Load failed'));
      await expect(
        vlmService.init('/bad/model.gguf', '/bad/mmproj.gguf'),
      ).rejects.toThrow('Load failed');
      expect(vlmService.isReady).toBe(false);
    });
  });

  describe('identify', () => {
    it('throws if called before init', async () => {
      await expect(
        vlmService.identify('file:///image.jpg'),
      ).rejects.toThrow('VLM not initialized. Call init() first.');
    });

    it('returns VlmFoodResult', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      const ctx = await mockInitLlama.mock.results[0].value;
      ctx.completion.mockResolvedValueOnce({
        text: '{"dishes":[{"name":"pad thai","cuisine":"Thai","ingredients":["noodles","shrimp"]}]}',
      });

      const result = await vlmService.identify('file:///image.jpg');
      expect(result.dishes).toHaveLength(1);
      expect(result.dishes[0].name).toBe('pad thai');
      expect(result.dishes[0].cuisine).toBe('Thai');
      expect(result.dishes[0].ingredients).toEqual(['noodles', 'shrimp']);
    });

    it('passes user text to prompt builder', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      const ctx = await mockInitLlama.mock.results[0].value;

      await vlmService.identify('file:///image.jpg', 'massaman curry');

      // Check that completion was called with messages containing user text
      const callArgs = ctx.completion.mock.calls[0][0];
      const textContent = JSON.stringify(callArgs.messages);
      expect(textContent).toContain('massaman curry');
    });

    it('returns empty dishes on JSON parse failure', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      const ctx = await mockInitLlama.mock.results[0].value;
      ctx.completion.mockResolvedValueOnce({ text: 'not valid json' });

      const result = await vlmService.identify('file:///image.jpg');
      expect(result).toEqual({ dishes: [] });
    });
  });

  describe('release', () => {
    it('sets isReady to false', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      expect(vlmService.isReady).toBe(true);
      await vlmService.release();
      expect(vlmService.isReady).toBe(false);
    });

    it('is safe to call when not initialized', async () => {
      // Should not throw
      await expect(vlmService.release()).resolves.toBeUndefined();
    });

    it('calls context.release()', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      const ctx = await mockInitLlama.mock.results[0].value;
      await vlmService.release();
      expect(ctx.release).toHaveBeenCalled();
    });
  });

  describe('inactivity timer', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('releases context after 60s inactivity', async () => {
      await vlmService.init('/path/model.gguf', '/path/mmproj.gguf');
      const ctx = await mockInitLlama.mock.results[0].value;

      await vlmService.identify('file:///image.jpg');
      expect(vlmService.isReady).toBe(true);

      // Advance timers by 60 seconds
      jest.advanceTimersByTime(60_000);

      // Wait for any pending async operations from the timer callback
      await Promise.resolve();

      expect(ctx.release).toHaveBeenCalled();
      expect(vlmService.isReady).toBe(false);
    });
  });
});
