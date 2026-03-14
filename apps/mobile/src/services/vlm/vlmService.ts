/**
 * VLM inference service singleton.
 *
 * Manages the llama.rn context lifecycle (init, infer, release)
 * for on-device food identification using SmolVLM models.
 *
 * Features:
 * - Lazy init with idempotent guard
 * - Grammar-constrained JSON output via FOOD_IDENTIFICATION_SCHEMA
 * - Inactivity timeout (60s) to free RAM when not in use
 * - Graceful error recovery (no dangling state on failure)
 */

import { initLlama } from 'llama.rn';
import { buildFoodPrompt } from './vlmPrompts';
import { FOOD_IDENTIFICATION_SCHEMA, type VlmFoodResult } from './vlmTypes';

/** Inactivity timeout before releasing context (milliseconds). */
const INACTIVITY_TIMEOUT_MS = 60_000;

// Module-level state (private to this module)
let context: Awaited<ReturnType<typeof initLlama>> | null = null;
let releaseTimer: ReturnType<typeof setTimeout> | null = null;

function clearInactivityTimer(): void {
  if (releaseTimer !== null) {
    clearTimeout(releaseTimer);
    releaseTimer = null;
  }
}

function resetInactivityTimer(): void {
  clearInactivityTimer();
  releaseTimer = setTimeout(() => {
    vlmService.release();
  }, INACTIVITY_TIMEOUT_MS);
}

/**
 * Singleton VLM inference service.
 *
 * Usage:
 * ```typescript
 * await vlmService.init(modelPath, mmprojPath);
 * const result = await vlmService.identify(imageUri, 'pad thai');
 * // result.dishes[0].name === 'pad thai'
 * await vlmService.release(); // or let inactivity timer handle it
 * ```
 */
export const vlmService = {
  /**
   * Initialize the VLM context with a model and multimodal projector.
   * Idempotent -- calling twice with an active context is a no-op.
   *
   * @param modelPath Absolute path to the GGUF model file.
   * @param mmprojPath Absolute path to the mmproj GGUF file.
   * @throws If llama.rn initialization fails.
   */
  async init(modelPath: string, mmprojPath: string): Promise<void> {
    if (context !== null) return;

    try {
      const ctx = await initLlama({
        model: modelPath,
        n_ctx: 2048,
        n_gpu_layers: 99,
        use_mlock: true,
        ctx_shift: false,
      });

      await ctx.initMultimodal({ path: mmprojPath, use_gpu: true });

      context = ctx;
    } catch (err) {
      // Ensure no dangling state on failure
      context = null;
      throw err;
    }
  },

  /**
   * Identify food items in an image using the VLM.
   *
   * @param imageUri File URI or path to the image.
   * @param userText Optional user-provided text for disambiguation.
   * @returns Structured food identification result.
   * @throws If init() has not been called.
   */
  async identify(
    imageUri: string,
    userText?: string,
  ): Promise<VlmFoodResult> {
    if (context === null) {
      throw new Error('VLM not initialized. Call init() first.');
    }

    resetInactivityTimer();

    const prompt = buildFoodPrompt(userText);

    const result = await context.completion({
      messages: [
        {
          role: 'user' as const,
          content: [
            { type: 'image_url', image_url: { url: imageUri } },
            { type: 'text', text: prompt },
          ],
        },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: FOOD_IDENTIFICATION_SCHEMA,
      },
      n_predict: 256,
      temperature: 0.1,
    });

    try {
      return JSON.parse(result.text) as VlmFoodResult;
    } catch {
      // Grammar constraint should prevent this, but gracefully handle
      return { dishes: [] };
    }
  },

  /**
   * Release the VLM context and free RAM.
   * Safe to call when not initialized (no-op).
   */
  async release(): Promise<void> {
    clearInactivityTimer();

    if (context !== null) {
      await context.release();
      context = null;
    }
  },

  /** Whether the VLM context is initialized and ready for inference. */
  get isReady(): boolean {
    return context !== null;
  },
};
