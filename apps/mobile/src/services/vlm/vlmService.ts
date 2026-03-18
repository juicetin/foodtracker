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
import { FOOD_IDENTIFICATION_SCHEMA, type VlmDish, type VlmFoodResult } from './vlmTypes';

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
      // Pass json_schema directly as native param (string) instead of
      // response_format — the response_format→json_schema extraction in
      // llama.rn's completion() wasn't applying the grammar correctly.
      json_schema: JSON.stringify(FOOD_IDENTIFICATION_SCHEMA.schema),
      n_predict: 256,
      temperature: 0.1,
    } as any);

    // Try JSON parse first (works when grammar constraint is active)
    try {
      const parsed = JSON.parse(result.text) as VlmFoodResult;
      if (parsed.dishes?.length > 0) {
        return parsed;
      }
    } catch {
      // Grammar constraint not active for multimodal — fall through to text parser
    }

    // Fallback: parse food names from unstructured VLM text output.
    // llama.rn doesn't apply json_schema grammar for multimodal completions,
    // so SmolVLM outputs plain text lists instead of JSON.
    return { dishes: parsePlainTextDishes(result.text) };
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

/**
 * Extract food names from unstructured VLM text output.
 *
 * Handles common VLM output patterns:
 *   - "- Pineapple: description..."  (bulleted with colon)
 *   - "- Pineapple"                  (bulleted without description)
 *   - "1. Pad Thai"                  (numbered list)
 *   - "Pineapple, Orange, Grapes"    (comma-separated)
 */
function parsePlainTextDishes(text: string): VlmDish[] {
  const seen = new Set<string>();
  const dishes: VlmDish[] = [];

  const lines = text.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Match "- FoodName" or "- FoodName: description" or "* FoodName"
    const bulletMatch = trimmed.match(/^[-*•]\s+([A-Z][A-Za-z\s'-]+?)(?:\s*[:.]|$)/);
    if (bulletMatch) {
      const name = bulletMatch[1].trim();
      // Skip lines that are clearly descriptions (start with lowercase continuation)
      if (name.length >= 2 && name.length <= 50 && !seen.has(name.toLowerCase())) {
        seen.add(name.toLowerCase());
        dishes.push({ name, cuisine: '', ingredients: [] });
      }
      continue;
    }

    // Match "1. FoodName" or "1) FoodName"
    const numberedMatch = trimmed.match(/^\d+[.)]\s+([A-Z][A-Za-z\s'-]+?)(?:\s*[:.]|$)/);
    if (numberedMatch) {
      const name = numberedMatch[1].trim();
      if (name.length >= 2 && name.length <= 50 && !seen.has(name.toLowerCase())) {
        seen.add(name.toLowerCase());
        dishes.push({ name, cuisine: '', ingredients: [] });
      }
    }
  }

  // Fallback: try comma-separated on first line if no bullets found
  if (dishes.length === 0 && lines.length > 0) {
    const parts = lines[0].split(/[,;]+/).map((s) => s.trim()).filter(Boolean);
    for (const part of parts) {
      const name = part.replace(/^[-*•\d.)]+\s*/, '').trim();
      if (name.length >= 2 && name.length <= 50 && /^[A-Z]/.test(name) && !seen.has(name.toLowerCase())) {
        seen.add(name.toLowerCase());
        dishes.push({ name, cuisine: '', ingredients: [] });
      }
    }
  }

  return dishes;
}
