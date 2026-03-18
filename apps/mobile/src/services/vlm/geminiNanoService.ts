/**
 * GeminiNanoService -- Tier 0 food identification via ML Kit GenAI Prompt API.
 *
 * Uses AICore's system-managed Gemini Nano model. No download required.
 * Session-scoped availability cache (same pattern as ramDetector.ts).
 * Falls back silently to SmolVLM when unavailable (see vlmPipeline.ts).
 */

import { geminiNanoModule } from 'gemini-nano';
import type { VlmFoodResult } from './vlmTypes';

// ---------------------------------------------------------------------------
// Prompt (Wave 1 spike version -- keep concise to stay within 256-token limit)
// ---------------------------------------------------------------------------

/**
 * Spike prompt: ask for name + cuisine + ingredients only.
 * Omits portion_hint to reduce token count and avoid truncation.
 * Wave 2 may expand this based on spike output quality observations.
 */
export const SPIKE_PROMPT =
  'Identify the food items in this image. Return JSON with this exact shape: ' +
  '{ "dishes": [{ "name": string, "cuisine": string, "ingredients": string[] }] }. ' +
  'Only include dishes you can see. Be specific (e.g. "pad thai" not "noodles").';

/**
 * Weighted ingredients prompt: dish + ingredient names + estimated gram weights only.
 * Nutrition (macros/micros) is looked up deterministically from a nutrition DB — not
 * estimated by the LLM. Used in the test screen to evaluate weight estimation quality.
 */
export const SPIKE_NUTRITION_PROMPT =
  'Identify all food in this image. Return only valid JSON — no extra text:\n' +
  '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n' +
  'Estimate amount_g using surrounding objects (plates, cutlery, cups, hands) as size references; ' +
  'fall back to a typical restaurant serving size if no reference objects are visible. ' +
  'Be specific with ingredient names (e.g. "basmati rice" not "rice").';

/**
 * Production prompt: ingredients with gram weights, for KG nutrition lookup.
 * Gemini Nano identifies what's in the photo; KG provides the nutrition.
 */
export const FOOD_PROMPT = SPIKE_NUTRITION_PROMPT;

// ---------------------------------------------------------------------------
// Session-scoped availability cache
// ---------------------------------------------------------------------------

/** Cached per app session. Reset on next launch (not persisted). */
let _cachedAvailability: boolean | null = null;

/** Last raw string returned by the native module (before JSON parsing). Used by debug popup. */
let _lastRawOutput: string | null = null;

// ---------------------------------------------------------------------------
// Public service
// ---------------------------------------------------------------------------

export const geminiNanoService = {
  /**
   * Returns true if Gemini Nano is available on this device right now.
   * Caches result for the app session -- AICore state does not change mid-session.
   */
  async isAvailable(): Promise<boolean> {
    if (_cachedAvailability !== null) return _cachedAvailability;
    try {
      const status = await geminiNanoModule.checkAvailability();
      _cachedAvailability = status === 'available';
    } catch {
      _cachedAvailability = false;
    }
    if (__DEV__) {
      console.log('[GeminiNano] Availability:', _cachedAvailability);
    }
    return _cachedAvailability;
  },

  /**
   * Identify food in a photo using Gemini Nano.
   * Parses JSON response into VlmFoodResult.
   * Returns { dishes: [] } on parse failure (same fallback as vlmService).
   *
   * @param photoUri - file:// or content:// URI of the photo
   * @param _userText - optional (ignored by Gemini Nano in Wave 1; reserved for Wave 2 prompt injection)
   */
  async identify(photoUri: string, _userText?: string): Promise<VlmFoodResult> {
    const raw = await geminiNanoModule.identifyFood(photoUri, FOOD_PROMPT);
    // Capture raw output BEFORE parsing -- used by debug popup in DetectionScreen (Wave 2)
    _lastRawOutput = raw;
    try {
      const parsed = JSON.parse(raw) as VlmFoodResult;
      if (parsed?.dishes?.length > 0) return parsed;
    } catch {
      /* fall through */
    }
    return { dishes: [] };
  },

  /**
   * Returns the raw string from the last identifyFood() call (before JSON parsing).
   * Used by DetectionScreen debug popup to show raw Gemini Nano output.
   * Returns null if identify() has not been called yet this session.
   */
  getLastRawOutput(): string | null {
    return _lastRawOutput;
  },

  /** Reset cached availability (for testing only). */
  _resetCache(): void {
    _cachedAvailability = null;
    _lastRawOutput = null;
  },
};
