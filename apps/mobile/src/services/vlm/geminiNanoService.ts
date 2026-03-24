/**
 * GeminiNanoService -- Tier 0 food identification via ML Kit GenAI Prompt API.
 *
 * Uses AICore's system-managed Gemini Nano model. No download required.
 * Session-scoped availability cache (same pattern as ramDetector.ts).
 * Falls back silently to SmolVLM when unavailable (see vlmPipeline.ts).
 *
 * Multi-pass identification strategy:
 *   Pass 1 — Dish discovery: list dish names only (~20-50 tokens, always fits in 256).
 *   Pass 2-N — Per-dish detail: ingredients + gram weights for each dish individually.
 * This avoids JSON truncation caused by the 256 maxOutputTokens hard limit.
 */

import { geminiNanoModule } from 'gemini-nano';
import type { VlmFoodResult, VlmDish } from './vlmTypes';

// ---------------------------------------------------------------------------
// Prompts
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
  '{"dishes":[{"name":string,"cuisine":string,"recipe_name":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n' +
  'recipe_name: a concise human-friendly name for the dish as a recipe (e.g. "Chicken Stir Fry with Vegetables"). ' +
  'Estimate amount_g using surrounding objects (plates, cutlery, cups, hands) as size references; ' +
  'fall back to a typical restaurant serving size if no reference objects are visible. ' +
  'Be specific with ingredient names (e.g. "basmati rice" not "rice").';

/**
 * Production prompt: ingredients with gram weights, for KG nutrition lookup.
 * Gemini Nano identifies what's in the photo; KG provides the nutrition.
 */
export const FOOD_PROMPT = SPIKE_NUTRITION_PROMPT;

/** Pass 1: discover dish names only. Tiny output that always fits in 256 tokens. */
const DISCOVERY_PROMPT =
  'List the food dishes visible in this image. Return only valid JSON — no extra text:\n' +
  '{"dishes":["dish name 1","dish name 2"]}\n' +
  'Be specific (e.g. "pad thai" not "noodles").';

/** Pass 2-N: detail a single dish with ingredients + gram weights. */
function dishDetailPrompt(dishName: string): string {
  return (
    `For the dish "${dishName}" visible in this image, list its ingredients with estimated gram weights. Return only valid JSON — no extra text:\n` +
    `{"name":"${dishName}","cuisine":"string","recipe_name":"string","ingredients":[{"name":"string","amount_g":number}]}\n` +
    'Estimate amount_g using surrounding objects as size references; fall back to a typical restaurant serving. ' +
    'Be specific with ingredient names (e.g. "basmati rice" not "rice").'
  );
}

// ---------------------------------------------------------------------------
// Truncation salvage
// ---------------------------------------------------------------------------

/**
 * Attempt to salvage truncated JSON from Gemini Nano output.
 *
 * If the response ends with `}` or `]`, it's likely complete — return as-is.
 * Otherwise, trim to the last complete JSON boundary and close open brackets.
 */
export function salvageTruncatedJson(raw: string): string {
  const trimmed = raw.trim();

  // Quick check: if it ends with a closer AND brackets are balanced, it's likely complete
  if (trimmed.endsWith('}') || trimmed.endsWith(']')) {
    // Verify brackets are balanced before returning as-is
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (const ch of trimmed) {
      if (esc) { esc = false; continue; }
      if (ch === '\\' && inStr) { esc = true; continue; }
      if (ch === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (ch === '[' || ch === '{') depth++;
      else if (ch === ']' || ch === '}') depth--;
    }
    if (depth === 0) return trimmed;
  }

  if (__DEV__) {
    console.warn('[GeminiNano] Truncated response detected, salvaging...');
  }

  // Strategy: trim to the last complete `}` or `]`, then close any unmatched openers
  // in the correct nesting order.
  let salvaged = trimmed;

  // Trim trailing chars after the last `}` or `]`
  const lastClose = Math.max(salvaged.lastIndexOf('}'), salvaged.lastIndexOf(']'));
  if (lastClose === -1) {
    return trimmed; // No valid JSON structure at all
  }
  salvaged = salvaged.substring(0, lastClose + 1);

  // Remove any trailing comma
  salvaged = salvaged.replace(/,\s*$/, '');

  // Track nesting order with a stack so we close in the correct order
  const stack: (']' | '}')[] = [];
  let inString = false;
  let escape = false;
  for (const ch of salvaged) {
    if (escape) { escape = false; continue; }
    if (ch === '\\' && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === '[') stack.push(']');
    else if (ch === '{') stack.push('}');
    else if (ch === ']' || ch === '}') stack.pop();
  }

  // Close remaining open brackets/braces in reverse nesting order
  while (stack.length > 0) {
    salvaged += stack.pop();
  }

  return salvaged;
}

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
   *
   * Uses a multi-pass strategy to avoid JSON truncation:
   *   1. Try single-call with SPIKE_NUTRITION_PROMPT (works for 1 dish).
   *   2. If truncated or multi-dish, use discovery + per-dish detail passes.
   *
   * @param photoUri - file:// or content:// URI of the photo
   * @param _userText - optional (ignored by Gemini Nano in Wave 1; reserved for Wave 2 prompt injection)
   */
  async identify(photoUri: string, _userText?: string): Promise<VlmFoodResult> {
    // --- Single-call attempt (optimization for common 1-dish case) ---
    const raw = await geminiNanoModule.identifyFood(photoUri, FOOD_PROMPT);

    // Check for native module error strings
    if (raw.startsWith('ERROR:')) {
      console.error('[GeminiNano] Native error:', raw);
      throw new Error(`GeminiNano native error: ${raw}`);
    }

    _lastRawOutput = raw;

    // If response looks complete, try parsing directly
    const trimmedRaw = raw.trim();
    if (trimmedRaw.endsWith('}') || trimmedRaw.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmedRaw) as VlmFoodResult;
        if (parsed?.dishes?.length > 0) {
          if (__DEV__) console.log(`[GeminiNano] Single-call success: ${parsed.dishes.length} dish(es)`);
          return parsed;
        }
      } catch (err) {
        if (__DEV__) console.warn('[GeminiNano] JSON parse failed on complete-looking response:', (err as Error).message);
      }
    }

    // Response is truncated — try salvaging first
    if (!trimmedRaw.endsWith('}') && !trimmedRaw.endsWith(']')) {
      const salvaged = salvageTruncatedJson(raw);
      try {
        const parsed = JSON.parse(salvaged) as VlmFoodResult;
        if (parsed?.dishes?.length === 1) {
          // Single dish salvaged successfully — no need for multi-pass
          if (__DEV__) console.log('[GeminiNano] Salvaged truncated single-dish response');
          return parsed;
        }
      } catch {
        // Salvage failed — fall through to multi-pass
      }
    }

    // --- Multi-pass: dish discovery then per-dish detail ---
    if (__DEV__) console.log('[GeminiNano] Falling back to multi-pass identification');

    // Pass 1: discover dish names
    const discoveryRaw = await geminiNanoModule.identifyFood(photoUri, DISCOVERY_PROMPT);

    if (discoveryRaw.startsWith('ERROR:')) {
      console.error('[GeminiNano] Native error in discovery pass:', discoveryRaw);
      throw new Error(`GeminiNano native error: ${discoveryRaw}`);
    }

    _lastRawOutput += '\n---PASS1---\n' + discoveryRaw;

    const discoveryJson = salvageTruncatedJson(discoveryRaw);
    let dishNames: string[] = [];
    try {
      const discoveryParsed = JSON.parse(discoveryJson);
      if (Array.isArray(discoveryParsed?.dishes)) {
        dishNames = discoveryParsed.dishes.filter((d: unknown) => typeof d === 'string');
      }
    } catch (err) {
      if (__DEV__) console.warn('[GeminiNano] Discovery pass parse failed:', (err as Error).message, 'raw:', discoveryRaw.slice(0, 200));
      return { dishes: [] };
    }

    if (dishNames.length === 0) {
      if (__DEV__) console.log('[GeminiNano] Discovery pass found 0 dishes');
      return { dishes: [] };
    }

    if (__DEV__) console.log(`[GeminiNano] Pass 1: discovered ${dishNames.length} dishes:`, dishNames);

    // Pass 2-N: detail each dish
    const dishes: VlmDish[] = [];
    for (let i = 0; i < dishNames.length; i++) {
      const dishName = dishNames[i];
      if (__DEV__) console.log(`[GeminiNano] Pass ${i + 2}/${dishNames.length + 1}: detailing "${dishName}"`);

      const detailRaw = await geminiNanoModule.identifyFood(photoUri, dishDetailPrompt(dishName));

      if (detailRaw.startsWith('ERROR:')) {
        console.error(`[GeminiNano] Native error detailing "${dishName}":`, detailRaw);
        continue; // Skip this dish but try others
      }

      _lastRawOutput += `\n---PASS${i + 2}---\n` + detailRaw;

      const detailJson = salvageTruncatedJson(detailRaw);
      try {
        const detail = JSON.parse(detailJson) as VlmDish;
        if (detail?.name && Array.isArray(detail?.ingredients)) {
          dishes.push(detail);
        }
      } catch (err) {
        if (__DEV__) console.warn(`[GeminiNano] Detail parse failed for "${dishName}":`, (err as Error).message);
      }
    }

    if (__DEV__) console.log(`[GeminiNano] Multi-pass complete: ${dishes.length}/${dishNames.length} dishes detailed`);

    return { dishes };
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
