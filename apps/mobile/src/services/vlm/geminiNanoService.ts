/**
 * GeminiNanoService -- Tier 0 food identification via ML Kit GenAI Prompt API.
 *
 * Uses AICore's system-managed Gemini Nano model. No download required.
 * Session-scoped availability cache (same pattern as ramDetector.ts).
 * Falls back silently to mock when unavailable (see vlmPipeline.ts).
 *
 * Strategy:
 *   1. Single-call with FOOD_PROMPT (handles 1-dish photos in one shot).
 *   2. If truncated with multiple dishes, fall back to multi-pass:
 *      a. Discovery pass — get dish names only.
 *      b. Detail pass per dish — get ingredients + gram weights.
 *   3. Retry transient AICore errors (code=4 policy check) before giving up.
 */

import { geminiNanoModule } from 'gemini-nano';
import type { VlmFoodResult, VlmDish } from './vlmTypes';

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

/**
 * Production prompt v5: step-by-step + few-shot examples.
 *
 * Grid-search tested (2026-03-28) against 12 labeled images via Chrome Built-in AI.
 * v5 vs v3 (previous production): +48% composite, +6% recall, +23% precision, 2x faster.
 * See scripts/dspy-eval/prompts/ for versioned prompt history and scores.
 *
 * Few-shot examples anchor the model's weight estimates to plate-realistic portions
 * instead of defaulting to nutrition-label "per 100g" values. Three diverse food types
 * (stir-fry, soup, salad) cover the weight distribution range.
 */
export const FOOD_PROMPT =
  'Look at this food photo. First identify each dish, then list ingredients with gram estimates.\n' +
  'Return only valid JSON:\n' +
  '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n' +
  'For weights: a standard dinner plate is ~25cm, a cup is ~240ml, a fist-sized portion is ~150g.\n' +
  '\n' +
  'Examples:\n' +
  'Fried rice → {"dishes":[{"name":"fried rice","cuisine":"Asian","ingredients":[{"name":"cooked white rice","amount_g":250},{"name":"chicken","amount_g":80},{"name":"soy sauce","amount_g":20},{"name":"green onion","amount_g":10},{"name":"egg","amount_g":50},{"name":"sesame oil","amount_g":5}]}]}\n' +
  'Ramen → {"dishes":[{"name":"ramen","cuisine":"Japanese","ingredients":[{"name":"ramen noodles","amount_g":200},{"name":"pork broth","amount_g":350},{"name":"chashu pork","amount_g":80},{"name":"soft-boiled egg","amount_g":50},{"name":"nori seaweed","amount_g":5},{"name":"green onion","amount_g":10}]}]}\n' +
  'Salad → {"dishes":[{"name":"greek salad","cuisine":"Mediterranean","ingredients":[{"name":"frisee lettuce","amount_g":60},{"name":"feta cheese","amount_g":50},{"name":"cherry tomatoes","amount_g":40},{"name":"kalamata olives","amount_g":25},{"name":"red bell pepper","amount_g":30},{"name":"olive oil","amount_g":10}]}]}';

/** Legacy v3 prompt — kept for A/B testing. */
export const SPIKE_NUTRITION_PROMPT =
  'Identify all food in this image. Return only valid JSON — no extra text:\n' +
  '{"dishes":[{"name":string,"cuisine":string,"recipe_name":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n' +
  'recipe_name: a concise human-friendly name for the dish as a recipe (e.g. "Chicken Stir Fry with Vegetables"). ' +
  'Estimate amount_g using surrounding objects (plates, cutlery, cups, hands) as size references; ' +
  'fall back to a typical restaurant serving size if no reference objects are visible. ' +
  'Be specific with ingredient names (e.g. "basmati rice" not "rice").';

/** Pass 1: discover dish names only. Tiny output fits in 256 tokens. */
const DISCOVERY_PROMPT =
  'List the food dishes visible in this image. Return only valid JSON — no extra text:\n' +
  '{"dishes":["dish name 1","dish name 2"]}\n' +
  'Be specific (e.g. "pad thai" not "noodles").';

/** Pass 2-N: detail a single dish with ingredients + gram weights. */
function dishDetailPrompt(dishName: string): string {
  return (
    `For the dish "${dishName}" visible in this image, list its ingredients with estimated gram weights. Return only valid JSON — no extra text:\n` +
    `{"name":"${dishName}","cuisine":"string","ingredients":[{"name":"string","amount_g":number}]}\n` +
    'For weights: a standard dinner plate is ~25cm, a cup is ~240ml, a fist-sized portion is ~150g. ' +
    'Be specific with ingredient names.'
  );
}

// ---------------------------------------------------------------------------
// Truncation salvage
// ---------------------------------------------------------------------------

/**
 * Attempt to salvage truncated JSON from Gemini Nano output.
 * AICore hard-limits maxOutputTokens to 256 — complex dishes may get truncated.
 */
export function salvageTruncatedJson(raw: string): string {
  let salvaged = raw.trim();

  // If it doesn't end with a closing bracket, cut to last one
  if (!salvaged.endsWith('}') && !salvaged.endsWith(']')) {
    if (__DEV__) console.warn('[GeminiNano] Truncated response detected, salvaging...');

    const lastBrace = salvaged.lastIndexOf('}');
    const lastBracket = salvaged.lastIndexOf(']');
    const cutPoint = Math.max(lastBrace, lastBracket);
    if (cutPoint <= 0) return salvaged;

    salvaged = salvaged.slice(0, cutPoint + 1);
  }

  // Track nesting — close any unmatched openers regardless of whether we cut
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

  while (stack.length > 0) {
    salvaged += stack.pop();
  }

  return salvaged;
}

// ---------------------------------------------------------------------------
// Session-scoped availability cache
// ---------------------------------------------------------------------------

let _cachedAvailability: boolean | null = null;
let _lastRawOutput: string | null = null;

// ---------------------------------------------------------------------------
// Markdown fence stripper
// ---------------------------------------------------------------------------

/** Strip markdown code fences (```json ... ```) that Gemini Nano sometimes wraps around JSON. */
function stripCodeFences(raw: string): string {
  let s = raw.trim();
  // Remove opening fence: ```json or ``` at start
  if (s.startsWith('```')) {
    const firstNewline = s.indexOf('\n');
    if (firstNewline !== -1) {
      s = s.slice(firstNewline + 1);
    }
  }
  // Remove closing fence: ``` at end
  if (s.endsWith('```')) {
    s = s.slice(0, -3);
  }
  return s.trim();
}

// ---------------------------------------------------------------------------
// Retry helper
// ---------------------------------------------------------------------------

const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

async function callWithRetry(photoUri: string, prompt: string): Promise<string> {
  let lastError = '';
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const raw = await geminiNanoModule.identifyFood(photoUri, prompt);
    if (!raw.startsWith('ERROR:')) return raw;
    lastError = raw;
    if (attempt < MAX_RETRIES - 1) {
      console.error(`[GeminiNano] Attempt ${attempt + 1} failed: ${raw}, retrying...`);
      await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
    }
  }
  return lastError;
}

// ---------------------------------------------------------------------------
// Discovery response parser — handles both string[] and object[] formats
// ---------------------------------------------------------------------------

function extractDishNames(discoveryRaw: string): string[] {
  const sanitized = salvageTruncatedJson(stripCodeFences(discoveryRaw));
  try {
    const parsed = JSON.parse(sanitized);
    if (!Array.isArray(parsed?.dishes)) return [];

    // Model may return strings OR objects — handle both
    return parsed.dishes
      .map((d: unknown) => {
        if (typeof d === 'string') return d;
        if (d && typeof d === 'object' && 'name' in d && typeof (d as { name: unknown }).name === 'string') {
          return (d as { name: string }).name;
        }
        return null;
      })
      .filter((n: string | null): n is string => n !== null && n.length > 0);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Public service
// ---------------------------------------------------------------------------

export const geminiNanoService = {
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
   * 1. Single call with FOOD_PROMPT (retries on transient errors).
   * 2. If result is valid and complete → return it.
   * 3. If truncated with 1 dish → salvage and return.
   * 4. If truncated with multiple dishes → multi-pass: discovery + per-dish detail.
   */
  async identify(photoUri: string, _userText?: string): Promise<VlmFoodResult> {
    // --- Step 1: Single call with retry ---
    const raw = await callWithRetry(photoUri, FOOD_PROMPT);

    if (raw.startsWith('ERROR:')) {
      console.error('[GeminiNano] All retries failed:', raw);
      return { dishes: [] };
    }

    _lastRawOutput = raw;

    // --- Step 2: Strip markdown fences + check if truncated ---
    const stripped = stripCodeFences(raw);
    const wasTruncated = !stripped.trim().endsWith('}') && !stripped.trim().endsWith(']');
    const sanitized = salvageTruncatedJson(stripped);

    // Only trust single-call if the response was NOT truncated.
    // Truncated responses may have lost entire dishes — we can't know what was cut off.
    if (!wasTruncated) {
      try {
        const parsed = JSON.parse(sanitized) as VlmFoodResult;
        if (parsed?.dishes?.length > 0) {
          const allComplete = parsed.dishes.every(
            (d) => d.name && Array.isArray(d.ingredients) && d.ingredients.length > 0,
          );
          if (allComplete) {
            if (__DEV__) console.log(`[GeminiNano] Single-call success: ${parsed.dishes.length} dish(es)`);
            return parsed;
          }
        }
      } catch (err) {
        if (__DEV__) console.warn('[GeminiNano] JSON parse failed:', (err as Error).message);
      }
    } else {
      if (__DEV__) console.log('[GeminiNano] Response was truncated — skipping single-call, using multi-pass');
    }

    // --- Step 3: Multi-pass (always runs if truncated) ---
    if (__DEV__) console.log('[GeminiNano] Multi-pass identification');

    // Discovery pass: get dish names
    const discoveryRaw = await callWithRetry(photoUri, DISCOVERY_PROMPT);

    if (discoveryRaw.startsWith('ERROR:')) {
      console.error('[GeminiNano] Discovery pass failed after retries:', discoveryRaw);
      return { dishes: [] };
    }

    _lastRawOutput += '\n---DISCOVERY---\n' + discoveryRaw;

    const dishNames = extractDishNames(discoveryRaw);

    if (dishNames.length === 0) {
      if (__DEV__) console.log('[GeminiNano] Discovery found 0 dishes');
      return { dishes: [] };
    }

    if (__DEV__) console.log(`[GeminiNano] Discovery found ${dishNames.length} dishes:`, dishNames);

    // Detail pass: get ingredients per dish
    const dishes: VlmDish[] = [];
    for (const dishName of dishNames) {
      if (__DEV__) console.log(`[GeminiNano] Detailing "${dishName}"...`);

      const detailRaw = await callWithRetry(photoUri, dishDetailPrompt(dishName));

      if (detailRaw.startsWith('ERROR:')) {
        console.error(`[GeminiNano] Detail pass failed for "${dishName}":`, detailRaw);
        continue;
      }

      _lastRawOutput += `\n---DETAIL:${dishName}---\n` + detailRaw;

      const detailJson = salvageTruncatedJson(stripCodeFences(detailRaw));
      try {
        const detail = JSON.parse(detailJson) as VlmDish;
        if (detail?.name && Array.isArray(detail?.ingredients)) {
          dishes.push(detail);
        }
      } catch (err) {
        if (__DEV__) console.warn(`[GeminiNano] Detail parse failed for "${dishName}":`, (err as Error).message);
      }
    }

    return { dishes };
  },

  getLastRawOutput(): string | null {
    return _lastRawOutput;
  },

  _resetCache(): void {
    _cachedAvailability = null;
    _lastRawOutput = null;
  },
};
