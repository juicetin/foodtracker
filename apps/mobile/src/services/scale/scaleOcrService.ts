/**
 * Scale OCR Service -- Extract weight readings from kitchen scale photos.
 *
 * Primary: Gemini Nano via ML Kit GenAI Prompt API (on-device, no download).
 * Fallback: ML Kit Text Recognition v2 (stub -- requires native dep, deferred).
 * Ultimate fallback: Manual weight input (handled by UI layer, not this service).
 */

import { geminiNanoModule } from 'gemini-nano';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ScaleReading = {
  weightG: number;
  unit: string;
  confidence: 'high' | 'low';
  source: 'gemini-nano' | 'ml-kit' | 'manual';
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum valid weight in grams (after unit conversion). */
const MIN_WEIGHT_G = 0.1;

/** Maximum valid weight in grams (after unit conversion). */
const MAX_WEIGHT_G = 50000;

/**
 * Prompt sent to Gemini Nano to extract scale weight from a photo.
 * Asks for structured JSON with weight number and unit.
 */
export const SCALE_OCR_PROMPT =
  'Look at this image. If there is a kitchen scale or digital display showing a weight, ' +
  'extract the number shown on the display. Return ONLY valid JSON: ' +
  '{ "weight": number | null, "unit": "g" | "kg" | "oz" | "lb" | null }. ' +
  'Return null values if no scale or display is visible.';

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/**
 * Parse raw Gemini Nano response into weight + unit.
 * Handles comma-separated numbers (e.g. "1,234") by stripping commas.
 * Rejects values outside 0.1g-50000g range (after conversion).
 *
 * @returns parsed weight/unit or null if invalid/missing
 */
export function parseScaleResponse(
  raw: string,
): { weight: number; unit: string } | null {
  try {
    const parsed = JSON.parse(raw);

    if (parsed.weight == null || parsed.unit == null) {
      return null;
    }

    let weight: number;
    if (typeof parsed.weight === 'string') {
      // Handle comma-separated numbers like "1,234"
      weight = parseFloat(parsed.weight.replace(/,/g, ''));
    } else {
      weight = Number(parsed.weight);
    }

    if (isNaN(weight)) return null;

    const unit = String(parsed.unit).toLowerCase();
    const weightG = convertToGrams(weight, unit);

    if (weightG < MIN_WEIGHT_G || weightG > MAX_WEIGHT_G) {
      return null;
    }

    return { weight, unit };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------

/**
 * Convert a weight value to grams based on unit.
 * Supports: g (passthrough), kg, oz, lb. Unknown units default to grams.
 */
export function convertToGrams(weight: number, unit: string): number {
  switch (unit.toLowerCase()) {
    case 'kg':
      return weight * 1000;
    case 'oz':
      return weight * 28.3495;
    case 'lb':
      return weight * 453.592;
    case 'g':
    default:
      return weight;
  }
}

// ---------------------------------------------------------------------------
// Main API
// ---------------------------------------------------------------------------

/**
 * Read weight from a kitchen scale photo using Gemini Nano.
 *
 * @param photoUri - file:// or content:// URI of the photo
 * @returns ScaleReading with weight in grams, or null if no scale detected / error
 */
export async function readScaleWeight(
  photoUri: string,
): Promise<ScaleReading | null> {
  try {
    const raw = await geminiNanoModule.identifyFood(photoUri, SCALE_OCR_PROMPT);
    const parsed = parseScaleResponse(raw);
    if (!parsed) return null;

    const weightG = convertToGrams(parsed.weight, parsed.unit);

    return {
      weightG,
      unit: parsed.unit,
      confidence: 'high',
      source: 'gemini-nano',
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// ML Kit Text Recognition v2 fallback (stub)
// ---------------------------------------------------------------------------

/**
 * ML Kit Text Recognition v2 fallback for scale reading.
 *
 * TODO: Implement when @react-native-ml-kit/text-recognition is added as a
 * native dependency. Deferred to gap closure plan if Gemini Nano spike fails
 * on physical device testing.
 *
 * @param _photoUri - unused in stub
 * @returns always null
 */
export async function readScaleWeightMlKit(
  _photoUri: string,
): Promise<ScaleReading | null> {
  // TODO: Implement ML Kit Text Recognition v2 fallback
  return null;
}
