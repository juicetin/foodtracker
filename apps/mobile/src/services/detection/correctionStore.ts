/**
 * CorrectionStore -- SQLite correction history with suggestion engine.
 *
 * Per locked decision: "Local correction history stored in SQLite -- over time,
 * suggest user's preferred label when similar detections recur (all on-device)"
 *
 * Uses userDb (drizzle) + correctionHistory table from db/schema.ts.
 */

import { eq } from 'drizzle-orm';
import { userDb } from '../../../db/client';
import { correctionHistory } from '../../../db/schema';

export interface CorrectionRecord {
  id: string;
  originalClassName: string;
  correctedClassName: string;
  confidence: number;
  correctedAt: string | null;
}

/** Minimum number of corrections required before generating a suggestion */
const SUGGESTION_THRESHOLD = 3;

export const CorrectionStore = {
  /**
   * Record a user correction: when the user changes the detected food class
   * to a different one.
   */
  async recordCorrection(
    original: string,
    corrected: string,
    confidence: number,
  ): Promise<void> {
    const id = crypto.randomUUID();
    await userDb.insert(correctionHistory).values({
      id,
      originalClassName: original,
      correctedClassName: corrected,
      confidence,
    });
  },

  /**
   * Get all corrections for a given original class name.
   */
  async getCorrections(
    originalClassName: string,
  ): Promise<CorrectionRecord[]> {
    const rows = await userDb
      .select()
      .from(correctionHistory)
      .where(eq(correctionHistory.originalClassName, originalClassName));
    return rows as CorrectionRecord[];
  },

  /**
   * Get the most frequently corrected-to class for a given original class.
   *
   * Only returns a suggestion when the most frequent correction has at least
   * SUGGESTION_THRESHOLD (3) occurrences -- ensuring suggestions are based on
   * a real pattern, not a single correction.
   */
  async getSuggestion(
    originalClassName: string,
  ): Promise<string | null> {
    const corrections = await this.getCorrections(originalClassName);

    if (corrections.length < SUGGESTION_THRESHOLD) {
      return null;
    }

    // Count occurrences of each corrected class
    const counts = new Map<string, number>();
    for (const c of corrections) {
      counts.set(
        c.correctedClassName,
        (counts.get(c.correctedClassName) ?? 0) + 1,
      );
    }

    // Find the most frequent
    let maxCount = 0;
    let bestClass: string | null = null;
    for (const [cls, count] of counts) {
      if (count > maxCount) {
        maxCount = count;
        bestClass = cls;
      }
    }

    // Only suggest if the most frequent correction meets the threshold
    if (maxCount < SUGGESTION_THRESHOLD) {
      return null;
    }

    return bestClass;
  },
};
