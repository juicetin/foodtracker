/**
 * OFF API rate limiter — adaptive debounce based on rolling request count.
 *
 * OFF rate limits (per user IP, rolling 60s window):
 *   - Product queries: 100/min
 *   - Search queries: 10/min
 *
 * Strategy: track timestamps of recent requests in a rolling window.
 * As we approach the limit, increase suggested delay. At the limit, block.
 * Delay ramps up starting at 60% of the limit.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RequestType = 'product' | 'search';

const LIMITS: Record<RequestType, number> = {
  product: 100,
  search: 10,
};

const WINDOW_MS = 60_000; // 60 seconds

/** Threshold (as fraction of limit) where delay starts ramping. */
const RAMP_START = 0.6;

/** Max delay in ms when at 95%+ of limit. */
const MAX_DELAY_MS = 6_000;

// ---------------------------------------------------------------------------
// Rate Limiter
// ---------------------------------------------------------------------------

export class OFFRateLimiter {
  private timestamps: Record<RequestType, number[]> = {
    product: [],
    search: [],
  };

  private now: () => number;

  constructor(nowFn?: () => number) {
    this.now = nowFn ?? (() => Date.now());
  }

  /** Remove timestamps outside the rolling window. */
  private prune(type: RequestType): void {
    const cutoff = this.now() - WINDOW_MS;
    this.timestamps[type] = this.timestamps[type].filter((t) => t > cutoff);
  }

  /** Record that a request was made. */
  recordRequest(type: RequestType): void {
    this.timestamps[type].push(this.now());
  }

  /** Check if a request is allowed (under the limit). */
  canRequest(type: RequestType): boolean {
    this.prune(type);
    return this.timestamps[type].length < LIMITS[type];
  }

  /**
   * Get the suggested delay in ms before making the next request.
   * Returns 0 when well under the limit. Ramps up as we approach it.
   */
  getDelay(type: RequestType): number {
    this.prune(type);
    const count = this.timestamps[type].length;
    const limit = LIMITS[type];
    const usage = count / limit;

    if (usage < RAMP_START) return 0;

    // Linear ramp from 0 to MAX_DELAY_MS between RAMP_START and 1.0
    const rampProgress = (usage - RAMP_START) / (1 - RAMP_START);
    return Math.round(Math.min(rampProgress, 1) * MAX_DELAY_MS);
  }

  /** Get current request count in the window (for debugging/display). */
  getCount(type: RequestType): number {
    this.prune(type);
    return this.timestamps[type].length;
  }

  /** Get remaining requests in the window. */
  getRemaining(type: RequestType): number {
    this.prune(type);
    return Math.max(0, LIMITS[type] - this.timestamps[type].length);
  }
}

/** Singleton instance for app-wide use. */
export const offRateLimiter = new OFFRateLimiter();
