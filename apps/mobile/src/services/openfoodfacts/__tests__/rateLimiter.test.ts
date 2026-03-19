/**
 * OFF rate limiter tests — adaptive debounce based on rolling request count.
 *
 * OFF limits: 100 product req/min, 10 search req/min.
 */

import { OFFRateLimiter } from '../rateLimiter';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let now = 1000000;
const mockNow = () => now;

function advanceTime(ms: number) {
  now += ms;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('OFFRateLimiter', () => {
  let limiter: OFFRateLimiter;

  beforeEach(() => {
    now = 1000000;
    limiter = new OFFRateLimiter(mockNow);
  });

  describe('product lookups (100/min)', () => {
    it('allows requests when under limit', () => {
      expect(limiter.canRequest('product')).toBe(true);
      expect(limiter.getDelay('product')).toBe(0);
    });

    it('records requests and tracks count', () => {
      for (let i = 0; i < 10; i++) {
        limiter.recordRequest('product');
      }
      expect(limiter.canRequest('product')).toBe(true);
    });

    it('increases delay as we approach the limit', () => {
      // Fill to 80% of limit (80 requests)
      for (let i = 0; i < 80; i++) {
        limiter.recordRequest('product');
      }
      const delay = limiter.getDelay('product');
      expect(delay).toBeGreaterThan(0);
    });

    it('blocks requests at the limit', () => {
      for (let i = 0; i < 100; i++) {
        limiter.recordRequest('product');
      }
      expect(limiter.canRequest('product')).toBe(false);
    });

    it('clears old requests after the window expires', () => {
      for (let i = 0; i < 100; i++) {
        limiter.recordRequest('product');
      }
      expect(limiter.canRequest('product')).toBe(false);

      // Advance past the 60s window
      advanceTime(61000);
      expect(limiter.canRequest('product')).toBe(true);
    });
  });

  describe('search queries (10/min)', () => {
    it('allows requests when under limit', () => {
      expect(limiter.canRequest('search')).toBe(true);
    });

    it('blocks at 10 requests', () => {
      for (let i = 0; i < 10; i++) {
        limiter.recordRequest('search');
      }
      expect(limiter.canRequest('search')).toBe(false);
    });

    it('increases delay above 60% of limit', () => {
      for (let i = 0; i < 7; i++) {
        limiter.recordRequest('search');
      }
      // At 70% (7/10), should have delay
      const delay = limiter.getDelay('search');
      expect(delay).toBeGreaterThan(0);
    });

    it('returns larger delay closer to the limit', () => {
      for (let i = 0; i < 7; i++) {
        limiter.recordRequest('search');
      }
      const delay70 = limiter.getDelay('search');

      for (let i = 0; i < 2; i++) {
        limiter.recordRequest('search');
      }
      const delay90 = limiter.getDelay('search');

      expect(delay90).toBeGreaterThan(delay70);
    });

    it('resets after window expires', () => {
      for (let i = 0; i < 10; i++) {
        limiter.recordRequest('search');
      }
      advanceTime(61000);
      expect(limiter.canRequest('search')).toBe(true);
      expect(limiter.getDelay('search')).toBe(0);
    });
  });

  describe('independent tracking', () => {
    it('tracks product and search separately', () => {
      for (let i = 0; i < 10; i++) {
        limiter.recordRequest('search');
      }
      // Search is maxed out
      expect(limiter.canRequest('search')).toBe(false);
      // Product should still be fine
      expect(limiter.canRequest('product')).toBe(true);
    });
  });
});
