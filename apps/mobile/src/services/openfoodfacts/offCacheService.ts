/**
 * OFF cache service -- SQLite cache for Open Food Facts API responses.
 *
 * Implements stale-while-revalidate:
 *  - Product cache: 7-day freshness window
 *  - Search cache: 24-hour freshness window
 *
 * All DB access uses opsqlite.executeSync following existing patterns.
 */

import { opsqlite } from '../../../db/client';
import type { OFFProduct } from './openFoodFactsService';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Product cache freshness threshold (7 days). */
export const PRODUCT_STALE_MS = 7 * 24 * 60 * 60 * 1000;

/** Search cache freshness threshold (24 hours). */
export const SEARCH_STALE_MS = 24 * 60 * 60 * 1000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CacheResult<T> {
  data: T;
  stale: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Normalize a search query for cache key consistency. */
export function normalizeQuery(q: string): string {
  return q.trim().toLowerCase();
}

/** Check if a cached_at timestamp exceeds the staleness threshold. */
export function isStale(cachedAt: string, thresholdMs: number): boolean {
  // SQLite datetime('now') produces 'YYYY-MM-DD HH:MM:SS' (UTC, no Z suffix)
  const timestamp = cachedAt.includes('T')
    ? new Date(cachedAt.endsWith('Z') ? cachedAt : cachedAt + 'Z').getTime()
    : new Date(cachedAt + 'Z').getTime();
  return (Date.now() - timestamp) > thresholdMs;
}

// ---------------------------------------------------------------------------
// Product cache
// ---------------------------------------------------------------------------

/**
 * Look up a cached product by barcode.
 * Returns null if not cached; includes stale flag if cached.
 */
export function getCachedProduct(barcode: string): CacheResult<OFFProduct> | null {
  const result = opsqlite.executeSync(
    'SELECT response_json, cached_at FROM off_product_cache WHERE barcode = ?',
    [barcode],
  );

  if (!result.rows || result.rows.length === 0) return null;

  const row = result.rows[0] as { response_json: string; cached_at: string };
  const data = JSON.parse(row.response_json) as OFFProduct;
  const stale = isStale(row.cached_at, PRODUCT_STALE_MS);

  return { data, stale };
}

/**
 * Insert or update a product in the cache.
 */
export function cacheProduct(barcode: string, product: OFFProduct): void {
  opsqlite.executeSync(
    `INSERT OR REPLACE INTO off_product_cache (barcode, name, brand, response_json, cached_at)
     VALUES (?, ?, ?, ?, datetime('now'))`,
    [barcode, product.name, product.brand ?? null, JSON.stringify(product)],
  );
}

// ---------------------------------------------------------------------------
// Search cache
// ---------------------------------------------------------------------------

/**
 * Look up cached search results by query + pageSize.
 * Returns null if not cached; includes stale flag if cached.
 */
export function getCachedSearch(query: string, pageSize: number): CacheResult<OFFProduct[]> | null {
  const cacheKey = normalizeQuery(query) + '::' + pageSize;
  const result = opsqlite.executeSync(
    'SELECT response_json, cached_at FROM off_search_cache WHERE cache_key = ?',
    [cacheKey],
  );

  if (!result.rows || result.rows.length === 0) return null;

  const row = result.rows[0] as { response_json: string; cached_at: string };
  const data = JSON.parse(row.response_json) as OFFProduct[];
  const stale = isStale(row.cached_at, SEARCH_STALE_MS);

  return { data, stale };
}

/**
 * Insert or update search results in the cache.
 */
export function cacheSearch(query: string, pageSize: number, products: OFFProduct[]): void {
  const normalized = normalizeQuery(query);
  const cacheKey = normalized + '::' + pageSize;
  opsqlite.executeSync(
    `INSERT OR REPLACE INTO off_search_cache (cache_key, query, page_size, response_json, result_count, cached_at)
     VALUES (?, ?, ?, ?, ?, datetime('now'))`,
    [cacheKey, normalized, pageSize, JSON.stringify(products), products.length],
  );
}
