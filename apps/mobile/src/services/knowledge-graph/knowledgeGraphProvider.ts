/**
 * Lazy singleton provider for KnowledgeGraphService.
 *
 * Resolves the bundled food-knowledge.db via expo-asset on first access,
 * then opens the KnowledgeGraphService once. Subsequent calls return the
 * cached instance immediately.
 *
 * If the database cannot be resolved or opened (e.g., missing asset,
 * corrupt file), logs a warning and returns null -- the detection
 * pipeline falls back to flat-rate proxy nutrition.
 */

import { Asset } from 'expo-asset';
import { KnowledgeGraphService } from './knowledgeGraphService';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const KG_DB_ASSET = require('../../../assets/data/food-knowledge.db');

/** Singleton KG service instance. */
let kgService: KnowledgeGraphService | null = null;

/** Cached local URI after first asset resolution. */
let cachedDbPath: string | null = null;

/** Prevents concurrent initialization races. */
let initPromise: Promise<KnowledgeGraphService | null> | null = null;

/**
 * Get or lazily initialize the KnowledgeGraphService singleton.
 *
 * On first call:
 * 1. Resolves the bundled food-knowledge.db via expo-asset (Asset.fromModule)
 * 2. Creates a KnowledgeGraphService instance
 * 3. Opens the database connection (loads SymSpell index)
 *
 * Returns null if initialization fails (asset not found, DB corrupt, etc.).
 * The detection pipeline will use flat-rate proxy nutrition in that case.
 */
export async function getKnowledgeGraphService(): Promise<KnowledgeGraphService | null> {
  // Already initialized -- return cached instance
  if (kgService) return kgService;

  // Prevent concurrent initialization (multiple detection flows racing)
  if (initPromise) return initPromise;

  initPromise = initializeKGService();
  const result = await initPromise;
  initPromise = null;
  return result;
}

/**
 * Internal initialization -- resolves asset and opens DB.
 */
async function initializeKGService(): Promise<KnowledgeGraphService | null> {
  try {
    // Resolve the bundled DB path
    let dbPath = cachedDbPath;

    if (!dbPath) {
      const asset = Asset.fromModule(KG_DB_ASSET);
      await asset.downloadAsync();

      if (!asset.localUri) {
        console.warn(
          '[KG] food-knowledge.db asset resolved but localUri is null'
        );
        return null;
      }

      // Strip file:// prefix if present -- op-sqlite expects a plain path
      dbPath = asset.localUri.replace(/^file:\/\//, '');
      cachedDbPath = dbPath;
    }

    const service = new KnowledgeGraphService();
    await service.open(dbPath);

    kgService = service;

    if (__DEV__) {
      console.log('[KG] KnowledgeGraphService initialized:', dbPath);
    }

    return kgService;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.warn('[KG] Failed to initialize KnowledgeGraphService:', message);
    return null;
  }
}

/**
 * Close and release the KG service singleton.
 * Used for testing and cleanup.
 */
export function releaseKnowledgeGraphService(): void {
  if (kgService) {
    kgService.close();
    kgService = null;
  }
  cachedDbPath = null;
  initPromise = null;
}
