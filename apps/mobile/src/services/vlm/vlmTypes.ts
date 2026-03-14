/**
 * VLM (Vision-Language Model) type contracts.
 *
 * Defines tier configuration, food identification schema,
 * and result types for grammar-constrained VLM output.
 */

/** Device RAM-based VLM tier selection. 'none' means VLM not supported. */
export type VlmTier = 'budget' | 'mid' | 'high' | 'none';

/** Configuration for a specific VLM tier (model files, sizes, RAM requirements). */
export interface VlmTierConfig {
  modelId: string;
  modelFile: string;
  mmprojFile: string;
  /** Model file size in bytes */
  modelSize: number;
  /** Multimodal projector file size in bytes */
  mmprojSize: number;
  /** Total download size in bytes (model + mmproj) */
  totalDownload: number;
  /** Estimated runtime RAM usage in bytes */
  runtimeRam: number;
}

/**
 * Tier configuration for each supported VLM tier.
 * All tiers use the SmolVLM family via llama.rn.
 */
export const VLM_TIER_CONFIG: Record<Exclude<VlmTier, 'none'>, VlmTierConfig> = {
  budget: {
    modelId: 'smolvlm-256m-q8',
    modelFile: 'SmolVLM-256M-Instruct-Q8_0.gguf',
    mmprojFile: 'mmproj-SmolVLM-256M-Instruct-f16.gguf',
    modelSize: 175_000_000,
    mmprojSize: 190_000_000,
    totalDownload: 365_000_000,
    runtimeRam: 500_000_000,
  },
  mid: {
    modelId: 'smolvlm-500m-q8',
    modelFile: 'SmolVLM-500M-Instruct-Q8_0.gguf',
    mmprojFile: 'mmproj-SmolVLM-500M-Instruct-Q8_0.gguf',
    modelSize: 437_000_000,
    mmprojSize: 109_000_000,
    totalDownload: 546_000_000,
    runtimeRam: 1_000_000_000,
  },
  high: {
    modelId: 'smolvlm2-2.2b-q4',
    modelFile: 'SmolVLM2-2.2B-Instruct-Q4_K_M.gguf',
    mmprojFile: 'mmproj-SmolVLM2-2.2B-Instruct-f16.gguf',
    modelSize: 1_110_000_000,
    mmprojSize: 190_000_000,
    totalDownload: 1_300_000_000,
    runtimeRam: 2_000_000_000,
  },
};

/** A single dish identified by the VLM. */
export interface VlmDish {
  name: string;
  cuisine: string;
  ingredients: string[];
  portion_hint?: string;
}

/** VLM food identification result (array of dishes). */
export interface VlmFoodResult {
  dishes: VlmDish[];
}

/**
 * JSON schema for grammar-constrained VLM output.
 * Passed to llama.rn response_format to ensure structurally valid JSON.
 */
export const FOOD_IDENTIFICATION_SCHEMA = {
  name: 'food_identification',
  strict: true,
  schema: {
    type: 'object',
    properties: {
      dishes: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            cuisine: { type: 'string' },
            ingredients: {
              type: 'array',
              items: { type: 'string' },
            },
            portion_hint: { type: 'string' },
          },
          required: ['name', 'cuisine', 'ingredients'],
        },
      },
    },
    required: ['dishes'],
  },
} as const;
