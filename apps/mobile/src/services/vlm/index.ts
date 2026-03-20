/**
 * VLM (Vision-Language Model) module barrel exports.
 *
 * Public API for on-device food identification via Gemini Nano.
 */

// Prompt builder
export { buildFoodPrompt } from './vlmPrompts';

// Types and config
export type { VlmTier, VlmTierConfig, VlmFoodResult, VlmDish } from './vlmTypes';
export { VLM_TIER_CONFIG, FOOD_IDENTIFICATION_SCHEMA } from './vlmTypes';

// RAM detection
export { detectVlmTier, getVlmTierConfig } from './ramDetector';
