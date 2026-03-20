/**
 * Knowledge Graph module barrel export.
 *
 * Re-exports the KG service, provider, types, and SymSpell index
 * for clean import paths from the detection pipeline.
 */

export { KnowledgeGraphService } from './knowledgeGraphService';
export type { MacroResult, DishResult, RecipeResult, IngredientResult } from './knowledgeGraphService';

export {
  getKnowledgeGraphService,
  releaseKnowledgeGraphService,
} from './knowledgeGraphProvider';

export { SymSpellIndex } from './symspellIndex';
export type { SymSpellMatch } from './symspellIndex';
