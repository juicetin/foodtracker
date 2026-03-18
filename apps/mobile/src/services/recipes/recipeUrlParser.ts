/**
 * Recipe URL parser — extract recipe data from HTML using JSON-LD schema.org.
 *
 * Strategy: parse all <script type="application/ld+json"> blocks, find the one
 * with @type "Recipe", and extract structured data (name, ingredients, nutrition,
 * instructions, yield, times).
 *
 * This works on ~80% of recipe sites (AllRecipes, BBC Good Food, Serious Eats,
 * Food Network, NYT Cooking, etc.) because they all embed schema.org Recipe markup.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ParsedNutrition {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface ParsedRecipe {
  name: string;
  description: string | null;
  ingredients: string[];
  instructions: string[] | null;
  servings: string | null;
  totalTime: string | null;
  prepTime: string | null;
  cookTime: string | null;
  nutrition: ParsedNutrition | null;
}

// ---------------------------------------------------------------------------
// JSON-LD extraction
// ---------------------------------------------------------------------------

/** Extract all JSON-LD objects from HTML. */
function extractJsonLd(html: string): unknown[] {
  const results: unknown[] = [];
  const regex = /<script[^>]*type\s*=\s*["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;

  let match;
  while ((match = regex.exec(html)) !== null) {
    try {
      const parsed = JSON.parse(match[1]);
      results.push(parsed);
    } catch {
      // Skip invalid JSON
    }
  }

  return results;
}

/** Recursively find an object with @type "Recipe" in a JSON-LD structure. */
function findRecipeObject(data: unknown): Record<string, unknown> | null {
  if (!data || typeof data !== 'object') return null;

  if (Array.isArray(data)) {
    for (const item of data) {
      const found = findRecipeObject(item);
      if (found) return found;
    }
    return null;
  }

  const obj = data as Record<string, unknown>;

  // Check @type
  const type = obj['@type'];
  if (type === 'Recipe' || (Array.isArray(type) && type.includes('Recipe'))) {
    return obj;
  }

  // Check @graph
  if (Array.isArray(obj['@graph'])) {
    return findRecipeObject(obj['@graph']);
  }

  return null;
}

// ---------------------------------------------------------------------------
// Nutrition parsing
// ---------------------------------------------------------------------------

/** Parse a nutrition value like "280 calories" or "4g" into a number. */
function parseNutritionValue(value: unknown): number {
  if (typeof value === 'number') return value;
  if (typeof value !== 'string') return 0;
  const match = value.match(/[\d.]+/);
  return match ? parseFloat(match[0]) : 0;
}

function parseNutrition(nutrition: unknown): ParsedNutrition | null {
  if (!nutrition || typeof nutrition !== 'object') return null;
  const n = nutrition as Record<string, unknown>;

  const calories = parseNutritionValue(n.calories);
  const protein = parseNutritionValue(n.proteinContent);
  const carbs = parseNutritionValue(n.carbohydrateContent);
  const fat = parseNutritionValue(n.fatContent);

  // If all zero, treat as no nutrition data
  if (calories === 0 && protein === 0 && carbs === 0 && fat === 0) return null;

  return { calories, protein, carbs, fat };
}

// ---------------------------------------------------------------------------
// Instructions parsing
// ---------------------------------------------------------------------------

function parseInstructions(instructions: unknown): string[] | null {
  if (!instructions) return null;

  if (typeof instructions === 'string') {
    return [instructions];
  }

  if (Array.isArray(instructions)) {
    return instructions.map((step) => {
      if (typeof step === 'string') return step;
      if (step && typeof step === 'object') {
        const obj = step as Record<string, unknown>;
        return (obj.text as string) ?? (obj.name as string) ?? String(step);
      }
      return String(step);
    });
  }

  return null;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Parse recipe data from raw HTML.
 * Returns null if no schema.org Recipe is found.
 */
export function parseRecipeFromHtml(html: string): ParsedRecipe | null {
  if (!html) return null;

  const jsonLdBlocks = extractJsonLd(html);
  if (jsonLdBlocks.length === 0) return null;

  // Search all blocks for a Recipe
  let recipe: Record<string, unknown> | null = null;
  for (const block of jsonLdBlocks) {
    recipe = findRecipeObject(block);
    if (recipe) break;
  }

  if (!recipe) return null;

  const name = recipe.name as string;
  if (!name) return null;

  const ingredients = recipe.recipeIngredient;
  if (!Array.isArray(ingredients) || ingredients.length === 0) return null;

  return {
    name,
    description: (recipe.description as string) ?? null,
    ingredients: ingredients.map(String),
    instructions: parseInstructions(recipe.recipeInstructions),
    servings: (recipe.recipeYield as string) ?? null,
    totalTime: (recipe.totalTime as string) ?? null,
    prepTime: (recipe.prepTime as string) ?? null,
    cookTime: (recipe.cookTime as string) ?? null,
    nutrition: parseNutrition(recipe.nutrition),
  };
}
