/**
 * Recipe URL parser tests — extract recipe data from HTML using JSON-LD schema.org.
 *
 * Most recipe sites embed structured data (schema.org/Recipe) in JSON-LD format.
 * This is the most reliable extraction method — works on 80%+ of recipe sites.
 */

import { parseRecipeFromHtml, type ParsedRecipe } from '../recipeUrlParser';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const JSON_LD_RECIPE = `
<html>
<head>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "Classic Banana Bread",
  "description": "Moist and delicious banana bread.",
  "recipeIngredient": [
    "3 ripe bananas",
    "1/3 cup melted butter",
    "1 cup sugar",
    "1 egg",
    "1 teaspoon vanilla extract",
    "1 teaspoon baking soda",
    "1.5 cups all-purpose flour"
  ],
  "recipeInstructions": [
    {"@type": "HowToStep", "text": "Preheat oven to 350°F."},
    {"@type": "HowToStep", "text": "Mix bananas and butter."},
    {"@type": "HowToStep", "text": "Add sugar, egg, vanilla."},
    {"@type": "HowToStep", "text": "Mix in baking soda and flour."},
    {"@type": "HowToStep", "text": "Bake 60 minutes."}
  ],
  "recipeYield": "1 loaf",
  "totalTime": "PT75M",
  "nutrition": {
    "@type": "NutritionInformation",
    "calories": "280 calories",
    "proteinContent": "4g",
    "carbohydrateContent": "45g",
    "fatContent": "10g"
  },
  "prepTime": "PT15M",
  "cookTime": "PT60M"
}
</script>
</head>
<body></body>
</html>`;

const JSON_LD_ARRAY = `
<html>
<head>
<script type="application/ld+json">
[
  {"@context": "https://schema.org", "@type": "WebPage", "name": "Blog"},
  {
    "@context": "https://schema.org",
    "@type": "Recipe",
    "name": "Quick Oats",
    "recipeIngredient": ["1 cup oats", "2 cups water", "pinch of salt"],
    "recipeYield": "2 servings"
  }
]
</script>
</head>
<body></body>
</html>`;

const JSON_LD_GRAPH = `
<html>
<head>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {"@type": "WebPage", "name": "My Blog"},
    {
      "@type": "Recipe",
      "name": "Scrambled Eggs",
      "recipeIngredient": ["3 eggs", "1 tbsp butter", "salt and pepper"],
      "nutrition": {
        "@type": "NutritionInformation",
        "calories": "220 calories"
      }
    }
  ]
}
</script>
</head>
<body></body>
</html>`;

const NO_RECIPE_HTML = `
<html>
<head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "Article", "name": "Not a recipe"}
</script>
</head>
<body><p>Just a blog post.</p></body>
</html>`;

const MULTIPLE_JSON_LD = `
<html>
<head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "WebPage", "name": "Site"}
</script>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "Grilled Cheese",
  "recipeIngredient": ["2 slices bread", "2 slices cheese", "1 tbsp butter"]
}
</script>
</head>
<body></body>
</html>`;

const STRING_INSTRUCTIONS = `
<html>
<head>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "Simple Rice",
  "recipeIngredient": ["1 cup rice", "2 cups water"],
  "recipeInstructions": "Boil water. Add rice. Simmer 20 minutes."
}
</script>
</head>
<body></body>
</html>`;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('parseRecipeFromHtml', () => {
  it('extracts recipe from standard JSON-LD', () => {
    const result = parseRecipeFromHtml(JSON_LD_RECIPE);

    expect(result).not.toBeNull();
    expect(result!.name).toBe('Classic Banana Bread');
    expect(result!.description).toBe('Moist and delicious banana bread.');
    expect(result!.ingredients).toHaveLength(7);
    expect(result!.ingredients[0]).toBe('3 ripe bananas');
    expect(result!.servings).toBe('1 loaf');
    expect(result!.totalTime).toBe('PT75M');
    expect(result!.prepTime).toBe('PT15M');
    expect(result!.cookTime).toBe('PT60M');
  });

  it('extracts nutrition info', () => {
    const result = parseRecipeFromHtml(JSON_LD_RECIPE);
    expect(result!.nutrition).not.toBeNull();
    expect(result!.nutrition!.calories).toBe(280);
    expect(result!.nutrition!.protein).toBe(4);
    expect(result!.nutrition!.carbs).toBe(45);
    expect(result!.nutrition!.fat).toBe(10);
  });

  it('extracts instructions as string array', () => {
    const result = parseRecipeFromHtml(JSON_LD_RECIPE);
    expect(result!.instructions).toHaveLength(5);
    expect(result!.instructions![0]).toContain('Preheat');
  });

  it('handles JSON-LD array format', () => {
    const result = parseRecipeFromHtml(JSON_LD_ARRAY);
    expect(result).not.toBeNull();
    expect(result!.name).toBe('Quick Oats');
    expect(result!.ingredients).toHaveLength(3);
  });

  it('handles @graph format', () => {
    const result = parseRecipeFromHtml(JSON_LD_GRAPH);
    expect(result).not.toBeNull();
    expect(result!.name).toBe('Scrambled Eggs');
    expect(result!.nutrition!.calories).toBe(220);
  });

  it('returns null for non-recipe pages', () => {
    const result = parseRecipeFromHtml(NO_RECIPE_HTML);
    expect(result).toBeNull();
  });

  it('finds recipe across multiple JSON-LD blocks', () => {
    const result = parseRecipeFromHtml(MULTIPLE_JSON_LD);
    expect(result).not.toBeNull();
    expect(result!.name).toBe('Grilled Cheese');
  });

  it('handles string instructions (not array)', () => {
    const result = parseRecipeFromHtml(STRING_INSTRUCTIONS);
    expect(result).not.toBeNull();
    expect(result!.instructions).toHaveLength(1);
    expect(result!.instructions![0]).toContain('Boil water');
  });

  it('returns null for empty HTML', () => {
    expect(parseRecipeFromHtml('')).toBeNull();
    expect(parseRecipeFromHtml('<html><body></body></html>')).toBeNull();
  });
});
