/**
 * VLM prompt templates for food identification.
 *
 * Builds structured prompts for the VLM to identify food items
 * in images, optionally incorporating user-provided text context
 * for disambiguation.
 */

const BASE_PROMPT = `Identify all food items visible in this image. For each food item, provide:
- name: the specific dish or food name (e.g., "massaman curry" not just "curry")
- cuisine: the cuisine type (e.g., "Thai", "Italian", "Japanese")
- ingredients: list of likely main ingredients
- portion_hint: estimated portion description (e.g., "large bowl", "single serving")`;

/**
 * Build a food identification prompt for the VLM.
 *
 * @param userText Optional user-provided text describing the meal.
 *                 When provided, it is appended to the base prompt
 *                 to help disambiguate similar-looking dishes.
 * @returns The complete prompt string.
 */
export function buildFoodPrompt(userText?: string): string {
  if (userText) {
    return `${BASE_PROMPT}\n\nThe user describes this meal as: "${userText}". Use this context to improve identification accuracy.`;
  }
  return BASE_PROMPT;
}
