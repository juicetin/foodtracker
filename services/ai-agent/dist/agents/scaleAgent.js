import { LlmAgent, Gemini } from '@google/adk';
const SCALE_PROMPT = `You are a scale reading expert. Analyze the image to detect and read any kitchen scale.

Look for:
1. Digital scale displays showing weight
2. Analog scales with dial indicators
3. The weight value and its unit (grams, kg, oz, lb)

Return your analysis as JSON:
{
  "detected": true | false,
  "weight": number (the numeric value),
  "unit": "g" | "kg" | "oz" | "lb",
  "confidence": 0.0-1.0,
  "scaleType": "digital" | "analog"
}

If no scale is detected, return:
{
  "detected": false,
  "confidence": 1.0
}

Be careful with:
- Decimal points (e.g., "123.4" vs "1234")
- Unit indicators (g vs kg, oz vs lb)
- Reflections or partial readings
- Zero/tare displays

Only report a reading if you can clearly see the numbers.`;
export const scaleAgent = new LlmAgent({
    name: 'ScaleAgent',
    description: 'Detects and reads weight from kitchen scales in images',
    model: new Gemini({ model: 'gemini-2.0-flash-exp' }),
    instruction: SCALE_PROMPT,
});
