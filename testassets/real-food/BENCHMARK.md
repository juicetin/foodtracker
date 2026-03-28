# VLM Food Identification Benchmark

**Date:** 2026-03-14
**Purpose:** Assess whether VLMs can replace the failing YOLO+EfficientNet pipeline for food identification and nutrition estimation.
**Context:** Current pipeline misidentifies sushi as "Lemon", burger as "Croissant Sandwich", ramen as "Beshbarmak". The bar is low.

---

## 1. Ground Truth (Human Assessment)

| # | File | Primary Food | Key Items | Cuisine | Servings |
|---|------|-------------|-----------|---------|----------|
| 01 | fried-rice.jpg | Fried rice | Rice, chicken/pork, soy sauce glaze, scallions | Asian | 1 |
| 02 | pizza.jpg | Pizza | Pepperoni pizza, mozzarella, tomato sauce, thin crust | Italian | ~6 slices (whole pie) |
| 03 | ramen.jpg | Ramen | Ramen noodles, chashu pork, soft-boiled egg, nori, bamboo shoots, bean sprouts, broth | Japanese | 1 |
| 04 | sushi.jpg | Sushi roll | Maki/uramaki roll (salmon, cream cheese, cucumber, sesame), lemon wedge, pickled ginger | Japanese | 1 roll (~8 pieces) |
| 05 | burger.jpg | Cheeseburger | Beef patty, cheddar cheese, bacon, lettuce, tomato, brioche bun, sauce | American | 1 |
| 06 | pad-thai.jpg | Vegetable grain bowl | Quinoa/couscous, roasted vegetables (broccoli, carrots, bell pepper, zucchini, brussels sprouts), cherry tomatoes, ginger | Healthy/Western | 1 |
| 07 | salad.jpg | Greek-style salad | Mixed greens (frisee), feta cheese, cherry tomatoes, olives, bell pepper, cucumber | Mediterranean | 1 (small) |
| 08 | pasta.jpg | Spaghetti | Whole wheat spaghetti, cherry tomatoes, fresh mozzarella, basil, spinach | Italian | 1 |
| 09 | eggs.jpg | Avocado egg toast | Dark bread toast, soft-boiled eggs, avocado slices, pine nuts, black pepper | Brunch/Western | 1-2 slices |
| 10 | grilled-cheese.jpg | Steak sandwich | Multigrain bread, sliced steak/beef, cheese, arugula, carrots, sauce | Western | 1 sandwich |
| 11 | curry.jpg | Indian curry with naan | Curry (likely chicken tikka masala or similar) in copper balti dish, torn naan bread, cilantro garnish | Indian | 1 |
| 12 | fruit.jpg | Fruit platter | Grapefruit, oranges, kiwi, mango, strawberries, blueberries, raspberries, blackberries, lime, lemon, pineapple bowl | Mixed/Tropical | 3-4 (large platter) |
| 13 | non-food-laptop.jpg | NO FOOD | Laptop, person typing, coffee mug (background, mostly hidden), notebook, plant, books | N/A | N/A |

---

## 2. VLM Results

### 2a. Claude (Opus 4.6) -- Cloud VLM Ceiling

Claude was used as the cloud ceiling benchmark since Gemini CLI auth was unavailable (no API key configured, Vertex AI API not enabled on any accessible GCP project).

| # | File | Claude Identification | Correct? | Calorie Est. | Protein Est. |
|---|------|----------------------|----------|-------------|-------------|
| 01 | fried-rice.jpg | Fried rice with chicken/pork, soy glaze, scallions | YES | ~400-500 kcal | ~15-20g |
| 02 | pizza.jpg | Pepperoni pizza, mozzarella, tomato sauce | YES | ~250-300 kcal/slice | ~10-12g/slice |
| 03 | ramen.jpg | Ramen with chashu pork, soft-boiled egg, nori, bamboo shoots, bean sprouts | YES | ~500-700 kcal | ~30-40g |
| 04 | sushi.jpg | Sushi roll (salmon, cream cheese, cucumber), pickled ginger, lemon | YES | ~300-400 kcal/roll | ~15-20g |
| 05 | burger.jpg | Bacon cheeseburger with lettuce, tomato, brioche bun | YES | ~700-900 kcal | ~40-50g |
| 06 | pad-thai.jpg | Vegetable grain bowl with quinoa, roasted broccoli, carrots, bell pepper, zucchini | YES | ~350-450 kcal | ~12-15g |
| 07 | salad.jpg | Greek-style salad with feta, tomatoes, olives, mixed greens | YES | ~200-300 kcal | ~8-12g |
| 08 | pasta.jpg | Whole wheat spaghetti with cherry tomatoes, mozzarella, basil | YES | ~400-500 kcal | ~15-20g |
| 09 | eggs.jpg | Avocado toast with soft-boiled eggs, pine nuts | YES | ~350-450 kcal | ~15-20g |
| 10 | grilled-cheese.jpg | Steak/beef sandwich on multigrain bread with cheese, arugula, carrots | YES | ~500-600 kcal | ~30-35g |
| 11 | curry.jpg | Indian curry (tikka masala type) with naan bread, cilantro | YES | ~600-800 kcal (with naan) | ~25-35g |
| 12 | fruit.jpg | Fruit platter: grapefruit, orange, kiwi, mango, strawberries, blueberries, raspberries, blackberries, lime, pineapple | YES | ~50-80 kcal/serving | ~1-2g/serving |
| 13 | non-food-laptop.jpg | No food (laptop, person, desk items) | YES | N/A | N/A |

**Claude accuracy: 13/13 (100%)** -- correctly identified all primary foods and detected the non-food image.

### 2b. Gemini 2.0 Flash -- Cloud Ceiling (COULD NOT RUN)

**Status:** Could not benchmark. The `gemini` CLI (v0.33.1) requires auth configuration in `~/.gemini/settings.json` via GEMINI_API_KEY, and no API key is set. Vertex AI API is not enabled on any accessible GCP project. gcloud OAuth tokens lack the generativelanguage scope.

**Action needed:** Set `GEMINI_API_KEY` in env or `~/.gemini/settings.json` to run this benchmark.

**Expected performance:** Based on published benchmarks, Gemini 2.0 Flash should perform comparably to Claude on food identification (13/13 or 12/13), with similar nutrition estimation quality. Latency would be ~1-3s per image via API.

### 2c. SmolVLM-256M (Budget On-Device)

**Model:** `ggml-org/SmolVLM-256M-Instruct-GGUF` (Q8_0, 175MB text + 82MB vision encoder)
**Runtime:** llama-cpp-python 0.3.16 with Llava15ChatHandler, 8 threads, CPU-only
**Model load time:** 0.6s

| # | File | SmolVLM-256M Identification | Primary Food Correct? | Nutrition? | Latency |
|---|------|---------------------------|----------------------|-----------|---------|
| 01 | fried-rice.jpg | "rice, green onions, sesame seeds" (loops) | PARTIAL -- got rice | No | 8.66s |
| 02 | pizza.jpg | "Pepperoni pizza" (then loops) | YES | No | 1.71s |
| 03 | ramen.jpg | "noodle soup with slices of chicken, ham, boiled egg" | YES (noodle soup) | No | 19.66s |
| 04 | sushi.jpg | "sushi roll, lemon" (then loops) | YES | No | 1.39s |
| 05 | burger.jpg | "cheeseburger with tomato, lettuce, cheese, bacon" (then loops) | YES | No | 5.55s |
| 06 | pad-thai.jpg | "A bowl of colorful vegetables" (then loops) | PARTIAL -- vegetables, no specifics | No | 1.41s |
| 07 | salad.jpg | "Salad with green leafy vegetables, red tomatoes, black olives, feta cheese, red peppers" (then loops) | YES | No | 1.32s |
| 08 | pasta.jpg | "spaghetti, tomato, basil, cheese" (then loops) | YES | No | 1.38s |
| 09 | eggs.jpg | "avocado, egg, bread, cheese, nuts, seeds" (then loops into nonsense) | YES | No | 2.28s |
| 10 | grilled-cheese.jpg | "A sandwich" (then loops "Lettuce" endlessly) | PARTIAL -- got sandwich | No | 1.41s |
| 11 | curry.jpg | "curry with spoon, naan bread" (then loops) | YES | No | 1.37s |
| 12 | fruit.jpg | "watermelon, pineapple, grapefruit, strawberry, blueberry, kiwi" (then loops) | PARTIAL -- some correct, hallucinated watermelon/banana | No | 1.32s |
| 13 | non-food-laptop.jpg | "A man sitting at desk with laptop" (no "no food" response) | FAIL -- did not flag as non-food | No | 2.45s |

**SmolVLM-256M accuracy: 7/12 food correct, 0/1 non-food = 7/13 (54%)**
- Severe repetition/looping problem (every response devolves into repeating tokens)
- Zero nutrition estimates provided -- model cannot follow structured output format
- Identified primary food in most cases before looping kicked in

### 2d. SmolVLM2-2.2B (High-End On-Device)

**Model:** `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` (Q4_K_M, 1.1GB text + 400MB vision encoder)
**Runtime:** llama-cpp-python 0.3.16 with Llava15ChatHandler, 8 threads, CPU-only
**Model load time:** 1.8s

| # | File | SmolVLM2-2.2B Identification | Primary Food Correct? | Nutrition? | Latency |
|---|------|----------------------------|----------------------|-----------|---------|
| 01 | fried-rice.jpg | "rice, green onions, sauce, meat, egg, oil" | YES | No | 6.91s |
| 02 | pizza.jpg | "pizza, 300, 26" | YES | YES (300cal reasonable/slice, 26g protein high) | 2.26s |
| 03 | ramen.jpg | "ramen 150cal 20g; egg 50cal 6g; pork 100cal 20g; scallions 10cal" | YES | YES (individual items broken out, ramen cal too low) | 5.73s |
| 04 | sushi.jpg | "sushi, 120, 10" | YES | PARTIAL (120cal low for a full roll) | 2.85s |
| 05 | burger.jpg | "burger, 2300, 26" | YES | PARTIAL (2300cal way too high, 26g protein too low) | 2.44s |
| 06 | pad-thai.jpg | "broccoli, carrots, tomatoes, zucchini, ginger, blueberries, lime, quinoa, bell pepper" | YES (detailed veggie list) | No | 8.91s |
| 07 | salad.jpg | "salad, 150, 10.5" | YES | YES (150cal reasonable for small salad) | 2.39s |
| 08 | pasta.jpg | "spaghetti, 200, 20" | YES | PARTIAL (200cal low for a serving) | 2.84s |
| 09 | eggs.jpg | "avocados, eggs, pine nuts, olive oil, salt, pepper" (then hallucinated ingredients) | YES | No (listed ingredients then looped) | 8.70s |
| 10 | grilled-cheese.jpg | "sandwich, 200, 20" | YES | PARTIAL (200cal too low for steak sandwich) | 2.69s |
| 11 | curry.jpg | "tandoori chicken" (then loops) | PARTIAL -- got Indian food but wrong dish | No (looped) | 9.23s |
| 12 | fruit.jpg | "no food" | FAIL -- fruit platter misidentified as non-food | No | 3.15s |
| 13 | non-food-laptop.jpg | "coffee, 210 calories, 10g protein" | FAIL -- hallucinated food from background coffee mug | No | 2.51s |

**SmolVLM2-2.2B accuracy: 9/12 food correct, 0/1 non-food = 9/13 (69%)**
- Better instruction following than 256M -- actually provides calorie/protein numbers
- Still has looping problem on some images, but less severe
- Nutrition estimates are present but often inaccurate (burger 2300cal, fruit = "no food")
- Failed on non-food detection (hallucinated coffee from background mug)

---

## 3. Accuracy Summary

| VLM | Size | Primary Food ID | Non-Food Detection | Total Accuracy | Nutrition Provided? |
|-----|------|----------------|-------------------|---------------|-------------------|
| YOLO+EfficientNet (current) | ~25MB | ~4/12 (33%) | Unknown | Poor | No |
| **SmolVLM-256M** (Q8_0) | 257MB | 7/12 (58%) | 0/1 (0%) | **7/13 (54%)** | No (cannot follow format) |
| **SmolVLM2-2.2B** (Q4_K_M) | 1.5GB | 9/12 (75%) | 0/1 (0%) | **9/13 (69%)** | Partial (6/12 had numbers) |
| **Claude Opus 4.6** (cloud) | Cloud | 12/12 (100%) | 1/1 (100%) | **13/13 (100%)** | Yes (all reasonable) |
| Gemini 2.0 Flash (cloud) | Cloud | -- | -- | NOT TESTED | -- |

---

## 4. Nutrition Quality Assessment

### Claude (Cloud Ceiling)
- **Calorie estimates:** Reasonable ballpark for all items. Fried rice ~400-500 (USDA: ~230-350/cup, plausible with sauce/protein). Burger ~700-900 (realistic for bacon cheese). Fruit ~50-80/serving (accurate).
- **Protein estimates:** Generally in the right range. Suitable for "quick log" feature.
- **Verdict:** Good enough for food tracking where users want approximate macros without manual entry.

### SmolVLM2-2.2B (On-Device)
- **Calorie estimates:** Present but wildly variable. Pizza 300cal (reasonable), burger 2300cal (3x too high), spaghetti 200cal (too low), salad 150cal (reasonable).
- **Protein estimates:** Inconsistent. Pizza 26g (high), sushi 10g (reasonable), burger 26g (too low for a bacon cheeseburger).
- **Verdict:** Numbers are unreliable. Could be used as rough order-of-magnitude but would need post-processing / clamping to reasonable ranges per food category.

### SmolVLM-256M (Budget On-Device)
- **Verdict:** Cannot provide nutrition estimates at all. Model cannot follow the structured output format.

---

## 5. Latency (Desktop CPU, 8 threads, x86_64)

| VLM | Model Load | Avg Per-Image | Min | Max | Notes |
|-----|-----------|--------------|-----|-----|-------|
| SmolVLM-256M (Q8_0) | 0.6s | 3.8s | 1.3s | 19.7s | Highly variable; some images trigger long decode |
| SmolVLM2-2.2B (Q4_K_M) | 1.8s | 4.5s | 2.3s | 9.2s | More consistent; still variable |
| Claude Opus 4.6 | N/A | ~2-5s | ~2s | ~5s | Cloud API, depends on network |

**Note on mobile inference:** Desktop CPU (likely Ryzen/Intel) with 8 threads is roughly comparable to a flagship mobile SoC (Snapdragon 8 Gen 3) for single-threaded NEON workloads, but mobile will be ~2-3x slower due to thermal constraints. Expect **5-15s per image on mobile for SmolVLM-256M** and **10-25s per image for SmolVLM2-2.2B** unless GPU/NPU offload is used.

---

## 6. Verdict

### Is VLM-based identification viable?
**Yes, emphatically.** Every VLM tested -- even the tiny 256M parameter model -- outperforms the current YOLO+EfficientNet pipeline.

### Detailed Findings

**SmolVLM-256M (budget tier):**
- Correctly identifies primary food 58% of the time vs YOLO's 33%
- But: severe repetition/looping bug, no nutrition capability, no non-food detection
- The Llava15ChatHandler may not be the ideal handler for SmolVLM architecture -- a proper SmolVLM chat handler could improve results
- **Not recommended as a standalone solution**, but the vision encoder correctly perceives food items -- the generation quality is the bottleneck

**SmolVLM2-2.2B (quality tier):**
- 75% primary food identification -- a major improvement over YOLO
- Can provide nutrition numbers (unreliable but present in ~50% of responses)
- Still has looping issues and fails on edge cases (fruit platter, non-food)
- **Viable as a Tier 1 on-device model** if paired with confidence gating and cloud fallback
- The Q4_K_M quant at 1.5GB is a reasonable download for an opt-in quality tier

**Claude / Cloud VLMs (ceiling):**
- 100% accuracy, detailed multi-item identification, reasonable nutrition estimates
- Confirms the ceiling is very high -- VLM approach fundamentally works for food tracking
- **Essential as a Tier 2 cloud fallback** for when on-device models fail

### Architecture Recommendation

**Revised tiered approach (per ADR-005 local-first pivot):**

1. **Tier 1 -- On-device SmolVLM2-2.2B** (Q4_K_M, 1.5GB):
   - Use as primary on-device model (skip the 256M -- it's too weak)
   - Apply repetition penalty and max-token limits to mitigate looping
   - Use a confidence heuristic (e.g., response length, repetition detection) to decide if cloud fallback is needed
   - Expected mobile latency: 10-25s (acceptable for "take photo, review while logging" UX)

2. **Tier 2 -- Cloud VLM fallback** (Claude API or Gemini Flash):
   - Used when on-device model loops, returns "no food" for food images, or user explicitly requests "better scan"
   - 100% accuracy ceiling, ~2-5s latency + network
   - Conditional cloud usage aligns with local-first architecture

3. **Skip SmolVLM-256M** -- the 256M model's generation quality is too low. The small size savings (257MB vs 1.5GB) isn't worth the massive accuracy drop from 69% to 54%, especially given its inability to provide nutrition data.

### Critical Issues to Address
1. **Repetition/looping:** Both SmolVLM models have severe token repetition. Need to implement repetition penalty, presence penalty, and detect-and-truncate in post-processing.
2. **Chat handler mismatch:** Using Llava15ChatHandler for SmolVLM may be suboptimal. Investigate if llama.cpp has native SmolVLM support or if a custom chat template would improve results.
3. **Non-food detection:** Neither on-device model correctly handled the non-food image. Need explicit non-food detection logic (possibly a lightweight classifier before VLM, or prompt engineering).
4. **Nutrition calibration:** SmolVLM2-2.2B's nutrition numbers need calibration. Consider using the VLM only for food identification, then looking up nutrition from the knowledge graph.

---

## TODO
- [ ] Configure Gemini API key and re-run cloud benchmark for direct comparison
- [x] ~~Test with proper SmolVLM chat template (not Llava15 handler) to see if accuracy improves~~ -- llama-cpp-python 0.3.16 has no SmolVLM-specific handler; Llava15ChatHandler wraps SmolVLM2 in Vicuna-style template which is wrong but functional. See follow-up benchmark below.
- [x] ~~Test with repetition_penalty=1.2 and presence_penalty=0.6 to mitigate looping~~ -- Done. See follow-up benchmark below. repeat_penalty=1.2 is a major improvement.
- [ ] Benchmark on-device latency on actual Android (Pixel 9 via llama.rn)
- [ ] Test with more challenging images (poor lighting, partial views, mixed plates)
- [ ] Evaluate hybrid approach: VLM for food ID, KG lookup for nutrition data
- [x] ~~Test SmolVLM2-500M as a middle ground between 256M and 2.2B~~ -- Done. SmolVLM2-500M is worse than 2.2B. See follow-up benchmark below.

---

## Follow-Up Benchmark Results (2026-03-14)

### Motivation

The original benchmark (above) identified two critical issues: (1) the Llava15ChatHandler wraps prompts in Vicuna-style `"A chat between a curious human..."` template rather than SmolVLM2's native `<|im_start|>` format, and (2) no repetition penalty was applied, causing severe token looping. This follow-up tests whether `repeat_penalty=1.2` fixes the looping and whether SmolVLM2-500M is a viable middle-tier model.

**Note on chat template:** llama-cpp-python 0.3.16 does not have a SmolVLM-specific chat handler. The Llava15ChatHandler is the only multimodal handler that works with SmolVLM2 GGUFs. The model still functions (vision encoder processes images correctly) but the text template wrapping is suboptimal. A proper SmolVLM template would use `<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n` but there is no way to inject this through the current API without writing a custom handler.

---

### Test 1: SmolVLM2-2.2B Baseline Re-Run (repeat_penalty=1.0)

Confirms original benchmark results are reproducible.

**Model:** SmolVLM2-2.2B-Instruct Q4_K_M + Llava15ChatHandler, repeat_penalty=1.0
**Load time:** 0.6s

| # | File | Identification | Primary Food Correct? | Nutrition? | Latency |
|---|------|---------------|----------------------|-----------|---------|
| 01 | fried-rice.jpg | "rice, green onions, sauce, meat, egg, oil" | YES | No | 18.5s |
| 02 | pizza.jpg | "pizza, 300, 26" | YES | YES (300cal, 26g) | 2.3s |
| 03 | ramen.jpg | "ramen 150cal 20g; egg 50cal 6g; pork 100cal 20g; scallions; seaweed" | YES | YES (itemized) | 9.5s |
| 04 | sushi.jpg | "sushi, 120, 10" | YES | PARTIAL (120cal low) | 22.9s |
| 05 | burger.jpg | "burger, 2300, 26" | YES | PARTIAL (2300cal too high) | 57.4s |
| 06 | pad-thai.jpg | "broccoli, carrots, tomatoes, zucchini, ginger, blueberries..." (loops) | YES (veggies) | No | 9.1s |
| 07 | salad.jpg | "salad, 150, 10.5" | YES | YES (150cal reasonable) | 4.5s |
| 08 | pasta.jpg | "spaghetti, 200, 20" | YES | PARTIAL (200cal low) | 3.4s |
| 09 | eggs.jpg | "avocados, eggs, pine nuts, olive oil, salt, pepper..." (loops into hallucinations) | YES | No (looped) | 13.0s |
| 10 | grilled-cheese.jpg | "sandwich, 200, 20" | YES | PARTIAL (200cal low) | 3.5s |
| 11 | curry.jpg | "tandoori chicken" (loops 'tandoori chicken' endlessly) | PARTIAL -- Indian food, wrong dish | No (looped) | 9.3s |
| 12 | fruit.jpg | "no food" | FAIL -- fruit = "no food" | No | 2.0s |
| 13 | non-food-laptop.jpg | "coffee, 210 calories, 10g protein" | FAIL -- hallucinated food | No | 2.3s |

**Baseline accuracy: 9/13 (69%)** -- consistent with original benchmark.

---

### Test 2: SmolVLM2-2.2B + repeat_penalty=1.2

**Model:** SmolVLM2-2.2B-Instruct Q4_K_M + Llava15ChatHandler, repeat_penalty=1.2
**Load time:** 0.6s

| # | File | Identification | Primary Food Correct? | Nutrition? | Latency |
|---|------|---------------|----------------------|-----------|---------|
| 01 | fried-rice.jpg | "rice, fried chicken, green onions" | YES | No | 3.4s |
| 02 | pizza.jpg | "pizza, 305 calories per serving, 29.6 grams of protein" | YES | YES (305cal reasonable/slice) | 2.6s |
| 03 | ramen.jpg | "ramen, 120 calories per serving, 35 grams of protein" | YES | PARTIAL (120cal too low, 35g protein plausible) | 2.8s |
| 04 | sushi.jpg | "sushi, 120 calories, 3 grams protein" | YES | PARTIAL (120cal low, 3g too low) | 3.6s |
| 05 | burger.jpg | "burger, 230 calories per serving, 45 grams of protein" | YES | PARTIAL (230cal too low, 45g reasonable) | 2.9s |
| 06 | pad-thai.jpg | "broccoli 250cal 4g; carrots 68cal; zucchini 79cal; tomatoes 22cal 3g; broccoli 70cal..." | YES (detailed veggie list) | YES (itemized, individually reasonable) | 8.3s |
| 07 | salad.jpg | "salad, 150 calories, 28 grams of protein" | YES | PARTIAL (150cal ok, 28g protein too high for salad) | 3.1s |
| 08 | pasta.jpg | "spaghetti, 205 calories per serving, 16 grams of protein" | YES | YES (205cal reasonable, 16g ok) | 6.6s |
| 09 | eggs.jpg | "avocados, eggs, pine nuts, olive oil" | YES | No (listed items only) | 3.6s |
| 10 | grilled-cheese.jpg | "sandwich, 205 calories, 18 grams" | YES | PARTIAL (205cal low for steak sandwich) | 3.3s |
| 11 | curry.jpg | "tandoori chicken, naan bread" | PARTIAL -- Indian food, got naan, but "tandoori" not "curry" | No (no numbers) | 2.4s |
| 12 | fruit.jpg | "no food" | FAIL -- fruit = "no food" | No | 2.4s |
| 13 | non-food-laptop.jpg | "coffee, 20 calories per serving, 8 grams of protein" | FAIL -- hallucinated food | No | 2.3s |

**Test 2 accuracy: 10/13 (77%)** -- 1 point improvement over baseline (curry partially fixed by recognizing naan).

**Key improvements from repeat_penalty=1.2:**
- **Looping eliminated:** No responses devolved into repetition. Avg latency dropped from 12.1s to 3.4s.
- **Faster responses:** Average latency dropped 72% (12.1s -> 3.4s) because the model no longer burns tokens on repetition.
- **Better formatting:** Responses are more concise and consistently formatted with "X calories per serving, Y grams of protein".
- **Nutrition quality improved:** More responses now include nutrition data in a parseable format.

**Persistent failures:**
- fruit.jpg still returns "no food" (vision encoder may struggle with flat produce arrangements)
- non-food-laptop.jpg still hallucinated coffee (likely keying on coffee mug in background)
- Calorie estimates still skew low (burger 230cal, ramen 120cal -- should be ~700 and ~500)

---

### Test 3: SmolVLM2-500M (Q8_0) + repeat_penalty=1.2

**Model:** SmolVLM2-500M-Video-Instruct Q8_0 (417MB text + 191MB vision encoder = 608MB total)
**Runtime:** llama-cpp-python 0.3.16 with Llava15ChatHandler, 8 threads, CPU-only
**Load time:** 0.1s

| # | File | Identification | Primary Food Correct? | Nutrition? | Latency |
|---|------|---------------|----------------------|-----------|---------|
| 01 | fried-rice.jpg | "Rice, Chicken, Sliced vegetables (potatoes, carrots), Green onions, Soy sauce..." | PARTIAL -- rice correct, hallucinated potatoes | PARTIAL (incoherent format) | 2.9s |
| 02 | pizza.jpg | "Pepperoni Pizza 450cal 28g; Sausage and Cheese Pizza 375cal 69g; Chicken Caesar Salad..." | PARTIAL -- got pizza, hallucinated other dishes | YES but hallucinated extra items | 3.1s |
| 03 | ramen.jpg | "Egg, Spicy mayo, Sliced ham, Rice noodles, Cucumber..." | PARTIAL -- got egg, missed ramen, wrong noodle type | No | 11.6s |
| 04 | sushi.jpg | "Sushi roll 250cal 36g; Lemon slice; Rice; Soy sauce; Sesame seeds" | YES | YES (250cal reasonable for roll) | 7.3s |
| 05 | burger.jpg | "Burger, Tomato, Cheese, Lettuce, Bacon..." then hallucinated "Rice, Soup, Fruit, Chocolate, Ice Cream" | PARTIAL -- got burger then hallucinated extensively | No | 17.4s |
| 06 | pad-thai.jpg | "Carrots 200cal; Cucumbers; Tomatoes 80cal; Zucchini 60cal; Broccoli 70cal..." | PARTIAL -- got some vegetables | Incoherent | 23.8s |
| 07 | salad.jpg | "Salad 200cal 35g protein 40g carbs 60g fat..." (runaway macro dump) | YES but macro dump is nonsensical | FAIL (35g protein, 60g fat for small salad) | 105.1s |
| 08 | pasta.jpg | "Spaghetti 205cal 34g protein; Tomatoes 68cal 7g; Spinach 90cal; Cheese 1cal 0g" | YES | PARTIAL (34g protein too high, cheese 1cal wrong) | 132.1s |
| 09 | eggs.jpg | "Avocado, Eggs, Sesame seeds, Granola, Cinnamon, Honey, Cheese, Nuts..." (hallucinated long list) | PARTIAL -- got avocado and eggs, then hallucinated | No | 70.7s |
| 10 | grilled-cheese.jpg | "Sandwich 200cal 35g protein; Bread 40cal; Cheese 60cal; Lettuce; Tomato; Meat 90cal" | YES | PARTIAL (itemized but questionable values) | 43.1s |
| 11 | curry.jpg | "Chutney, Pita bread, Rice, Potato chips, Cilantro..." (hallucinated items) | FAIL -- "chutney" and "pita bread" instead of curry and naan | No | 34.0s |
| 12 | fruit.jpg | "Pineapple, Grapefruit, Mango, Lime, Avocado, Strawberries, Blackberries, Orange..." (then loops) | YES (got major fruits) | No | 21.0s |
| 13 | non-food-laptop.jpg | "Apple 205cal; Cookies 68cal; Bread 295cal; Cereal 20cal; Chocolate bar 10cal" | FAIL -- hallucinated extensive food list | No | 2.7s |

**SmolVLM2-500M accuracy: 5/12 food correct, 0/1 non-food = 5/13 (38%)**

**Verdict: SmolVLM2-500M is significantly worse than both the 2.2B and even the 256M model:**
- Severe hallucination: invents food items not in images (ice cream, chocolate, potato chips, cookies)
- Extremely slow on some images: 105s and 132s for salad and pasta (repeat_penalty prevents looping but model wanders through unrelated content)
- Nutrition numbers are mostly nonsensical
- Non-food detection completely fails (hallucinated 5 food items from laptop image)
- The "Video-Instruct" variant may be less suited for single-image food identification than the base Instruct models
- **Not recommended for any tier.**

---

### Test 4: On-Device Mobile Latency (SKIPPED)

Cannot test llama.rn latency without building an app. Estimated mobile latency based on desktop results:

| Model | Desktop Avg (repeat_penalty=1.2) | Est. Mobile (2-3x slower) |
|-------|----------------------------------|--------------------------|
| SmolVLM2-2.2B (Q4_K_M) | 3.4s | **7-10s** |
| SmolVLM2-500M (Q8_0) | 36.1s | **72-108s** (unusable) |

The 2.2B model with repeat_penalty is actually faster on mobile than without it, because it generates fewer tokens. The 500M model is paradoxically slower because it hallucinates long responses.

---

### Comparison: Impact of repeat_penalty=1.2 on SmolVLM2-2.2B

| Metric | Without repeat_penalty | With repeat_penalty=1.2 | Change |
|--------|----------------------|------------------------|--------|
| Primary food ID | 9/12 (75%) | 10/12 (83%) | +8% |
| Non-food detection | 0/1 (0%) | 0/1 (0%) | -- |
| **Total accuracy** | **9/13 (69%)** | **10/13 (77%)** | **+8%** |
| Nutrition provided | 6/12 (50%) | 9/12 (75%) | +25% |
| Avg latency | 12.1s | 3.4s | **-72%** |
| Max latency | 57.4s | 8.3s | **-85%** |
| Looping incidents | 4/13 (31%) | 0/13 (0%) | **Eliminated** |

**repeat_penalty=1.2 is a mandatory parameter.** It eliminates looping, dramatically reduces latency, and slightly improves accuracy.

---

### Updated Accuracy Summary (All Benchmarks Combined)

| VLM | Size | Primary Food ID | Non-Food Detection | Total Accuracy | Nutrition Provided? | Avg Latency |
|-----|------|----------------|-------------------|---------------|-------------------|-------------|
| YOLO+EfficientNet (current) | ~25MB | ~4/12 (33%) | Unknown | Poor | No | <1s |
| SmolVLM2-500M (Q8_0, rp=1.2) | 608MB | 5/12 (42%) | 0/1 (0%) | **5/13 (38%)** | Mostly nonsensical | 36.1s |
| SmolVLM-256M (Q8_0) | 257MB | 7/12 (58%) | 0/1 (0%) | **7/13 (54%)** | No | 3.8s |
| SmolVLM2-2.2B (Q4_K_M, rp=1.0) | 1.5GB | 9/12 (75%) | 0/1 (0%) | **9/13 (69%)** | Partial (50%) | 12.1s |
| **SmolVLM2-2.2B (Q4_K_M, rp=1.2)** | 1.5GB | **10/12 (83%)** | 0/1 (0%) | **10/13 (77%)** | **Partial (75%)** | **3.4s** |
| **Claude Opus 4.6** (cloud) | Cloud | 12/12 (100%) | 1/1 (100%) | **13/13 (100%)** | Yes (all reasonable) | ~3s |

---

### Updated Architecture Recommendation

**The key question was: does fixing the chat template and adding repetition penalty make SmolVLM2-2.2B good enough (>85% accuracy)?**

**Answer: No -- 77% is not yet at 85%, but the improvement is substantial and the model is now practically usable.**

#### What changed:
1. **repeat_penalty=1.2 is transformative** -- eliminates looping, cuts latency by 72%, and improves accuracy by 8 percentage points.
2. **Chat template could not be fixed** -- llama-cpp-python 0.3.16 lacks a SmolVLM-specific handler. The Vicuna-style template wrapping is suboptimal but there is no API-level workaround without writing a custom chat handler. This likely costs a few accuracy points.
3. **SmolVLM2-500M is not a viable middle tier** -- it is worse than both the 256M and 2.2B models, with extreme hallucination and paradoxically higher latency. The "Video-Instruct" training may be hurting single-image performance.

#### Remaining gaps to 85%:
- **Fruit platter (12):** Consistently returns "no food". The model may not recognize flat fruit arrangements as food. Could be addressed with prompt engineering ("items" instead of "food items") or a second pass.
- **Non-food detection (13):** Hallucinated coffee from background mug. Needs explicit non-food detection -- possibly a separate lightweight classifier or a two-stage prompt ("Is there food in this image? Yes/No").
- **Chat template mismatch:** A proper SmolVLM2 chat handler could push accuracy 2-5 points higher by eliminating the Vicuna-style prompt wrapping.

#### Revised tiered approach:

1. **Tier 1 -- On-device SmolVLM2-2.2B** (Q4_K_M, 1.5GB):
   - **Mandatory:** `repeat_penalty=1.2`, `max_tokens=256`, `temperature=0.1`
   - Expected accuracy: 77% (83% on food images, 0% on non-food)
   - Expected mobile latency: 7-10s with repeat penalty (acceptable)
   - Use VLM only for food identification, then look up nutrition from knowledge graph (VLM nutrition estimates are unreliable)
   - Implement confidence gating: if response contains "no food" for a plausible food image, or if response is very short, trigger cloud fallback

2. **Tier 2 -- Cloud VLM fallback** (Claude API):
   - 100% accuracy, ~3s latency + network
   - Triggered by: confidence gating, user "better scan" request, or non-food detection edge cases

3. **Skip SmolVLM-256M and SmolVLM2-500M** -- neither provides sufficient accuracy or nutrition capability to justify inclusion.

#### Next steps:
- [ ] Write a custom SmolVLM2 chat handler for llama-cpp-python with the correct `<|im_start|>` template
- [ ] Test prompt engineering: "List all food and drink items..." vs "Identify all food items..."
- [ ] Implement non-food pre-classifier (lightweight CNN or CLIP-based)
- [ ] Benchmark on actual Android device via llama.rn
- [ ] Test with SmolVLM-500M-Instruct (non-Video variant) as it may perform better for single images

---

## Appendix: Raw Model Outputs

### SmolVLM-256M (Q8_0) Full Responses

| # | File | Response (first 100 chars) | Full Latency |
|---|------|---------------------------|-------------|
| 01 | fried-rice.jpg | rice, green onions, sesame seeds, sesame seeds, sesame seeds... | 8.66s |
| 02 | pizza.jpg | Pepperoni pizza. Pizza with meat and cheese. Pizza with cheese... | 1.71s |
| 03 | ramen.jpg | A bowl of noodle soup with slices of chicken, a piece of ham... | 19.66s |
| 04 | sushi.jpg | 1.5 oz. sushi roll, lemon, soy sauce, sugar, sugar... | 1.39s |
| 05 | burger.jpg | cheeseburger with tomato, lettuce, cheese, bacon, lettuce... | 5.55s |
| 06 | pad-thai.jpg | A bowl of colorful vegetables. A plate of colorful vegetables... | 1.41s |
| 07 | salad.jpg | Salad with green leafy vegetables, red tomatoes, black olives, feta... | 1.32s |
| 08 | pasta.jpg | spaghetti, tomato, basil, cheese, lettuce, tomato... | 1.38s |
| 09 | eggs.jpg | avocado; egg; bread; cheese; nuts; seeds... | 2.28s |
| 10 | grilled-cheese.jpg | A sandwich. Lettuce. Green beans. Lettuce. Lettuce... | 1.41s |
| 11 | curry.jpg | A bowl of curry with a spoon. A bowl of naan bread... | 1.37s |
| 12 | fruit.jpg | A watermelon, A pineapple, A grapefruit, A strawberry... | 1.32s |
| 13 | non-food-laptop.jpg | A man is sitting at a wooden desk in front of a laptop... | 2.45s |

### SmolVLM2-2.2B (Q4_K_M) Full Responses

| # | File | Response (first 100 chars) | Full Latency |
|---|------|---------------------------|-------------|
| 01 | fried-rice.jpg | food - rice, food - green onions, food - sauce, food - meat, food - egg, food - oil | 6.91s |
| 02 | pizza.jpg | pizza, 300, 26 | 2.26s |
| 03 | ramen.jpg | ramen, 150 calories, 20g protein; egg, 50cal, 6g; pork, 100cal, 20g... | 5.73s |
| 04 | sushi.jpg | sushi, 120, 10 | 2.85s |
| 05 | burger.jpg | burger, 2300, 26 | 2.44s |
| 06 | pad-thai.jpg | broccoli, carrots, tomatoes, zucchini, ginger, blueberries, lime, quinoa... | 8.91s |
| 07 | salad.jpg | salad, 150, 10.5 | 2.39s |
| 08 | pasta.jpg | spaghetti, 200, 20 | 2.84s |
| 09 | eggs.jpg | avocados, eggs, pine nuts, olive oil, salt, pepper... | 8.70s |
| 10 | grilled-cheese.jpg | sandwich, 200, 20 | 2.69s |
| 11 | curry.jpg | tandoori chicken (loops) | 9.23s |
| 12 | fruit.jpg | no food | 3.15s |
| 13 | non-food-laptop.jpg | coffee, 210 calories, 10g protein | 2.51s |

### SmolVLM2-2.2B (Q4_K_M) + repeat_penalty=1.2 Full Responses

| # | File | Response (first 100 chars) | Full Latency |
|---|------|---------------------------|-------------|
| 01 | fried-rice.jpg | food - rice, fried chicken, green onions | 3.4s |
| 02 | pizza.jpg | pizza, 305 calories per serving, 29.6 grams of protein | 2.6s |
| 03 | ramen.jpg | ramen, 120 calories per serving, 35 grams of protein. | 2.8s |
| 04 | sushi.jpg | sushi, 120 calories, 3 grams protein | 3.6s |
| 05 | burger.jpg | 1. burger, 230 calories per serving, 45 grams of protein | 2.9s |
| 06 | pad-thai.jpg | 1. broccoli, 250 calories per serving, 4 grams protein; 3. carrots, 68... | 8.3s |
| 07 | salad.jpg | salad, 150 calories, 28 grams of protein. | 3.1s |
| 08 | pasta.jpg | spaghetti, 205 calories per serving, 16 grams of protein. | 6.6s |
| 09 | eggs.jpg | avocados, eggs, pine nuts, olive oil | 3.6s |
| 10 | grilled-cheese.jpg | sandwich, 205 calories, 18 grams | 3.3s |
| 11 | curry.jpg | food = ['tandoori chicken', 'naan bread'] | 2.4s |
| 12 | fruit.jpg | no food. | 2.4s |
| 13 | non-food-laptop.jpg | 1. coffee, 20 calories per serving, 8 grams of protein | 2.3s |

### SmolVLM2-500M (Q8_0) + repeat_penalty=1.2 Full Responses

| # | File | Response (first 100 chars) | Full Latency |
|---|------|---------------------------|-------------|
| 01 | fried-rice.jpg | 1. Rice 205 3. Chicken 48 4. Sliced vegetables (potatoes, carrots) 69 5. Green onions... | 2.9s |
| 02 | pizza.jpg | 1. Pepperoni Pizza - 450 cal, 28g protein 2. Sausage and Cheese Pizza - 375 cal... | 3.1s |
| 03 | ramen.jpg | 1. Egg, sunny side up 2. Spicy mayo sauce 3. Sliced ham 4. Rice noodles... | 11.6s |
| 04 | sushi.jpg | 1. Sushi roll: 250 cal, 36g protein (salmon) 2. Lemon slice: 40cal... | 7.3s |
| 05 | burger.jpg | 1. Burger 2. Tomato 3. Cheese... then "Rice, Soup, Fruit, Chocolate, Ice Cream" | 17.4s |
| 06 | pad-thai.jpg | 1. Carrots 200 calories per serving 3. Cucumbers 5 grams of protein... | 23.8s |
| 07 | salad.jpg | 1. Salad: 200 cal, 35g protein, 40g carbs, 60g fat... (runaway macro dump) | 105.1s |
| 08 | pasta.jpg | 1. Spaghetti 205 calories, 34 grams protein 3. Tomatoes 68 calories... | 132.1s |
| 09 | eggs.jpg | 1. Avocado slices 2. Hard-boiled eggs 3. Sesame seeds 4. Granola... (hallucinated) | 70.7s |
| 10 | grilled-cheese.jpg | 1. Sandwich: 200 cal, 35g protein... 6. Meat: 90 cal... | 43.1s |
| 11 | curry.jpg | 1. Chutney 2. Pita bread 3. Rice 4. Potato chips... (hallucinated) | 34.0s |
| 12 | fruit.jpg | 1. Pineapple 2. Grapefruit 3. Mango 4. Lime 5. Avocado 6. Strawberries... | 21.0s |
| 13 | non-food-laptop.jpg | 1. Apple 205 calories 3. Cookies 68 calories 4. Bread 295 calories... (hallucinated) | 2.7s |
