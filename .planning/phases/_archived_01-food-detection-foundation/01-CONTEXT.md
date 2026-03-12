# Phase 1: Food Detection Foundation - Context

**Gathered:** 2026-02-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Train/select a food detection model that accurately identifies food items, dish types, estimates portions, and infers hidden ingredients from photos. Includes a go/no-go decision point on YOLO vs on-device LLM. All inference must run on-device. No cloud API calls for detection.

</domain>

<decisions>
## Implementation Decisions

### Detection approach (YOLO vs LLM)
- **Target accuracy: >95% correct identification** — the bar is high; user wants to trust auto-logging with minimal review
- **No paid per-photo API calls** — cloud LLM inference (OpenAI, Google, etc.) is unacceptable at any cost per image
- **All inference must run on-device** — even if an LLM is used, it must be a small multimodal model running locally on the phone (e.g. PaliGemma, Florence-2, or similar)
- **YOLO is the preferred approach** for speed, reproducibility, and reliability
- **If YOLO hits 85% but LLM hits 97%, invest more in YOLO training first** — don't pivot prematurely
- **Hybrid approach acceptable**: traditional ML + on-device small LLM, using whichever returns higher confidence per detection (not strictly fallback-only)
- **Benchmark on both**: public datasets (Food-101, ISIA-500 for breadth) AND user's real food photos (for real-world validation)
- **Traditional ML preferred over LLM** for speed and reproducibility — LLM inference is slower, non-deterministic

### Training data & cuisine coverage
- **Priority cuisines: Australian/Western + East Asian** (Chinese, Japanese, Korean, Vietnamese, Thai) — these represent the user's daily eating
- **Category count: as many as possible** — combine multiple datasets (Food-101, ISIA-500, Roboflow food datasets, etc.) to maximize coverage
- **Best-guess for unknown foods** — model should guess the closest known dish rather than saying "unrecognised", let user correct

### Portion estimation
- **Target accuracy: +/-10% weight estimation** when visual cues are available
- **Smart prompting for reference objects** — if estimation confidence is low, suggest including a reference object (coin, credit card) next time. One-time tip, not nagging.
- **Smart fallback when visual cues are insufficient:**
  - If NO historical data exists for a food: use USDA standard serving size as default
  - If some historical data exists: extrapolate from the user's previous serving sizes for that food
  - The app learns portion patterns over time

### Hidden ingredient inference
- **Full recipe breakdown** — if the app identifies "fried rice", list all nutrition-significant ingredients (rice, egg, soy sauce, sesame oil, garlic, spring onion, cooking oil, etc.), not just major components
- **Data sources: all of these combined:**
  - Public recipe databases (RecipeNLG, Recipe1M+, etc.)
  - Nutrition DB implied ingredients (USDA/AFCD entries for prepared foods)
  - User corrections (builds personalised recipe knowledge over time)
- **Full knowledge graph** — structured dish→ingredient→nutrient relationships with variant tracking (e.g. "nasi goreng" is a variant of "fried rice" with different seasoning). User corrections refine edges. Implementation details (Neo4j vs relational) are Claude's discretion.

### Claude's Discretion
- Knowledge graph database technology choice (Neo4j, SQLite with graph queries, etc.)
- Exact YOLO model variant selection (v8n, v11n, YOLO26, etc.)
- On-device LLM model selection for hybrid approach (PaliGemma, Florence-2, etc.)
- Training pipeline implementation (Ultralytics, custom PyTorch, etc.)
- Model quantisation strategy (INT8, FP16, etc.) for mobile deployment
- Benchmark evaluation framework design

</decisions>

<specifics>
## Specific Ideas

- Existing `spike/food-detection-poc/` notebook provides the starting pipeline — YOLO detection → dish classification → ingredient inference → weight estimation → USDA lookup
- ADR-003 documents the original YOLO vs LLM decision (may need updating based on go/no-go outcome)
- The Gemini CLI conversation in project history has detailed info on YOLO training, export to CoreML/TFLite, and food-specific datasets
- Research flagged YOLO26 (Jan 2026) as 43% faster CPU inference with cleaner CoreML/TFLite export than YOLO11
- Research flagged react-native-fast-tflite v2.0.0 as the standard for on-device ML in React Native (JSI zero-copy, 40-60% faster than bridge-based)

</specifics>

<deferred>
## Deferred Ideas

### Cloud LLM via subscription (alternative detection approach — TOS concerns)
A potential alternative to on-device-only inference: use on-device binary classification (food/not-food) as a lightweight gate, then send confirmed food photos to a cloud LLM (e.g. Claude Sonnet) for detailed analysis. The concept:
- **On-device gate**: cheap, fast binary classifier filters non-food photos (~95% of gallery)
- **Cloud LLM for analysis**: structured tool calls returning JSON — dish name, ingredients list, macro/micro nutrients, portion estimate
- **Batch/background processing**: photos queued and processed asynchronously as they appear in gallery, not blocking user interaction
- **Parallel requests**: multiple photos processed concurrently for throughput
- **Cost model**: flat subscription (e.g. Claude Max) instead of per-photo API costs — economically attractive if subscription covers sufficient volume

**Why deferred:**
- Using a subscription (Max plan) for automated/programmatic API-style calls likely violates provider TOS (designed for interactive human use, not background batch processing)
- If a legitimate API-based approach becomes cost-effective in the future, this architecture is worth revisiting
- Documented here to keep the option visible without committing to it

**If revisited:** Would require a proper API plan with usage-based or tiered pricing that explicitly permits automated batch inference. The on-device gate + cloud LLM architecture itself is sound — the blocker is licensing, not technology.

</deferred>

---

*Phase: 01-food-detection-foundation*
*Context gathered: 2026-02-12*
