#!/usr/bin/env python3
"""
Prompt grid search — test multiple prompt variants against labeled data
on Gemini Nano (via Chrome Built-in AI or adb device).

Usage:
    python grid_search.py --chrome               # All variants via Chrome
    python grid_search.py --chrome --variants 0 3 # Specific variants
    python grid_search.py --chrome --prompt prompts/v4-step-by-step.json  # Single versioned prompt
    python grid_search.py --serial 48181FDAP00A1U  # Via adb device
"""

import argparse
import json
import sys
import time
from pathlib import Path

from food_metric import score_detailed

DATASET_DIR = Path(__file__).parent / "dataset"
PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Prompt variants to test
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = [
    # 0: Current production prompt
    {
        "name": "production (current)",
        "prompt": (
            'Identify all food in this image. Return only valid JSON — no extra text:\n'
            '{"dishes":[{"name":string,"cuisine":string,"recipe_name":string,'
            '"ingredients":[{"name":string,"amount_g":number}]}]}\n'
            'recipe_name: a concise human-friendly name for the dish as a recipe '
            '(e.g. "Chicken Stir Fry with Vegetables"). '
            'Estimate amount_g using surrounding objects (plates, cutlery, cups, hands) '
            'as size references; fall back to a typical restaurant serving size if no '
            'reference objects are visible. '
            'Be specific with ingredient names (e.g. "basmati rice" not "rice").'
        ),
    },
    # 1: Minimal — less instruction, more room for output
    {
        "name": "minimal",
        "prompt": (
            'What food is in this photo? Return JSON:\n'
            '{"dishes":[{"name":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n'
            'Estimate weights in grams. Be specific with names.'
        ),
    },
    # 2: Structured with explicit examples
    {
        "name": "with-example",
        "prompt": (
            'Identify food in image. Return valid JSON only:\n'
            '{"dishes":[{"name":"dish","cuisine":"type","ingredients":[{"name":"item","amount_g":100}]}]}\n'
            'Example: {"dishes":[{"name":"fried rice","cuisine":"Chinese","ingredients":'
            '[{"name":"jasmine rice","amount_g":250},{"name":"egg","amount_g":50}]}]}\n'
            'Estimate grams using plate size as reference (~25cm diameter).'
        ),
    },
    # 3: Chain-of-thought style
    {
        "name": "step-by-step",
        "prompt": (
            'Look at this food photo. First identify each dish, then list ingredients with gram estimates.\n'
            'Return only valid JSON:\n'
            '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n'
            'For weights: a standard dinner plate is ~25cm, a cup is ~240ml, a fist-sized portion is ~150g.'
        ),
    },
    # 4: Emphasize accuracy over completeness
    {
        "name": "accuracy-focus",
        "prompt": (
            'Identify the main food items visible in this image. Only include ingredients you are '
            'confident about. Return valid JSON:\n'
            '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n'
            'For amount_g: use the plate (~25cm) and utensils as scale references. '
            'Use specific ingredient names (e.g. "jasmine rice" not "rice", "chicken thigh" not "chicken").'
        ),
    },
    # 5: Compact schema — saves output tokens
    {
        "name": "compact-schema",
        "prompt": (
            'Food in image → JSON:\n'
            '{"d":[{"n":string,"c":string,"i":[{"n":string,"g":number}]}]}\n'
            'd=dishes, n=name, c=cuisine, i=ingredients, g=grams.\n'
            'Estimate grams from plate/utensil size. Be specific with ingredient names.'
        ),
    },
    # 6: Role-based
    {
        "name": "nutritionist-role",
        "prompt": (
            'You are a nutritionist analyzing a meal photo. Identify each dish and '
            'estimate ingredient weights for calorie tracking. Return only valid JSON:\n'
            '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n'
            'Use surrounding objects for scale. Be precise with ingredient names and amounts.'
        ),
    },
    # 7: Explicit weight anchors
    {
        "name": "weight-anchors",
        "prompt": (
            'Identify food in this image. Return valid JSON:\n'
            '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}\n'
            'Weight guide: rice portion ~200g, meat portion ~150g, vegetables ~80g, '
            'sauce ~20g, oil ~10g. Adjust based on what you see. '
            'Be specific (e.g. "basmati rice" not "rice").'
        ),
    },
]


def load_dataset() -> list[dict]:
    """Load labeled examples."""
    examples = []
    for lf in sorted(DATASET_DIR.glob("*.json")):
        if lf.name == "EXAMPLE.json":
            continue
        data = json.loads(lf.read_text())
        img = DATASET_DIR / data["image"]
        if not img.exists():
            print(f"  SKIP: {lf.name} — image not found: {img}")
            continue
        examples.append({"label_file": lf.name, "image": str(img), "dishes": data["dishes"]})
    return examples


def run_variant(lm: GeminiNanoLM, variant: dict, examples: list[dict]) -> list[dict]:
    """Run a single prompt variant across all examples."""
    import base64
    results = []

    for ex in examples:
        # Build messages with image
        img_bytes = Path(ex["image"]).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(ex["image"]).suffix
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": variant["prompt"]},
                ],
            }
        ]

        try:
            output = lm(messages=messages)
            raw = output[0] if output else ""
        except Exception as e:
            raw = f"ERROR:{e}"

        detail = score_detailed(raw, ex["dishes"])
        detail["label_file"] = ex["label_file"]
        results.append(detail)

    return results


def load_versioned_prompts(paths: list[str]) -> list[dict]:
    """Load prompt variants from versioned JSON files."""
    variants = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = PROMPTS_DIR / path
        data = json.loads(path.read_text())
        variants.append({
            "name": f"{data['version']}-{data['name']}",
            "prompt": data["food_prompt"],
            "version_file": str(path.name),
        })
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="48181FDAP00A1U")
    parser.add_argument("--variants", nargs="*", type=int, help="Which variants to test (indices)")
    parser.add_argument("--chrome", action="store_true", help="Use Chrome Built-in AI instead of adb")
    parser.add_argument("--prompt", nargs="*", help="Versioned prompt JSON files to test (from prompts/)")
    args = parser.parse_args()

    examples = load_dataset()
    if not examples:
        print(f"No labeled examples in {DATASET_DIR}/")
        print("Create JSON label files (see EXAMPLE.json for format)")
        sys.exit(1)

    print(f"Loaded {len(examples)} labeled examples")

    if args.chrome:
        from gemini_nano_chrome_lm import GeminiNanoChromeLM
        lm = GeminiNanoChromeLM()
        if not lm.ensure_model_ready():
            print("Chrome Gemini Nano model not ready")
            sys.exit(1)
        print("Using Chrome Built-in AI")
    else:
        from gemini_nano_lm import GeminiNanoLM
        lm = GeminiNanoLM(serial=args.serial, cooldown=5.0, retry_backoff=60.0, max_retries=2)
        print(f"Using adb device: {args.serial}")

    # Determine which prompts to test
    if args.prompt:
        variants_to_test = load_versioned_prompts(args.prompt)
    else:
        variant_indices = args.variants or list(range(len(PROMPT_VARIANTS)))
        variants_to_test = []
        for idx in variant_indices:
            if idx >= len(PROMPT_VARIANTS):
                print(f"Variant {idx} out of range (0-{len(PROMPT_VARIANTS)-1})")
                continue
            v = PROMPT_VARIANTS[idx].copy()
            v["name"] = f"{idx}:{v['name']}"
            variants_to_test.append(v)

    all_results = {}

    for variant in variants_to_test:
        print(f"\n{'='*60}")
        print(f"Variant: {variant['name']}")
        print(f"{'='*60}")
        print(f"Prompt: {variant['prompt'][:100]}...")

        start = time.time()
        results = run_variant(lm, variant, examples)
        elapsed = time.time() - start

        # Aggregate
        n = len(results)
        avg = lambda k: sum(r[k] for r in results) / n if n else 0

        print(f"\n  Results ({elapsed:.1f}s, {elapsed/max(n,1):.1f}s/image):")
        print(f"  Composite:   {avg('composite'):.3f}")
        print(f"  Dish name:   {avg('dish_name_f1'):.3f}")
        print(f"  Ing recall:  {avg('ingredient_recall'):.3f}")
        print(f"  Ing prec:    {avg('ingredient_precision'):.3f}")
        print(f"  Weight MAE:  {avg('weight_mae_score'):.3f}")
        print(f"  JSON parse:  {sum(1 for r in results if r['json_parsed'])}/{n}")

        all_results[variant["name"]] = {
            "composite": round(avg("composite"), 3),
            "dish_name_f1": round(avg("dish_name_f1"), 3),
            "ingredient_recall": round(avg("ingredient_recall"), 3),
            "ingredient_precision": round(avg("ingredient_precision"), 3),
            "weight_mae_score": round(avg("weight_mae_score"), 3),
            "json_parse_rate": sum(1 for r in results if r["json_parsed"]) / max(n, 1),
            "avg_latency": round(elapsed / max(n, 1), 1),
            "per_example": results,
        }

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Variant':<30} {'Comp':>6} {'Name':>6} {'Recall':>7} {'Prec':>6} {'Weight':>7} {'Parse':>6} {'Lat':>5}")
    print(f"{'='*80}")

    for key, data in sorted(all_results.items(), key=lambda x: -x[1]["composite"]):
        print(
            f"{key:<30} "
            f"{data['composite']:>6.3f} "
            f"{data['dish_name_f1']:>6.3f} "
            f"{data['ingredient_recall']:>7.3f} "
            f"{data['ingredient_precision']:>6.3f} "
            f"{data['weight_mae_score']:>7.3f} "
            f"{data['json_parse_rate']:>6.1%} "
            f"{data['avg_latency']:>4.1f}s"
        )

    # Save full results
    output_path = Path(__file__).parent / "grid_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nFull results saved to {output_path}")
    print(f"Stats: {lm.stats()}")

    if hasattr(lm, "close"):
        lm.close()


if __name__ == "__main__":
    main()
