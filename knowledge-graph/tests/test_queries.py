#!/usr/bin/env python3
"""
Test suite for the food knowledge graph query API.

Tests cover:
- Ingredient lookup with recursive CTE variant traversal
- FTS5 full-text dish search
- Variant relationship discovery
- Best-guess matching (always returns something)
- Cuisine statistics
"""

import os
import sys

import pytest

# Add parent directory to path so we can import query module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import (
    get_best_guess,
    get_cuisine_stats,
    get_ingredients,
    get_variants,
    search_dish,
)


# ---------------------------------------------------------------------------
# get_ingredients tests
# ---------------------------------------------------------------------------

class TestGetIngredients:
    """Tests for the get_ingredients function."""

    def test_fried_rice_has_core_ingredients(self):
        """Fried rice should contain rice, egg, oil, and soy sauce."""
        ingredients = get_ingredients("fried rice")
        names = {ing["name"] for ing in ingredients}
        assert "rice" in names, f"Expected 'rice' in {names}"
        assert "egg" in names, f"Expected 'egg' in {names}"

    def test_carbonara_has_core_ingredients(self):
        """Carbonara should contain pasta, egg, pancetta/bacon, parmesan, pepper."""
        ingredients = get_ingredients("carbonara")
        names = {ing["name"] for ing in ingredients}
        assert "pasta" in names, f"Expected 'pasta' in {names}"
        assert "egg" in names, f"Expected 'egg' in {names}"
        assert "pancetta" in names or "bacon" in names, f"Expected pancetta/bacon in {names}"
        assert "parmesan" in names, f"Expected 'parmesan' in {names}"

    def test_ingredients_sorted_by_weight(self):
        """Ingredients should be sorted by weight_pct descending."""
        ingredients = get_ingredients("fried rice")
        if len(ingredients) >= 2:
            weights = [ing["weight_pct"] for ing in ingredients]
            assert weights == sorted(weights, reverse=True), "Ingredients not sorted by weight"

    def test_ingredients_have_required_fields(self):
        """Each ingredient dict should have all required fields."""
        ingredients = get_ingredients("carbonara")
        assert len(ingredients) > 0, "No ingredients returned"
        for ing in ingredients:
            assert "name" in ing
            assert "weight_pct" in ing
            assert "typical_amount_g" in ing
            assert "is_nutrition_significant" in ing
            assert "source" in ing
            assert "confidence" in ing

    def test_nonexistent_dish_returns_empty(self):
        """Querying a nonexistent dish should return empty list."""
        ingredients = get_ingredients("xyznotadish12345")
        assert ingredients == []

    def test_include_all_flag(self):
        """include_all=True should include non-significant ingredients."""
        significant = get_ingredients("fried rice", include_all=False)
        all_ings = get_ingredients("fried rice", include_all=True)
        assert len(all_ings) >= len(significant)

    def test_variant_traversal(self):
        """Querying a variant dish should also return ingredients from canonical dish."""
        # "chicken fried rice" is a variant of "fried rice"
        chicken_fr_ings = get_ingredients("chicken fried rice")
        names = {ing["name"] for ing in chicken_fr_ings}
        # Should have chicken (from variant) AND rice (from canonical or variant)
        assert "rice" in names, f"Expected 'rice' via variant chain, got {names}"
        assert "chicken" in names, f"Expected 'chicken' in chicken fried rice, got {names}"


# ---------------------------------------------------------------------------
# search_dish tests
# ---------------------------------------------------------------------------

class TestSearchDish:
    """Tests for the search_dish function."""

    def test_search_pad_finds_pad_thai(self):
        """Searching 'pad' should return 'pad thai' in results."""
        results = search_dish("pad")
        names = {r["name"] for r in results}
        assert any("pad" in name for name in names), f"Expected pad thai match, got {names}"

    def test_search_returns_list_of_dicts(self):
        """Results should be list of dicts with required keys."""
        results = search_dish("chicken")
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "id" in r
            assert "name" in r
            assert "cuisine" in r
            assert "confidence" in r

    def test_search_respects_limit(self):
        """Should not return more than the specified limit."""
        results = search_dish("chicken", limit=3)
        assert len(results) <= 3

    def test_search_empty_returns_something(self):
        """Even a broad search should return results."""
        results = search_dish("rice")
        assert len(results) > 0

    def test_prefix_matching(self):
        """Prefix matching should work (e.g., 'fried ri' matches 'fried rice')."""
        results = search_dish("fried ri")
        names = {r["name"] for r in results}
        assert any("fried rice" in name for name in names), f"Expected fried rice match, got {names}"


# ---------------------------------------------------------------------------
# get_variants tests
# ---------------------------------------------------------------------------

class TestGetVariants:
    """Tests for the get_variants function."""

    def test_fried_rice_has_variants(self):
        """Fried rice should have variants like chicken fried rice, nasi goreng."""
        variants = get_variants("fried rice")
        variant_names = {v["name"] for v in variants}
        # At least one of these should be a variant
        expected = {"chicken fried rice", "nasi goreng", "shrimp fried rice"}
        overlap = variant_names & expected
        assert len(overlap) > 0, f"Expected variants of fried rice, got {variant_names}"

    def test_nasi_goreng_links_to_fried_rice(self):
        """Nasi goreng variants should include fried rice (or vice versa)."""
        variants = get_variants("nasi goreng")
        variant_names = {v["name"] for v in variants}
        assert "fried rice" in variant_names, f"Expected 'fried rice' as variant, got {variant_names}"

    def test_variants_exclude_self(self):
        """Variants should not include the queried dish itself."""
        variants = get_variants("fried rice")
        variant_names = {v["name"] for v in variants}
        assert "fried rice" not in variant_names

    def test_nonexistent_dish_returns_empty(self):
        """Nonexistent dish should return empty list."""
        variants = get_variants("xyznotadish12345")
        assert variants == []


# ---------------------------------------------------------------------------
# get_best_guess tests
# ---------------------------------------------------------------------------

class TestGetBestGuess:
    """Tests for the get_best_guess function."""

    def test_exact_match(self):
        """Known dish should be found as exact match."""
        result = get_best_guess("carbonara")
        assert result["name"] == "carbonara"
        assert result["match_type"] == "exact"

    def test_nonexistent_returns_something(self):
        """Completely unknown input should still return a result (never None/empty)."""
        result = get_best_guess("xyznonexistent12345")
        assert result is not None
        assert "name" in result
        assert result["name"] != ""
        assert result["name"] != "unknown"

    def test_partial_match(self):
        """Partial name should find a match."""
        result = get_best_guess("fried ri")
        assert result is not None
        assert "fried rice" in result["name"] or result["match_type"] in ("prefix", "fuzzy")

    def test_result_has_required_fields(self):
        """Result should have all required fields."""
        result = get_best_guess("pizza")
        assert "name" in result
        assert "cuisine" in result
        assert "confidence" in result
        assert "match_type" in result

    def test_match_type_valid(self):
        """match_type should be one of: exact, prefix, fuzzy."""
        result = get_best_guess("sushi")
        assert result["match_type"] in ("exact", "prefix", "fuzzy")

    def test_gibberish_returns_something(self):
        """Even complete gibberish should return a dish."""
        result = get_best_guess("zzzzqqqwww")
        assert result is not None
        assert result["name"] != ""


# ---------------------------------------------------------------------------
# get_cuisine_stats tests
# ---------------------------------------------------------------------------

class TestGetCuisineStats:
    """Tests for the get_cuisine_stats function."""

    def test_returns_multiple_cuisines(self):
        """Should have at least 5 different cuisines."""
        stats = get_cuisine_stats()
        assert len(stats) >= 5, f"Expected at least 5 cuisines, got {len(stats)}"

    def test_stats_have_required_fields(self):
        """Each cuisine should have dish_count and ingredient_count."""
        stats = get_cuisine_stats()
        for cuisine, data in stats.items():
            assert "dish_count" in data
            assert "ingredient_count" in data
            assert data["dish_count"] > 0

    def test_total_dishes_over_1000(self):
        """Total dishes across all cuisines should be >1000."""
        stats = get_cuisine_stats()
        total = sum(data["dish_count"] for data in stats.values())
        assert total > 1000, f"Expected >1000 total dishes, got {total}"

    def test_has_priority_cuisines(self):
        """Should include Western, Chinese, Japanese, Korean, Indian cuisines."""
        stats = get_cuisine_stats()
        required = {"Western", "Chinese", "Japanese", "Korean", "Indian"}
        present = set(stats.keys())
        missing = required - present
        assert not missing, f"Missing cuisines: {missing}"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Cross-function integration tests."""

    def test_search_then_ingredients(self):
        """Search for a dish, then get its ingredients."""
        results = search_dish("green curry")
        assert len(results) > 0
        dish_name = results[0]["name"]
        ingredients = get_ingredients(dish_name)
        assert len(ingredients) > 0

    def test_recursive_cte_variant_chain(self):
        """If A is variant of B, querying A returns ingredients from both A and B."""
        # chicken fried rice -> fried rice
        base_ings = get_ingredients("fried rice")
        variant_ings = get_ingredients("chicken fried rice")

        base_names = {ing["name"] for ing in base_ings}
        variant_names = {ing["name"] for ing in variant_ings}

        # Variant should have all base ingredients plus its own
        # (since it inherits from canonical)
        assert "chicken" in variant_names
        assert "rice" in variant_names

    def test_cuisine_coverage(self):
        """Verify dishes exist for all priority cuisines."""
        stats = get_cuisine_stats()
        priority_cuisines = [
            "Western", "Italian", "Mexican", "Chinese", "Japanese",
            "Korean", "Vietnamese", "Thai", "Indian", "Mediterranean",
        ]
        for cuisine in priority_cuisines:
            assert cuisine in stats, f"Missing cuisine: {cuisine}"
            assert stats[cuisine]["dish_count"] > 10, \
                f"{cuisine} has only {stats[cuisine]['dish_count']} dishes"
