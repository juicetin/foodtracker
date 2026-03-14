"""Tests for parse_quantities module."""

import sys
from pathlib import Path

# Ensure knowledge-graph dir is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from parse_quantities import parse_quantity_grams


class TestBasicUnits:
    """Test standard unit conversions."""

    def test_cups_flour(self):
        result = parse_quantity_grams("2 cups flour")
        assert result is not None
        assert 220 <= result <= 260  # ~240g (120g/cup for flour)

    def test_pounds_chicken(self):
        result = parse_quantity_grams("1 lb chicken breast")
        assert result is not None
        assert 440 <= result <= 470  # ~454g

    def test_tablespoons_olive_oil(self):
        result = parse_quantity_grams("3 tablespoons olive oil")
        assert result is not None
        assert 38 <= result <= 48  # ~42g (14g/tbsp for oil)

    def test_teaspoon_salt(self):
        result = parse_quantity_grams("1/2 teaspoon salt")
        assert result is not None
        assert 2 <= result <= 4  # ~3g

    def test_ounces(self):
        result = parse_quantity_grams("8 oz cream cheese")
        assert result is not None
        assert 220 <= result <= 230  # 8 * 28.35 ~ 227

    def test_grams_direct(self):
        result = parse_quantity_grams("200 g sugar")
        assert result is not None
        assert result == pytest.approx(200.0, abs=1)

    def test_kg(self):
        result = parse_quantity_grams("1.5 kg potatoes")
        assert result is not None
        assert result == pytest.approx(1500.0, abs=1)

    def test_ml(self):
        result = parse_quantity_grams("250 ml milk")
        assert result is not None
        assert result == pytest.approx(250.0, abs=1)


class TestCountableItems:
    """Test countable item recognition."""

    def test_eggs(self):
        result = parse_quantity_grams("3 eggs")
        assert result is not None
        assert 140 <= result <= 160  # 3 * 50g

    def test_cloves_garlic(self):
        result = parse_quantity_grams("4 cloves garlic")
        assert result is not None
        assert 10 <= result <= 14  # 4 * 3g

    def test_medium_onion(self):
        result = parse_quantity_grams("1 medium onion")
        assert result is not None
        assert 100 <= result <= 120  # ~110g

    def test_slices_bread(self):
        result = parse_quantity_grams("2 slices bread")
        assert result is not None
        assert 50 <= result <= 70  # 2 * 30g


class TestFractions:
    """Test fraction parsing."""

    def test_slash_fraction(self):
        result = parse_quantity_grams("1/2 cup sugar")
        assert result is not None
        assert 95 <= result <= 105  # 0.5 * 200g

    def test_unicode_half(self):
        result = parse_quantity_grams("\u00bd cup sugar")
        assert result is not None
        assert 95 <= result <= 105

    def test_unicode_quarter(self):
        result = parse_quantity_grams("\u00bc teaspoon cinnamon")
        assert result is not None
        assert 1 <= result <= 2  # 0.25 * 5g

    def test_unicode_three_quarter(self):
        result = parse_quantity_grams("\u00be cup rice")
        assert result is not None
        assert 130 <= result <= 145  # 0.75 * 185g

    def test_mixed_number(self):
        result = parse_quantity_grams("1 1/2 cups flour")
        assert result is not None
        assert 170 <= result <= 190  # 1.5 * 120g


class TestRanges:
    """Test range parsing (takes midpoint)."""

    def test_range_cups(self):
        result = parse_quantity_grams("2-3 cups water")
        assert result is not None
        assert 580 <= result <= 620  # 2.5 * 240g

    def test_range_with_spaces(self):
        result = parse_quantity_grams("3 - 4 tablespoons butter")
        assert result is not None
        assert 49 <= result <= 55  # 3.5 * ~14g


class TestEdgeCases:
    """Test edge cases and special patterns."""

    def test_empty_string(self):
        assert parse_quantity_grams("") is None

    def test_to_taste(self):
        result = parse_quantity_grams("salt and pepper to taste")
        assert result is not None
        assert 0.5 <= result <= 2  # minimal fallback

    def test_pinch(self):
        result = parse_quantity_grams("a pinch of nutmeg")
        assert result is not None
        assert 0.5 <= result <= 2

    def test_dash(self):
        result = parse_quantity_grams("a dash of hot sauce")
        assert result is not None
        assert 0.5 <= result <= 2

    def test_unparseable_returns_none(self):
        """Completely unparseable text should return None."""
        result = parse_quantity_grams("some random text with no quantity")
        assert result is None

    def test_unit_abbreviations(self):
        result = parse_quantity_grams("2 tbsp soy sauce")
        assert result is not None
        assert 34 <= result <= 38  # 2 * 18g (soy sauce density override)

    def test_tsp_abbreviation(self):
        result = parse_quantity_grams("1 tsp vanilla extract")
        assert result is not None
        assert 4 <= result <= 6  # ~5g

    def test_lbs_abbreviation(self):
        result = parse_quantity_grams("2 lbs ground beef")
        assert result is not None
        assert 900 <= result <= 910  # 2 * 453.6g

    def test_cup_specific_flour(self):
        """Flour should use 120g/cup, not default 240g."""
        result = parse_quantity_grams("1 cup all-purpose flour")
        assert result is not None
        assert 115 <= result <= 130

    def test_cup_specific_sugar(self):
        """Sugar should use 200g/cup."""
        result = parse_quantity_grams("1 cup sugar")
        assert result is not None
        assert 195 <= result <= 210

    def test_cup_default_water(self):
        """Default cup is 240g (water-based)."""
        result = parse_quantity_grams("1 cup water")
        assert result is not None
        assert 235 <= result <= 245
