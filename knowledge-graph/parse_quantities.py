#!/usr/bin/env python3
"""
Ingredient quantity parser for recipe text.

Extracts numeric amounts and unit conversions from raw ingredient strings
like "2 cups flour" or "1/2 teaspoon salt", returning the estimated weight
in grams.

Designed for batch processing of 500K+ recipes at build time.
Aims for 70-80% accuracy on common patterns; caller provides fallback.
"""

import re
from typing import Optional

# --------------------------------------------------------------------------- #
# Unicode fraction map
# --------------------------------------------------------------------------- #
_UNICODE_FRACTIONS = {
    "\u00bc": 0.25,  # 1/4
    "\u00bd": 0.5,   # 1/2
    "\u00be": 0.75,  # 3/4
    "\u2150": 1 / 7,
    "\u2151": 1 / 9,
    "\u2152": 1 / 10,
    "\u2153": 1 / 3,
    "\u2154": 2 / 3,
    "\u2155": 1 / 5,
    "\u2156": 2 / 5,
    "\u2157": 3 / 5,
    "\u2158": 4 / 5,
    "\u2159": 1 / 6,
    "\u215a": 5 / 6,
    "\u215b": 1 / 8,
    "\u215c": 3 / 8,
    "\u215d": 5 / 8,
    "\u215e": 7 / 8,
}

# --------------------------------------------------------------------------- #
# Unit-to-grams (default for water/liquid; ingredient overrides below)
# --------------------------------------------------------------------------- #
_UNIT_GRAMS_DEFAULT = {
    # Volume
    "cup": 240.0,
    "tablespoon": 15.0,
    "teaspoon": 5.0,
    "quart": 960.0,
    "pint": 480.0,
    "gallon": 3840.0,
    "liter": 1000.0,
    "litre": 1000.0,
    "ml": 1.0,
    "milliliter": 1.0,
    "millilitre": 1.0,
    "fl oz": 30.0,
    # Weight
    "oz": 28.35,
    "ounce": 28.35,
    "lb": 453.6,
    "pound": 453.6,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "g": 1.0,
    "gram": 1.0,
    # Kitchen items
    "stick": 113.0,   # stick of butter
    "can": 400.0,     # standard 14oz can
    "jar": 450.0,
    "package": 225.0,
    "pkg": 225.0,
    "box": 340.0,
    "bottle": 500.0,
    "bag": 340.0,
}

# Unit aliases -> canonical name
_UNIT_ALIASES = {
    "cups": "cup",
    "c.": "cup",
    "c": "cup",
    "tablespoons": "tablespoon",
    "tbsp": "tablespoon",
    "tbsp.": "tablespoon",
    "tbs": "tablespoon",
    "tbs.": "tablespoon",
    "T": "tablespoon",
    "teaspoons": "teaspoon",
    "tsp": "teaspoon",
    "tsp.": "teaspoon",
    "t": "teaspoon",
    "quarts": "quart",
    "qt": "quart",
    "qt.": "quart",
    "pints": "pint",
    "pt": "pint",
    "pt.": "pint",
    "gallons": "gallon",
    "gal": "gallon",
    "gal.": "gallon",
    "liters": "liter",
    "litres": "litre",
    "l": "liter",
    "l.": "liter",
    "milliliters": "milliliter",
    "millilitres": "millilitre",
    "ounces": "ounce",
    "oz.": "oz",
    "pounds": "pound",
    "lbs": "lb",
    "lbs.": "lb",
    "lb.": "lb",
    "kilograms": "kilogram",
    "kg.": "kg",
    "grams": "gram",
    "g.": "g",
    "sticks": "stick",
    "cans": "can",
    "jars": "jar",
    "packages": "package",
    "pkgs": "pkg",
    "pkg.": "pkg",
    "boxes": "box",
    "bottles": "bottle",
    "bags": "bag",
    "fl oz": "fl oz",
    "fluid ounce": "fl oz",
    "fluid ounces": "fl oz",
}

# Per-ingredient cup overrides (grams per 1 cup)
_CUP_OVERRIDES = {
    "flour": 120.0,
    "all-purpose flour": 120.0,
    "all purpose flour": 120.0,
    "bread flour": 120.0,
    "cake flour": 115.0,
    "whole wheat flour": 120.0,
    "sugar": 200.0,
    "granulated sugar": 200.0,
    "white sugar": 200.0,
    "brown sugar": 220.0,
    "powdered sugar": 120.0,
    "confectioners sugar": 120.0,
    "icing sugar": 120.0,
    "rice": 185.0,
    "uncooked rice": 185.0,
    "white rice": 185.0,
    "brown rice": 190.0,
    "oil": 218.0,
    "olive oil": 218.0,
    "vegetable oil": 218.0,
    "canola oil": 218.0,
    "coconut oil": 218.0,
    "butter": 227.0,
    "unsalted butter": 227.0,
    "salted butter": 227.0,
    "honey": 340.0,
    "maple syrup": 315.0,
    "oats": 80.0,
    "rolled oats": 80.0,
    "cocoa powder": 85.0,
    "cornstarch": 128.0,
    "corn starch": 128.0,
    "milk": 244.0,
    "cream": 240.0,
    "heavy cream": 240.0,
    "sour cream": 230.0,
    "yogurt": 245.0,
    "peanut butter": 258.0,
    "cheese": 113.0,
    "shredded cheese": 113.0,
}

# Per-ingredient tablespoon overrides (grams per 1 tbsp)
_TBSP_OVERRIDES = {
    "butter": 14.2,
    "unsalted butter": 14.2,
    "oil": 14.0,
    "olive oil": 14.0,
    "vegetable oil": 14.0,
    "canola oil": 14.0,
    "coconut oil": 14.0,
    "honey": 21.0,
    "maple syrup": 20.0,
    "sugar": 12.5,
    "flour": 8.0,
    "cornstarch": 8.0,
    "corn starch": 8.0,
    "cocoa powder": 5.0,
    "soy sauce": 18.0,
}

# --------------------------------------------------------------------------- #
# Countable items (grams per 1 item)
# --------------------------------------------------------------------------- #
_COUNT_ITEM_GRAMS = {
    "egg": 50.0,
    "eggs": 50.0,
    "clove garlic": 3.0,
    "cloves garlic": 3.0,
    "garlic clove": 3.0,
    "garlic cloves": 3.0,
    "slice bread": 30.0,
    "slices bread": 30.0,
    "slice": 30.0,
    "slices": 30.0,
    "medium onion": 110.0,
    "onion": 110.0,
    "onions": 110.0,
    "medium potato": 150.0,
    "potato": 150.0,
    "potatoes": 150.0,
    "medium tomato": 123.0,
    "tomato": 123.0,
    "tomatoes": 123.0,
    "medium apple": 182.0,
    "apple": 182.0,
    "apples": 182.0,
    "medium banana": 118.0,
    "banana": 118.0,
    "bananas": 118.0,
    "stalk celery": 40.0,
    "stalks celery": 40.0,
    "celery stalk": 40.0,
    "celery stalks": 40.0,
    "sprig": 2.0,
    "sprigs": 2.0,
    "head garlic": 50.0,
    "heads garlic": 50.0,
    "bunch": 30.0,
    "bunches": 30.0,
    "lemon": 85.0,
    "lemons": 85.0,
    "lime": 67.0,
    "limes": 67.0,
    "orange": 131.0,
    "oranges": 131.0,
    "avocado": 200.0,
    "avocados": 200.0,
    "carrot": 72.0,
    "carrots": 72.0,
    "bell pepper": 120.0,
    "bell peppers": 120.0,
    "pepper": 120.0,
    "peppers": 120.0,
    "chicken breast": 174.0,
    "chicken breasts": 174.0,
    "tortilla": 50.0,
    "tortillas": 50.0,
}

# --------------------------------------------------------------------------- #
# Seasonings / "to taste" patterns
# --------------------------------------------------------------------------- #
_TASTE_RE = re.compile(
    r'\b(to taste|pinch|dash|for garnish|as needed|as desired|optional)\b',
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Amount extraction
# --------------------------------------------------------------------------- #

# Match numbers: "2", "2.5", "1/2", "1 1/2", unicode fractions, ranges "2-3"
_NUM_PATTERN = (
    r'(?:'
    r'(\d+)\s*[-\u2013]\s*(\d+)'           # range: 2-3
    r'|(\d+)\s+(\d+)\s*/\s*(\d+)'          # mixed: 1 1/2
    r'|(\d+)\s*/\s*(\d+)'                  # fraction: 1/2
    r'|(\d+\.?\d*)'                         # decimal: 2.5 or 2
    r'|([' + ''.join(_UNICODE_FRACTIONS.keys()) + r'])'  # unicode fraction
    r')'
)

_NUM_RE = re.compile(_NUM_PATTERN)

# Unit pattern -- match unit at start of remaining string (after number)
_ALL_UNITS = sorted(
    list(_UNIT_GRAMS_DEFAULT.keys()) + list(_UNIT_ALIASES.keys()),
    key=len,
    reverse=True,  # longest first to avoid partial matches
)
# Escape dots for regex
_UNIT_PATTERN = r'(?:' + '|'.join(re.escape(u) for u in _ALL_UNITS) + r')\b\.?'
_UNIT_RE = re.compile(_UNIT_PATTERN, re.IGNORECASE)


def _parse_number(s: str) -> tuple[Optional[float], str]:
    """
    Extract the first numeric value from a string.

    Returns (value, remainder) where remainder is the string after the number.
    Returns (None, s) if no number found.
    """
    # First, replace unicode fractions with their values
    for ufrac, val in _UNICODE_FRACTIONS.items():
        if ufrac in s:
            idx = s.index(ufrac)
            before = s[:idx].strip()
            after = s[idx + 1:].strip()

            # Check if there's a whole number before the fraction
            whole = 0.0
            if before:
                try:
                    whole = float(before)
                except ValueError:
                    pass

            return whole + val, after

    m = _NUM_RE.match(s.strip())
    if not m:
        return None, s

    remainder = s[m.end():].strip()

    if m.group(1) is not None and m.group(2) is not None:
        # Range: take midpoint
        lo = float(m.group(1))
        hi = float(m.group(2))
        return (lo + hi) / 2, remainder

    if m.group(3) is not None and m.group(4) is not None and m.group(5) is not None:
        # Mixed: whole + fraction
        whole = float(m.group(3))
        num = float(m.group(4))
        den = float(m.group(5))
        return whole + (num / den) if den != 0 else whole, remainder

    if m.group(6) is not None and m.group(7) is not None:
        # Simple fraction
        num = float(m.group(6))
        den = float(m.group(7))
        return (num / den) if den != 0 else None, remainder

    if m.group(8) is not None:
        return float(m.group(8)), remainder

    if m.group(9) is not None:
        return _UNICODE_FRACTIONS.get(m.group(9)), remainder

    return None, s


def _resolve_unit(unit_str: str) -> str:
    """Resolve a unit string to its canonical form."""
    lower = unit_str.lower().rstrip(".")
    # Direct match
    if lower in _UNIT_GRAMS_DEFAULT:
        return lower
    # Alias match
    if lower in _UNIT_ALIASES:
        return _UNIT_ALIASES[lower]
    # With period
    if unit_str in _UNIT_ALIASES:
        return _UNIT_ALIASES[unit_str]
    return lower


def _extract_ingredient_name(remainder: str) -> str:
    """Extract the ingredient name from the remaining string after amount+unit."""
    s = remainder.strip()
    # Remove leading "of "
    s = re.sub(r'^of\s+', '', s, flags=re.IGNORECASE)
    # Remove parenthetical notes
    s = re.sub(r'\s*\(.*?\)', '', s)
    # Remove trailing comma and everything after
    s = re.sub(r',.*$', '', s)
    # Remove common modifiers for matching
    s = re.sub(
        r'\b(chopped|diced|minced|sliced|grated|shredded|crushed|'
        r'fresh|dried|frozen|canned|cooked|raw|melted|softened|'
        r'finely|thinly|thickly|roughly|peeled|seeded|deveined|'
        r'boneless|skinless|lean|extra|self-rising|'
        r'granulated|powdered|confectioners|'
        r'divided|optional|or more|for garnish|if desired)\b',
        '', s, flags=re.IGNORECASE,
    )
    return re.sub(r'\s+', ' ', s).strip().lower()


def _get_cup_grams(ingredient_name: str) -> float:
    """Get grams-per-cup for a specific ingredient, or default."""
    lower = ingredient_name.lower()
    # Check exact match first
    if lower in _CUP_OVERRIDES:
        return _CUP_OVERRIDES[lower]
    # Check if any override key is a substring of the ingredient
    for key, val in _CUP_OVERRIDES.items():
        if key in lower:
            return val
    return _UNIT_GRAMS_DEFAULT["cup"]  # 240g default


def _get_tbsp_grams(ingredient_name: str) -> float:
    """Get grams-per-tablespoon for a specific ingredient, or default."""
    lower = ingredient_name.lower()
    if lower in _TBSP_OVERRIDES:
        return _TBSP_OVERRIDES[lower]
    for key, val in _TBSP_OVERRIDES.items():
        if key in lower:
            return val
    return _UNIT_GRAMS_DEFAULT["tablespoon"]  # 15g default


def _check_countable(amount: float, remainder: str) -> Optional[float]:
    """Check if the remainder matches a countable item."""
    lower = remainder.lower().strip()
    # Remove modifiers for matching
    clean = re.sub(
        r'\b(large|medium|small|whole|fresh|ripe|green|red|yellow|'
        r'chopped|diced|minced|sliced|peeled|boneless|skinless)\b',
        '', lower, flags=re.IGNORECASE,
    ).strip()
    clean = re.sub(r'\s+', ' ', clean)

    # Check direct match
    if clean in _COUNT_ITEM_GRAMS:
        return amount * _COUNT_ITEM_GRAMS[clean]

    # Check if any count item key is in the cleaned string
    for key, grams in _COUNT_ITEM_GRAMS.items():
        if key in clean:
            return amount * grams

    return None


def parse_quantity_grams(ingredient_str: str) -> Optional[float]:
    """
    Parse an ingredient string and return estimated weight in grams.

    Examples:
        parse_quantity_grams("2 cups flour")       -> ~240.0
        parse_quantity_grams("1 lb chicken breast") -> ~454.0
        parse_quantity_grams("3 eggs")              -> ~150.0
        parse_quantity_grams("")                    -> None

    Returns None when the string cannot be parsed at all.
    Caller should provide a fallback (e.g. 50g) for None results.
    """
    if not ingredient_str or not ingredient_str.strip():
        return None

    s = ingredient_str.strip()

    # Check for "to taste" / "pinch" / "dash" patterns first
    if _TASTE_RE.search(s):
        return 1.0

    # 1. Extract the numeric amount
    amount, remainder = _parse_number(s)

    if amount is None:
        # No number found at all -- can't parse
        return None

    if amount <= 0:
        return None

    # 2. Try to find a unit in the remainder
    remainder_stripped = remainder.strip()
    unit_match = _UNIT_RE.match(remainder_stripped)

    if unit_match:
        raw_unit = unit_match.group(0).rstrip(".")
        canonical_unit = _resolve_unit(raw_unit)
        after_unit = remainder_stripped[unit_match.end():].strip()
        ingredient_name = _extract_ingredient_name(after_unit)

        # Get unit conversion with ingredient-specific overrides
        if canonical_unit in ("cup",):
            grams_per_unit = _get_cup_grams(ingredient_name)
        elif canonical_unit in ("tablespoon",):
            grams_per_unit = _get_tbsp_grams(ingredient_name)
        elif canonical_unit in _UNIT_GRAMS_DEFAULT:
            grams_per_unit = _UNIT_GRAMS_DEFAULT[canonical_unit]
        elif canonical_unit in _UNIT_ALIASES:
            resolved = _UNIT_ALIASES[canonical_unit]
            grams_per_unit = _UNIT_GRAMS_DEFAULT.get(resolved, None)
            if grams_per_unit is None:
                return None
        else:
            # Unknown unit after resolution
            return None

        return round(amount * grams_per_unit, 1)

    # 3. No unit found -- check for countable items
    countable_result = _check_countable(amount, remainder_stripped)
    if countable_result is not None:
        return round(countable_result, 1)

    # 4. Nothing matched -- return None (caller provides fallback)
    return None
