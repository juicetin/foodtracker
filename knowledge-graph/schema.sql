-- Food Knowledge Graph Schema (v2 - Hierarchical)
-- SQLite database for cuisine -> dish -> recipe -> ingredient -> nutrition
-- with multilingual aliases, SymSpell fuzzy matching, and FTS5 search

-- ── Core Hierarchy ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cuisine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    region TEXT
);

CREATE TABLE IF NOT EXISTS dish_category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cuisine_id INTEGER REFERENCES cuisine(id),
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES dish_category(id)
);

CREATE TABLE IF NOT EXISTS dish (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER REFERENCES dish_category(id),
    canonical_name TEXT UNIQUE NOT NULL,
    description TEXT,
    avg_calories_per_serving REAL,
    avg_protein_per_serving REAL,
    avg_carbs_per_serving REAL,
    avg_fat_per_serving REAL,
    default_serving_grams REAL,
    source TEXT NOT NULL DEFAULT 'generated',
    confidence REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS dish_alias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dish_id INTEGER NOT NULL REFERENCES dish(id),
    alias TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    alias_type TEXT DEFAULT 'spelling'
);

-- ── Recipes & Ingredients ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS recipe (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dish_id INTEGER NOT NULL REFERENCES dish(id),
    name TEXT,
    source TEXT DEFAULT 'generated',
    total_weight_grams REAL,
    servings INTEGER DEFAULT 1,
    is_canonical INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recipe_ingredient (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL REFERENCES recipe(id),
    usda_fdc_id INTEGER,
    ingredient_name TEXT NOT NULL,
    quantity_grams REAL,
    sort_order INTEGER DEFAULT 0
);

-- ── USDA Nutrition ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS usda_food (
    fdc_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    food_group TEXT,
    calories_per_100g REAL,
    protein_per_100g REAL,
    fat_per_100g REAL,
    carbs_per_100g REAL,
    fiber_per_100g REAL,
    vitamin_a_ug REAL,
    vitamin_c_mg REAL,
    vitamin_d_ug REAL,
    calcium_mg REAL,
    iron_mg REAL,
    potassium_mg REAL,
    sodium_mg REAL,
    zinc_mg REAL,
    magnesium_mg REAL
);

-- ── SymSpell Fuzzy Matching ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS symspell_deletes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dish_id INTEGER REFERENCES dish(id),
    delete_variant TEXT NOT NULL
);

-- ── Indexes ──────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_dish_category_cuisine ON dish_category(cuisine_id);
CREATE INDEX IF NOT EXISTS idx_dish_category_parent ON dish_category(parent_id);
CREATE INDEX IF NOT EXISTS idx_dish_category_id ON dish(category_id);
CREATE INDEX IF NOT EXISTS idx_dish_canonical_name ON dish(canonical_name);
CREATE INDEX IF NOT EXISTS idx_dish_alias_dish_id ON dish_alias(dish_id);
CREATE INDEX IF NOT EXISTS idx_dish_alias_alias ON dish_alias(alias);
CREATE INDEX IF NOT EXISTS idx_recipe_dish_id ON recipe(dish_id);
CREATE INDEX IF NOT EXISTS idx_recipe_ingredient_recipe ON recipe_ingredient(recipe_id);
CREATE INDEX IF NOT EXISTS idx_recipe_ingredient_usda ON recipe_ingredient(usda_fdc_id);
CREATE INDEX IF NOT EXISTS idx_usda_food_group ON usda_food(food_group);
CREATE INDEX IF NOT EXISTS idx_symspell_variant ON symspell_deletes(delete_variant);

-- ── FTS5 Full-Text Search ────────────────────────────────────────────

CREATE VIRTUAL TABLE IF NOT EXISTS dish_fts USING fts5(
    canonical_name,
    description,
    content='dish',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS dish_alias_fts USING fts5(
    alias,
    content='dish_alias',
    content_rowid='id'
);

-- ── FTS5 Triggers for dish_fts ───────────────────────────────────────

CREATE TRIGGER IF NOT EXISTS dish_fts_ai AFTER INSERT ON dish BEGIN
    INSERT INTO dish_fts(rowid, canonical_name, description)
    VALUES (new.id, new.canonical_name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS dish_fts_ad AFTER DELETE ON dish BEGIN
    INSERT INTO dish_fts(dish_fts, rowid, canonical_name, description)
    VALUES ('delete', old.id, old.canonical_name, old.description);
END;

CREATE TRIGGER IF NOT EXISTS dish_fts_au AFTER UPDATE ON dish BEGIN
    INSERT INTO dish_fts(dish_fts, rowid, canonical_name, description)
    VALUES ('delete', old.id, old.canonical_name, old.description);
    INSERT INTO dish_fts(rowid, canonical_name, description)
    VALUES (new.id, new.canonical_name, new.description);
END;

-- ── FTS5 Triggers for dish_alias_fts ─────────────────────────────────

CREATE TRIGGER IF NOT EXISTS dish_alias_fts_ai AFTER INSERT ON dish_alias BEGIN
    INSERT INTO dish_alias_fts(rowid, alias)
    VALUES (new.id, new.alias);
END;

CREATE TRIGGER IF NOT EXISTS dish_alias_fts_ad AFTER DELETE ON dish_alias BEGIN
    INSERT INTO dish_alias_fts(dish_alias_fts, rowid, alias)
    VALUES ('delete', old.id, old.alias);
END;

CREATE TRIGGER IF NOT EXISTS dish_alias_fts_au AFTER UPDATE ON dish_alias BEGIN
    INSERT INTO dish_alias_fts(dish_alias_fts, rowid, alias)
    VALUES ('delete', old.id, old.alias);
    INSERT INTO dish_alias_fts(rowid, alias)
    VALUES (new.id, new.alias);
END;
