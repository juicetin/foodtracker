-- Food Knowledge Graph Schema
-- SQLite database for dish -> ingredient -> nutrient relationships
-- with variant tracking and FTS5 full-text search

CREATE TABLE IF NOT EXISTS dishes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    canonical_id INTEGER REFERENCES dishes(id),  -- variant-of relationship
    cuisine TEXT,                                  -- Western, Chinese, Japanese, Korean, Vietnamese, Thai, Indian, Other
    source TEXT NOT NULL DEFAULT 'recipenlg',      -- recipenlg, usda, user_correction
    confidence REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingredients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    usda_fdc_id INTEGER,                          -- USDA FoodData Central ID for nutrient lookup
    category TEXT,                                 -- protein, grain, vegetable, oil, seasoning, dairy, etc.
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dish_ingredients (
    dish_id INTEGER REFERENCES dishes(id) ON DELETE CASCADE,
    ingredient_id INTEGER REFERENCES ingredients(id) ON DELETE CASCADE,
    weight_pct REAL,                              -- proportion of total dish weight (0.0-1.0)
    is_nutrition_significant BOOLEAN DEFAULT 1,   -- FALSE for garnishes/trace amounts
    typical_amount_g REAL,                        -- typical weight in grams for standard serving
    source TEXT NOT NULL DEFAULT 'recipenlg',
    confidence REAL DEFAULT 0.5,
    PRIMARY KEY (dish_id, ingredient_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_dishes_cuisine ON dishes(cuisine);
CREATE INDEX IF NOT EXISTS idx_dishes_canonical ON dishes(canonical_id);
CREATE INDEX IF NOT EXISTS idx_dishes_name ON dishes(name);
CREATE INDEX IF NOT EXISTS idx_ingredients_usda ON ingredients(usda_fdc_id);
CREATE INDEX IF NOT EXISTS idx_ingredients_name ON ingredients(name);
CREATE INDEX IF NOT EXISTS idx_dish_ingredients_dish ON dish_ingredients(dish_id);
CREATE INDEX IF NOT EXISTS idx_dish_ingredients_ingredient ON dish_ingredients(ingredient_id);

-- FTS5 virtual table for fuzzy dish name search
CREATE VIRTUAL TABLE IF NOT EXISTS dishes_fts USING fts5(name, cuisine, content=dishes, content_rowid=id);

-- Triggers to keep FTS5 in sync with dishes table
CREATE TRIGGER IF NOT EXISTS dishes_ai AFTER INSERT ON dishes BEGIN
    INSERT INTO dishes_fts(rowid, name, cuisine) VALUES (new.id, new.name, new.cuisine);
END;

CREATE TRIGGER IF NOT EXISTS dishes_ad AFTER DELETE ON dishes BEGIN
    INSERT INTO dishes_fts(dishes_fts, rowid, name, cuisine) VALUES('delete', old.id, old.name, old.cuisine);
END;

CREATE TRIGGER IF NOT EXISTS dishes_au AFTER UPDATE ON dishes BEGIN
    INSERT INTO dishes_fts(dishes_fts, rowid, name, cuisine) VALUES('delete', old.id, old.name, old.cuisine);
    INSERT INTO dishes_fts(rowid, name, cuisine) VALUES (new.id, new.name, new.cuisine);
END;
