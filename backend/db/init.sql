-- Food Tracker Database Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    region VARCHAR(10) DEFAULT 'AU', -- AU, US, CA, UK, FR, global
    units VARCHAR(10) DEFAULT 'metric', -- metric, imperial
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Nutrition goals table
CREATE TABLE IF NOT EXISTS nutrition_goals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    calories DECIMAL(10, 2) DEFAULT 2000,
    protein DECIMAL(10, 2) DEFAULT 150,
    carbs DECIMAL(10, 2) DEFAULT 200,
    fat DECIMAL(10, 2) DEFAULT 65,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id)
);

-- Food entries table (meals/logs)
CREATE TABLE IF NOT EXISTS food_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    meal_type VARCHAR(50) NOT NULL, -- breakfast, lunch, dinner, snack
    entry_date DATE NOT NULL DEFAULT CURRENT_DATE,
    total_calories DECIMAL(10, 2) DEFAULT 0,
    total_protein DECIMAL(10, 2) DEFAULT 0,
    total_carbs DECIMAL(10, 2) DEFAULT 0,
    total_fat DECIMAL(10, 2) DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Photos table (linked to entries)
CREATE TABLE IF NOT EXISTS photos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entry_id UUID REFERENCES food_entries(id) ON DELETE CASCADE,
    uri TEXT NOT NULL,
    gcs_url TEXT, -- Google Cloud Storage URL
    width INTEGER,
    height INTEGER,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ingredients table (individual food items in an entry)
CREATE TABLE IF NOT EXISTS ingredients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entry_id UUID REFERENCES food_entries(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    quantity DECIMAL(10, 2) NOT NULL,
    unit VARCHAR(50) NOT NULL, -- g, kg, oz, lb, ml, cup, tbsp, etc.
    calories DECIMAL(10, 2) NOT NULL,
    protein DECIMAL(10, 2) DEFAULT 0,
    carbs DECIMAL(10, 2) DEFAULT 0,
    fat DECIMAL(10, 2) DEFAULT 0,
    fiber DECIMAL(10, 2) DEFAULT 0,
    sugar DECIMAL(10, 2) DEFAULT 0,
    -- AI detection metadata
    ai_confidence DECIMAL(3, 2), -- 0.00 to 1.00
    bounding_box_x DECIMAL(5, 2),
    bounding_box_y DECIMAL(5, 2),
    bounding_box_width DECIMAL(5, 2),
    bounding_box_height DECIMAL(5, 2),
    -- Database source
    database_source VARCHAR(50), -- AFCD, USDA, CoFID, CIQUAL, OpenFoodFacts
    database_id VARCHAR(255), -- ID in the source database
    -- User modifications
    user_modified BOOLEAN DEFAULT FALSE,
    original_quantity DECIMAL(10, 2), -- Original AI estimate
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Modification history table (for retrospective editing tracking)
CREATE TABLE IF NOT EXISTS modification_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ingredient_id UUID REFERENCES ingredients(id) ON DELETE CASCADE,
    field_name VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_by UUID REFERENCES users(id)
);

-- Custom recipes table (saved from previous AI scans)
CREATE TABLE IF NOT EXISTS custom_recipes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    -- Reference to original entry this was created from
    source_entry_id UUID REFERENCES food_entries(id) ON DELETE SET NULL,
    -- Aggregated nutrition info
    total_calories DECIMAL(10, 2) DEFAULT 0,
    total_protein DECIMAL(10, 2) DEFAULT 0,
    total_carbs DECIMAL(10, 2) DEFAULT 0,
    total_fat DECIMAL(10, 2) DEFAULT 0,
    -- Usage tracking
    times_used INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recipe ingredients (normalized ingredients for a recipe)
CREATE TABLE IF NOT EXISTS recipe_ingredients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recipe_id UUID REFERENCES custom_recipes(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    quantity DECIMAL(10, 2) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    calories DECIMAL(10, 2) NOT NULL,
    protein DECIMAL(10, 2) DEFAULT 0,
    carbs DECIMAL(10, 2) DEFAULT 0,
    fat DECIMAL(10, 2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recipe photos (reference photos for the recipe)
CREATE TABLE IF NOT EXISTS recipe_photos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recipe_id UUID REFERENCES custom_recipes(id) ON DELETE CASCADE,
    gcs_url TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_food_entries_user_date ON food_entries(user_id, entry_date DESC);
CREATE INDEX idx_food_entries_user_id ON food_entries(user_id);
CREATE INDEX idx_ingredients_entry_id ON ingredients(entry_id);
CREATE INDEX idx_photos_entry_id ON photos(entry_id);
CREATE INDEX idx_custom_recipes_user_id ON custom_recipes(user_id);
CREATE INDEX idx_recipe_ingredients_recipe_id ON recipe_ingredients(recipe_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_nutrition_goals_updated_at BEFORE UPDATE ON nutrition_goals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_food_entries_updated_at BEFORE UPDATE ON food_entries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ingredients_updated_at BEFORE UPDATE ON ingredients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_custom_recipes_updated_at BEFORE UPDATE ON custom_recipes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert a test user for development
INSERT INTO users (email, name, region, units)
VALUES ('test@example.com', 'Test User', 'AU', 'metric')
ON CONFLICT (email) DO NOTHING;

-- Insert nutrition goals for test user
INSERT INTO nutrition_goals (user_id, calories, protein, carbs, fat)
SELECT id, 2000, 150, 200, 65 FROM users WHERE email = 'test@example.com'
ON CONFLICT (user_id) DO NOTHING;
