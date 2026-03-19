# On-Device Food Knowledge Graphs & Recipe-Based Nutrition Estimation

Research date: 2026-03-14

---

## 1. Food Knowledge Graphs

### 1.1 Major Food Knowledge Graph Projects

#### FoodKG
- **Size**: ~97 million RDF triples across three files (usda-links.trig: ~4.1M triples, foodon-links.trig: ~30K triples, foodkg-core.trig: ~63M triples)
- **Sources**: Recipe1M+ (1M+ recipes, 13M food images), USDA nutrient data, FoodOn ontology
- **Structure**: Recipes -> ingredients -> nutrition. Each recipe has a unique ID, name, tags, and ingredient set. Each ingredient points to its name, unit, and quantity. Ingredients link to USDA nutrient profiles (per 100g) via Jaccard similarity matching
- **Coverage**: 90.87% of ingredients linked to USDA, 92.35% to FoodOn, 81.80% to FooDB in expanded version
- **Problem for mobile**: 97M triples is far too large. Full RDF serialization is multiple GB. Need a curated subset
- Source: [FoodKG](https://foodkg.github.io/), [FoodKG Construction](https://foodkg.github.io/foodkg.html)

#### FoodOn
- **Size**: 9,445+ food product type classes in a hierarchical taxonomy
- **Structure**: Farm-to-fork ontology with facets: product type, food source (plants/animals), processing methods, packaging
- **Multilingual**: Terms include multilingual labels, synonyms, and globally unique identifiers
- **Format**: OWL/RDF ontology (OBO Foundry consortium)
- **Mobile relevance**: The class hierarchy itself (without full ontology reasoning) could be extracted as a compact taxonomy lookup table
- Source: [FoodOn](https://foodon.org/), [FoodOn Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550238/)

#### Open Food Facts
- **Size**: ~4 million product entries (as of late 2025), 206 columns per product
- **Full database**: ~9 GB uncompressed CSV, ~0.9 GB gzip compressed
- **Mini SQLite version**: 160 MB with only barcode, product name, countries (2.45M records, 98% size reduction from full DB, but NO nutrition data)
- **Key limitation**: Full nutrition version is too large for bundling in a mobile app (~1GB+). Would need a curated subset by region/cuisine
- **License**: Open Database License (ODbL v1.0)
- Source: [Open Food Facts Data](https://world.openfoodfacts.org/data), [OFF SQLite Mini](https://github.com/Accessibilly/openfoodfacts-sqlite-mini)

#### USDA FoodData Central
- **SR Legacy**: 7,793 food items, up to 150 nutrients per food, nutrients per 100g. Last updated April 2018
- **Foundation Foods**: Updated December 2025, 467K zipped JSON / 6.5M uncompressed
- **Branded Foods**: 195M zipped / 3.1G uncompressed (too large for mobile)
- **SR Legacy SQLite**: Community conversions exist. SR28 version has 8,789 foods. Estimated SQLite size: **~10-15 MB** for SR Legacy (based on the JSON-indexed version being ~7MB for food descriptions alone)
- **Full FoodData Central SQLite**: ~430 MB (includes branded foods -- too large)
- **License**: Public domain (CC0 1.0)
- Source: [USDA FDC Downloads](https://fdc.nal.usda.gov/download-datasets/), [USDA SQLite (alyssaq)](https://github.com/alyssaq/usda-sqlite), [USDA SQLite (MenuLogistics)](https://github.com/MenuLogistics/USDASQLite)

### 1.2 Building a Compact On-Device Food KG

#### Target Size Budget: 50-100 MB

Recommended composition for an on-device food knowledge graph:

| Component | Estimated Size | Contents |
|-----------|---------------|----------|
| USDA SR Legacy (SQLite) | ~10-15 MB | 7,793 generic foods with full nutrition per 100g, portion weights |
| Recipe database (curated) | ~15-25 MB | 20K-50K canonical recipes with ingredient lists + quantities |
| Food taxonomy/ontology | ~2-5 MB | Hierarchical dish categories, cuisine mapping, aliases |
| Food name embeddings (Model2Vec) | ~8-15 MB | Static embeddings for ~30K food/ingredient names for fuzzy search |
| Vector search index (libSQL/USearch) | ~2-5 MB | HNSW index for embedding similarity search |
| FTS5 index | ~3-5 MB | Full-text search index for food names and ingredients |
| **Total** | **~40-70 MB** | Fits within 50-100 MB budget |

#### Recommended Schema (SQLite relational, no graph extension needed)

```sql
-- Core food taxonomy
CREATE TABLE cuisine (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,          -- "Thai", "Italian", "Mexican"
  region TEXT                  -- "Southeast Asia", "Southern Europe"
);

CREATE TABLE dish_category (
  id INTEGER PRIMARY KEY,
  cuisine_id INTEGER REFERENCES cuisine(id),
  name TEXT NOT NULL,          -- "Noodle dishes", "Curries", "Salads"
  parent_id INTEGER REFERENCES dish_category(id)  -- hierarchical
);

CREATE TABLE dish (
  id INTEGER PRIMARY KEY,
  category_id INTEGER REFERENCES dish_category(id),
  canonical_name TEXT NOT NULL,  -- "Pad Thai"
  description TEXT,
  avg_calories_per_serving REAL,
  avg_protein_per_serving REAL,
  avg_carbs_per_serving REAL,
  avg_fat_per_serving REAL,
  default_serving_grams REAL
);

-- Aliases and multilingual names
CREATE TABLE dish_alias (
  id INTEGER PRIMARY KEY,
  dish_id INTEGER REFERENCES dish(id),
  alias TEXT NOT NULL,          -- "pad thai", "phad thai", "fried rice noodles"
  language TEXT DEFAULT 'en',
  alias_type TEXT               -- "spelling_variant", "translation", "colloquial"
);

-- Canonical recipes (one or more per dish)
CREATE TABLE recipe (
  id INTEGER PRIMARY KEY,
  dish_id INTEGER REFERENCES dish(id),
  name TEXT,
  source TEXT,                  -- "RecipeDB", "manual"
  total_calories REAL,
  total_weight_grams REAL,
  servings INTEGER DEFAULT 1,
  is_canonical BOOLEAN DEFAULT 0  -- flag the "default" recipe for each dish
);

-- Recipe ingredients with quantities
CREATE TABLE recipe_ingredient (
  id INTEGER PRIMARY KEY,
  recipe_id INTEGER REFERENCES recipe(id),
  usda_food_id INTEGER,         -- FK to USDA nutrition table
  ingredient_name TEXT NOT NULL, -- "rice noodles"
  quantity REAL,                 -- 200
  unit TEXT,                     -- "g", "cup", "tbsp"
  quantity_grams REAL,           -- normalized to grams for nutrition calc
  sort_order INTEGER
);

-- USDA nutrition data (SR Legacy)
CREATE TABLE usda_food (
  id INTEGER PRIMARY KEY,       -- fdc_id
  description TEXT NOT NULL,
  food_group TEXT,
  calories_per_100g REAL,
  protein_per_100g REAL,
  fat_per_100g REAL,
  carbs_per_100g REAL,
  fiber_per_100g REAL,
  sugar_per_100g REAL,
  sodium_per_100g REAL
  -- additional micronutrients as needed
);

CREATE TABLE usda_portion (
  id INTEGER PRIMARY KEY,
  food_id INTEGER REFERENCES usda_food(id),
  portion_description TEXT,     -- "1 cup", "1 medium", "1 slice"
  portion_grams REAL
);

-- Full-text search
CREATE VIRTUAL TABLE dish_fts USING fts5(
  canonical_name, description,
  content='dish', content_rowid='id'
);

CREATE VIRTUAL TABLE dish_alias_fts USING fts5(
  alias,
  content='dish_alias', content_rowid='id'
);
```

This relational schema is effectively a knowledge graph stored in normalized tables. The key relationships are:
- **cuisine -> dish_category -> dish**: Hierarchical food ontology
- **dish -> recipe -> recipe_ingredient -> usda_food**: Recipe decomposition path
- **dish -> dish_alias**: Multilingual and variant name support

No graph database extension is needed -- SQLite's JOIN capabilities handle these traversals efficiently.

### 1.3 Graph Database Options for Mobile

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Plain SQLite with JOINs** | Universal mobile support, zero dependencies, FTS5 built-in, tiny footprint | No native graph traversal syntax | **Best choice** -- food KG queries are simple 2-3 hop joins |
| **libSQL (Turso fork)** | Native vector search (DiskANN), React Native SDK (OP-SQLite), knowledge graph + vector in one DB | React Native SDK still WIP, iOS linking conflicts with expo-sqlite | **Strong option** if vector search needed |
| **simple-graph (SQLite)** | Nodes as JSON + edges with CTE traversals, Dart/Swift bindings | No JS/TS binding, performance degrades at millions of nodes | Not ideal for React Native |
| **GraphQLite** | Cypher query language, graph algorithms, SQLite extension | Alpha stage, Rust/Python only, not production-ready | Too early |
| **CozoDB** | Datalog queries, graph+vector, SQLite backend, 250K QPS reads | No React Native binding, complex query language | Overkill for food KG |
| **RDFox** | Full RDF reasoning on device | Commercial license, heavy for mobile | Too heavy |

**Recommendation**: Use plain SQLite for the food knowledge graph structure, and either libSQL or a separate USearch index for vector similarity search if needed.

---

## 2. Recipe Databases for Nutrition Decomposition

### 2.1 Open Recipe Datasets

| Dataset | Recipes | Ingredients | Nutrition? | License | Download |
|---------|---------|-------------|------------|---------|----------|
| **Recipe1M+** | 1M+ | ~16K unique | Partial (via USDA matching) | Research only | [MIT](https://pic2recipe.csail.mit.edu/) |
| **RecipeNLG** | 2.23M (1.6M high-quality) | NER-tagged | No | CC BY-NC-SA | [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/) |
| **RecipeDB** | 118,171 | 23,548 (mapped to 1,636 generic names) | Yes (via USDA SR) | Research | [cosylab.iiitd.edu.in](https://cosylab.iiitd.edu.in/recipedb/) |
| **Open Recipes** | ~175K | Unstructured | No | Various | Various scraped sources |
| **FoodKG recipes** | ~1M (from Recipe1M+) | Structured with units/qty | Via USDA linkage | Research | [foodkg.github.io](https://foodkg.github.io/) |

**Best starting point**: RecipeDB -- it has 118K recipes across 74 countries, with ingredients already mapped to USDA nutrition data via Jaccard similarity. The 1,636 generic ingredient names provide a manageable vocabulary to map against USDA SR Legacy's 7,793 foods.

### 2.2 The Dish-to-Nutrition Pipeline

```
User identifies dish: "pad thai"
         |
         v
[1. Dish Lookup] -- FTS5/fuzzy search in dish + dish_alias tables
         |
         v
[2. Recipe Selection] -- Find canonical recipe(s) for this dish
         |
         v
[3. Ingredient List] -- rice noodles 200g, chicken 150g, bean sprouts 100g,
         |               peanuts 30g, egg 1, fish sauce 2 tbsp, tamarind 1 tbsp...
         v
[4. USDA Matching] -- Map each ingredient to usda_food entry
         |
         v
[5. Nutrition Calculation]
         |   For each ingredient:
         |     nutrition = (quantity_grams / 100) * nutrient_per_100g
         |   Sum across all ingredients
         |   Divide by servings
         v
[6. Result] -- ~450 kcal, 20g protein, 55g carbs, 16g fat per serving
```

### 2.3 Handling Portion Sizes and Variations

**Portion size strategies (no depth sensor)**:
1. **Default serving**: Store a default serving size per dish in grams (e.g., pad thai = 350g). Source from recipe averages
2. **Multiplier UI**: Let user select 0.5x, 1x, 1.5x, 2x of default serving
3. **Named portions**: "small", "medium", "large" mapped to gram weights (e.g., 250g/350g/500g)
4. **USDA portions table**: Use `usda_portion` for common measures ("1 cup cooked rice" = 158g)
5. **Thumb reference** (future): Research shows thumb-beside-food gives scale for monocular volume estimation

**Handling dish variations**:
- Store multiple recipes per dish (e.g., "pad thai - chicken", "pad thai - shrimp", "pad thai - tofu")
- Use a "canonical" recipe as default, let user switch variants
- For unknown variants, use the canonical recipe's nutrition as a reasonable estimate
- Restaurant portions are typically 80-100% more calories than homemade -- consider a "restaurant mode" multiplier (1.5-2x)

---

## 3. Combining Text Hints with Image Classification

### 3.1 Research Summary

The state-of-the-art approach from "Beyond Images: Adaptive Fusion of Visual and Textual Data for Food Classification" achieves:
- Image-only: 73.60% accuracy
- Text-only: 88.84% accuracy
- **Dynamic fusion: 97.84% accuracy** (24+ points over image-only)

Key insight: **text metadata alone outperforms images** for food classification. The fusion approach uses uncertainty-weighted combination where text confidence modulates visual contribution.

### 3.2 Practical On-Device Architecture

```
User takes photo of brown curry dish
         |
         v
[Image Classifier] -- GGCD/YOLO model
  Returns top-5 predictions with confidence:
    1. "massaman curry" (0.25)
    2. "panang curry" (0.22)
    3. "rendang" (0.18)
    4. "beef stew" (0.15)
    5. "adobo" (0.10)
         |
         v
[User types: "massaman"]
         |
         v
[Text Matching] -- Fuzzy search against dish names + aliases
  Returns matches with scores:
    1. "massaman curry" (0.95)
    2. "massaman paste" (0.60)
         |
         v
[Bayesian Re-ranking]
  P(dish | image, text) proportional to P(image | dish) * P(text | dish)

  For each candidate dish:
    combined_score = image_confidence * text_similarity_score

    "massaman curry": 0.25 * 0.95 = 0.2375  <-- winner
    "panang curry":   0.22 * 0.10 = 0.0220
    "rendang":        0.18 * 0.05 = 0.0090
         |
         v
[Result]: "Massaman Curry" with high confidence
```

### 3.3 Implementation Approaches

**Approach 1: Simple score multiplication (recommended for v1)**
```typescript
function rerank(
  imageResults: Array<{label: string, confidence: number}>,
  userText: string,
  dishNames: string[]
): Array<{label: string, score: number}> {
  const textScores = fuzzyMatch(userText, dishNames); // SymSpell or embedding similarity
  return imageResults.map(r => ({
    label: r.label,
    score: r.confidence * (textScores[r.label] ?? 0.01) // small prior for unmatched
  })).sort((a, b) => b.score - a.score);
}
```

**Approach 2: Embedding-based re-ranking (MobileCLIP/SigLIP)**
- Use MobileCLIP-S0 (~50M params, 4.8x faster than ViT-B/16, ~20-30MB quantized)
- Compute text embedding for user input
- Compute image embedding for the photo
- Find nearest dish names in pre-computed embedding space
- Combine cosine similarities from both modalities
- **Note**: MobileCLIP-S0 runs at 3-15ms latency on iPhone 12 Pro Max

**Approach 3: Uncertainty-weighted fusion (research-grade)**
- softmax(softmax(text_features) + (sigmoid(text_uncertainty) - 0.5) * softmax(image_features))
- When text is confident: rely primarily on text
- When text is uncertain: weight image more heavily

### 3.4 Lightweight Vision-Language Models for Mobile

| Model | Size | Latency (iPhone 12) | Zero-shot Accuracy | Notes |
|-------|------|---------------------|-------------------|-------|
| MobileCLIP-S0 | ~50M params | 3ms | ~ViT-B/16 level | Apple, ONNX/CoreML |
| MobileCLIP-S2 | ~80M params | ~5ms | > SigLIP ViT-B/16 | Apple, 2.3x faster than SigLIP |
| SigLIP2 ViT-B | 86M params | ~10ms | Strong multilingual | Google, INT8 quantizable |
| MobileCLIP2-S4 | ~150M params | ~15ms | = SigLIP-SO400M | Apple, 2x fewer params |

Source: [MobileCLIP (Apple)](https://github.com/apple/ml-mobileclip), [SigLIP 2 (HuggingFace)](https://huggingface.co/blog/siglip2)

---

## 4. Hierarchical Food Ontologies

### 4.1 Recommended Hierarchy Structure

```
Level 0: Cuisine Region
  "Southeast Asian", "East Asian", "South Asian", "Mediterranean",
  "Latin American", "North American", "African", "Middle Eastern"...

Level 1: National Cuisine
  "Thai", "Vietnamese", "Japanese", "Chinese (Cantonese)",
  "Chinese (Sichuan)", "Indian (North)", "Indian (South)"...

Level 2: Dish Category
  "Noodle dishes", "Rice dishes", "Curries", "Soups",
  "Salads", "Grilled/BBQ", "Fried dishes", "Desserts"...

Level 3: Specific Dish
  "Pad Thai", "Pad See Ew", "Pad Kra Pao",
  "Khao Pad", "Tom Yum", "Green Curry"...

Level 4: Dish Variant
  "Pad Thai - Chicken", "Pad Thai - Shrimp", "Pad Thai - Tofu",
  "Pad Thai - Vegetarian"...
```

### 4.2 Handling Cultural/Regional Names and Aliases

Key challenges:
- **Same dish, different names**: "Pad Kra Pao" = "Thai basil chicken" = "holy basil stir fry"
- **Same name, different dishes**: "biscuit" (US soft bread vs UK hard cookie)
- **Transliteration variants**: "Pad Thai" / "Phad Thai" / "Phat Thai"
- **Regional variations**: "biryani" varies dramatically across South Asia

Solution: The `dish_alias` table with `language` and `alias_type` fields. For disambiguation of same-name-different-dish cases, include the cuisine context:
```sql
-- "biscuit" in American context
INSERT INTO dish_alias (dish_id, alias, language, alias_type)
VALUES (123, 'biscuit', 'en-US', 'regional');

-- "biscuit" in British context
INSERT INTO dish_alias (dish_id, alias, language, alias_type)
VALUES (456, 'biscuit', 'en-GB', 'regional');
```

### 4.3 Scale Estimates

Based on RecipeDB's coverage of 74 countries and 118K recipes:
- ~200 cuisine regions/national cuisines
- ~50-100 dish categories
- ~5,000-10,000 distinct dishes (covering 80% of commonly eaten foods worldwide)
- ~2-5 aliases per dish = ~10K-50K alias entries
- ~2-3 recipe variants per dish = ~10K-30K recipes
- **Total taxonomy size**: ~2-5 MB in SQLite

Source: [AMALTHEIA Ontology](https://www.mdpi.com/2306-5729/6/4/41), [RecipeDB (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687679)

---

## 5. On-Device Text Matching and Search

### 5.1 Fuzzy Matching for Food Names

#### SymSpell (Recommended for v1)
- **Algorithm**: Symmetric Delete spelling correction -- only uses deletes, not inserts/replaces/transposes
- **Speed**: 1,870x faster than BK-tree at dictionary size 500K, edit distance 3
- **Memory**: Pre-computed delete dictionary. For 30K food names at max edit distance 2: ~10-20 MB RAM
- **JS/TS packages**: `symspell-ex` (npm), `SymSpell.js` (TypeScript)
- **How it works**: Pre-compute all delete variants of dictionary words. At query time, compute deletes of query word. Intersection = fuzzy matches
- Source: [SymSpell GitHub](https://github.com/wolfgarbe/SymSpell)

#### Levenshtein Automaton
- Build a finite automaton that accepts all strings within edit distance k of the query
- Intersect with an FST (finite state transducer) built from the food name dictionary
- Optimal for large dictionaries but more complex to implement

### 5.2 Embedding-Based Similarity Search

#### Model2Vec (Recommended for lightweight embeddings)
- **How**: Distill any sentence transformer into static per-token embeddings. Sentence embedding = mean of token vectors
- **Size**: 8-30 MB on disk (smallest model on MTEB is ~8 MB)
- **Speed**: Up to 500x faster than the original transformer model on CPU
- **Quality**: Small accuracy drop vs full transformer, but excellent for food name matching where exact semantic understanding isn't critical
- **Distillation**: Takes ~30 seconds on CPU, needs only a vocabulary and a model
- Source: [Model2Vec (GitHub)](https://github.com/MinishLab/model2vec), [Model2Vec Blog](https://huggingface.co/blog/Pringled/model2vec)

#### all-MiniLM-L6-v2 (Full transformer option)
- **Size**: ~22 MB (ONNX format)
- **Dimensions**: 384-dimensional embeddings
- **Android**: Available via `sentence-embeddings` library (ONNX Runtime)
- **Speed**: Slower than Model2Vec but better contextual understanding
- Source: [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [Android Implementation](https://github.com/shubham0204/Sentence-Embeddings-Android)

#### Vector Search Indexes

**USearch** (Recommended):
- Memory-mapped file access (don't load entire index into RAM)
- 10x faster than FAISS in many benchmarks
- Supports iOS, Android, WebAssembly
- Bindings: C++, C, Python, JavaScript, Rust, Java, Objective-C, Swift
- Tested: 100M+ entries on iPhone
- Source: [USearch GitHub](https://github.com/unum-cloud/USearch)

**libSQL native vector search** (Alternative -- combines with SQLite):
- DiskANN algorithm, native to SQLite fork
- Float8 compression: 8x index size reduction
- React Native support via OP-SQLite
- Index overhead: ~2.4 MB for ~1K nodes with float8 compression + max_neighbors=20
- For 30K food name embeddings at 384 dimensions: estimated ~10-15 MB index with float8
- Source: [Turso Vector Search](https://turso.tech/blog/building-vector-search-and-personal-knowledge-graphs-on-mobile-with-libsql-and-react-native)

### 5.3 Trie/Prefix Trees for Autocomplete

- O(p + k) lookup time where p = prefix length, k = number of matches
- For 30K food names: trie would use ~1-3 MB RAM
- Can combine with SymSpell: trie for prefix completion, SymSpell for typo correction
- JavaScript libraries: `mnemonist` (includes SymSpell + trie), `trie-search`

### 5.4 SQLite FTS5 with BM25 for Recipe Search

- Built into SQLite, zero additional dependencies
- BM25 ranking built-in: `SELECT * FROM recipe_fts WHERE recipe_fts MATCH 'chicken curry' ORDER BY bm25(recipe_fts, 1.0, 0.5)`
- Column weighting supported (weight dish name higher than description)
- Excellent for "search recipes containing these ingredients" queries
- Performance: instant for databases under 1M rows on mobile
- Source: [SQLite FTS5](https://www.sqlite.org/fts5.html)

### 5.5 Recommended Search Architecture

```
User types: "cesear salad"
         |
    [Layer 1: Prefix Trie] -- No prefix match (typo)
         |
    [Layer 2: SymSpell] -- Edit distance 1-2 correction
         |  Returns: "caesar salad" (distance=1)
         |
    [Layer 3: FTS5] -- Full-text search on corrected query
         |  Returns: dish + recipe results ranked by BM25
         |
    [Optional Layer 4: Embedding Search]
         |  If FTS5 returns few/no results, fall back to
         |  semantic similarity via Model2Vec + USearch
         |  Handles: "green leafy salad with croutons" -> "Caesar salad"
         v
    [Results]: Ranked list of dishes with nutrition info
```

---

## 6. Nutrition Estimation from Ingredients

### 6.1 USDA SR Database for On-Device Use

**SR Legacy structure** (6 core tables):

| Table | Contents | Rows |
|-------|----------|------|
| food | Food descriptions with groups | 7,793 |
| nutrient | Nutrient definitions (names, units) | ~150 |
| nutrition | Nutrient amounts per food per 100g | ~680K |
| weight | Portion size descriptions + gram weights | ~15K |
| food_group | Food group categories | 25 |

**Key fields per food**: calories, protein, total fat, carbohydrates, fiber, sugars, sodium, plus ~140 micronutrients (vitamins, minerals, amino acids, fatty acids).

**Estimated on-device SQLite size**: 10-15 MB for full SR Legacy with indexes. Can be reduced to ~5 MB by keeping only the most common 20-30 nutrients.

### 6.2 Mapping Free-Text Ingredients to USDA Entries

**Approach 1: Pre-computed mapping table (Recommended)**
- RecipeDB already maps 23,548 ingredients to 1,636 generic names via Jaccard similarity
- Build a static `ingredient_to_usda` mapping table shipped with the app
- ~1,636 generic ingredients -> ~7,793 USDA foods = manageable mapping
- Store as: `{ingredient_name: "rice noodles", usda_food_id: 12345, confidence: 0.95}`

**Approach 2: On-device NER + fuzzy matching**
- Parse "2 cups of chopped chicken breast" into: quantity=2, unit=cup, state=chopped, ingredient=chicken breast
- Fuzzy match "chicken breast" against USDA food descriptions
- Lightweight NER models: Fine-tuned DistilBERT (~66 MB) or rule-based regex patterns
- The TASTEset dataset provides 700 annotated recipe ingredient lists for training

**Approach 3: Embedding similarity**
- Pre-compute embeddings for all USDA food descriptions
- At query time, embed the ingredient text and find nearest USDA entry
- Model2Vec + USearch makes this fast on device

### 6.3 Nutrition Calculation Algorithm

```typescript
interface NutritionResult {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  // ... other nutrients
}

function calculateDishNutrition(
  recipeIngredients: RecipeIngredient[],
  usdaDatabase: USDADatabase,
  servings: number = 1
): NutritionResult {
  const totals: NutritionResult = { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0 };

  for (const ingredient of recipeIngredients) {
    const usdaFood = usdaDatabase.getFood(ingredient.usdaFoodId);
    if (!usdaFood) continue;

    // Convert quantity to grams
    let grams = ingredient.quantityGrams;
    if (!grams && ingredient.unit) {
      // Look up portion: "1 cup" of this food = X grams
      const portion = usdaDatabase.getPortion(ingredient.usdaFoodId, ingredient.unit);
      grams = (portion?.portionGrams ?? 100) * ingredient.quantity;
    }
    if (!grams) grams = 100; // fallback: assume 100g

    // Scale nutrition from per-100g to actual quantity
    const scale = grams / 100;
    totals.calories += usdaFood.caloriesPer100g * scale;
    totals.protein += usdaFood.proteinPer100g * scale;
    totals.carbs += usdaFood.carbsPer100g * scale;
    totals.fat += usdaFood.fatPer100g * scale;
    totals.fiber += usdaFood.fiberPer100g * scale;
  }

  // Divide by servings
  return {
    calories: Math.round(totals.calories / servings),
    protein: Math.round(totals.protein / servings * 10) / 10,
    carbs: Math.round(totals.carbs / servings * 10) / 10,
    fat: Math.round(totals.fat / servings * 10) / 10,
    fiber: Math.round(totals.fiber / servings * 10) / 10,
  };
}
```

### 6.4 Portion Size Estimation Approaches

| Method | Accuracy | Complexity | On-Device? |
|--------|----------|------------|------------|
| Default serving per dish (from recipe DB) | Low-medium | Trivial | Yes |
| User selects S/M/L multiplier | Medium | Trivial | Yes |
| USDA portion table ("1 cup", "1 slice") | Medium | Simple | Yes |
| Monocular image + thumb reference | Medium-high | Complex | Yes (future) |
| Depth sensor volume estimation | High (5-7% error) | Complex | Requires LiDAR |
| 3D reconstruction from multiple views | High | Very complex | Possible but slow |

**Recommended for v1**: Default serving size from recipe data + S/M/L multiplier + optional USDA portion lookup. This covers 80% of use cases with minimal complexity.

### 6.5 How Existing Apps Handle This

**Cronometer**:
- Curated database from NCCDB (17K entries, 70 nutrients) + USDA (8K entries, 70 nutrients)
- Nutritionix for barcode scanning (400K+ products)
- Every user-submitted food reviewed by curation team before inclusion
- Highest accuracy reputation among nutrition trackers

**MyFitnessPal**:
- 18M+ global foods (largest database)
- Mostly user-generated content (quality varies)
- Green checkmark system for verified entries
- Relies heavily on barcode scanning + crowd-sourced data

**Cal AI** (2024):
- Phone depth sensor for food volume estimation
- Claims 90% accuracy
- Cloud-based AI processing (not on-device)

**OpenNutriTracker** (open source):
- Flutter/Dart app
- Uses Open Food Facts + USDA FoodData Central
- All data encrypted and stored locally
- Privacy-focused, minimal data collection
- Source: [OpenNutriTracker GitHub](https://github.com/simonoppowa/OpenNutriTracker)

---

## 7. Implementation Roadmap Recommendation

### Phase 1: Basic dish lookup + nutrition (MVP)
- Ship SQLite database with:
  - USDA SR Legacy nutrition data (~10 MB)
  - 5K-10K curated dishes with canonical recipes from RecipeDB
  - Dish aliases and FTS5 index
  - Pre-computed ingredient-to-USDA mappings
- User flow: search dish name -> see nutrition per serving -> adjust portion
- Total on-device data: ~20-30 MB

### Phase 2: Text + image fusion
- Add SymSpell for fuzzy food name matching
- Integrate text hint re-ranking with existing GGCD image classifier
- Simple Bayesian score multiplication (image confidence * text similarity)
- Total additional data: ~5-10 MB (SymSpell dictionary)

### Phase 3: Semantic search + expanded coverage
- Add Model2Vec embeddings for food names (~8-15 MB)
- Add USearch or libSQL vector index (~5-10 MB)
- Expand to 20K-50K dishes
- Add multilingual food name support
- Total on-device data: ~50-70 MB

### Phase 4: Advanced estimation
- MobileCLIP for joint text-image understanding
- Portion size estimation from monocular images
- Personalized food recognition based on user eating patterns
- Custom recipe creation with ingredient-level nutrition

---

## Key Data Sources Summary

| Source | What It Provides | Size (on-device) | License |
|--------|-----------------|-------------------|---------|
| USDA SR Legacy | 7,793 foods, 150 nutrients, portions | ~10-15 MB SQLite | Public domain |
| RecipeDB | 118K recipes, 74 countries, USDA-linked ingredients | ~15-25 MB curated subset | Research |
| RecipeNLG | 2.2M recipes, NER-tagged ingredients | Too large to ship; use for training/curation | CC BY-NC-SA |
| FoodOn | 9,445 food product classes, hierarchical taxonomy | ~2-5 MB extracted taxonomy | Open (OBO) |
| Open Food Facts | 4M products, barcode data | 160 MB mini (no nutrition) | ODbL |
| Model2Vec embeddings | Static sentence embeddings | 8-30 MB | MIT |

---

## Sources

### Food Knowledge Graphs
- [FoodKG Project](https://foodkg.github.io/)
- [FoodKG Construction](https://foodkg.github.io/foodkg.html)
- [FoodKG Paper (ISWC 2019)](https://link.springer.com/chapter/10.1007/978-3-030-30796-7_10)
- [FoodOn Ontology](https://foodon.org/)
- [FoodOn Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550238/)
- [Food Data in the Semantic Web (2025 Survey)](https://arxiv.org/html/2509.00986v1)
- [Applications of Knowledge Graphs for Food Science (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9122965/)

### Databases and Downloads
- [USDA FoodData Central Downloads](https://fdc.nal.usda.gov/download-datasets/)
- [USDA SQLite (alyssaq)](https://github.com/alyssaq/usda-sqlite)
- [USDA SQLite (MenuLogistics)](https://github.com/MenuLogistics/USDASQLite)
- [USDA FoodData SQLite3 Fields](https://github.com/hogand/USDA-FoodData-SQLite3/blob/master/Fields.md)
- [Open Food Facts Data](https://world.openfoodfacts.org/data)
- [Open Food Facts SQLite Mini](https://github.com/Accessibilly/openfoodfacts-sqlite-mini)

### Recipe Datasets
- [Recipe1M+ (MIT)](https://pic2recipe.csail.mit.edu/)
- [RecipeNLG (HuggingFace)](https://huggingface.co/datasets/mbien/recipe_nlg)
- [RecipeNLG Project](https://recipenlg.cs.put.poznan.pl/)
- [RecipeDB Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687679)

### Multimodal Food Classification
- [Beyond Images: Adaptive Fusion for Food Classification](https://arxiv.org/html/2308.02562)
- [Multimodal Food Classification with LLMs (MDPI)](https://www.mdpi.com/2079-9292/13/22/4552)
- [FMiFood: Multi-modal Contrastive Learning](https://arxiv.org/html/2408.03922v1)
- [Context-Based Food Image Analysis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5448793/)
- [MobileCLIP (Apple)](https://github.com/apple/ml-mobileclip)
- [SigLIP 2 (HuggingFace)](https://huggingface.co/blog/siglip2)

### Food Ontologies
- [AMALTHEIA Dish Ontology (MDPI)](https://www.mdpi.com/2306-5729/6/4/41)
- [Open Food Facts Ingredients Ontology](https://wiki.openfoodfacts.org/Project:Ingredients_ontology)

### On-Device Search and Embeddings
- [SymSpell Algorithm](https://github.com/wolfgarbe/SymSpell)
- [Model2Vec (GitHub)](https://github.com/MinishLab/model2vec)
- [Model2Vec Blog (HuggingFace)](https://huggingface.co/blog/Pringled/model2vec)
- [USearch Vector Search (GitHub)](https://github.com/unum-cloud/USearch)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Sentence Embeddings Android](https://github.com/shubham0204/Sentence-Embeddings-Android)
- [all-MiniLM-L6-v2 (HuggingFace)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Graph Databases for Mobile
- [simple-graph (GitHub)](https://github.com/dpapathanasiou/simple-graph)
- [GraphQLite (GitHub)](https://github.com/colliery-io/graphqlite)
- [CozoDB](https://github.com/cozodb/cozo)
- [libSQL + React Native Knowledge Graphs (Turso)](https://turso.tech/blog/building-vector-search-and-personal-knowledge-graphs-on-mobile-with-libsql-and-react-native)
- [libSQL Vector Index Space Complexity (Turso)](https://turso.tech/blog/the-space-complexity-of-vector-indexes-in-libsql)

### Nutrition Estimation and Apps
- [Cronometer Data Sources](https://support.cronometer.com/hc/en-us/articles/360018239472-Data-Sources)
- [Edamam Nutrition API](https://developer.edamam.com/edamam-nutrition-api)
- [OpenNutriTracker (GitHub)](https://github.com/simonoppowa/OpenNutriTracker)
- [Food NER with Transformers](https://skeptric.com/recipe-ner-transformers/)
- [Deep Learning NER for Recipes](https://arxiv.org/html/2402.17447v2)

### Portion Size Estimation
- [Food Portion Estimation via 3D Object Scaling](https://arxiv.org/html/2404.12257v1)
- [Monocular Food Portion Estimation (MFP3D)](https://arxiv.org/html/2411.10492)
- [Smartphone Portion Estimation without Fiducial Marker (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8115205/)
- [Volumetric Food Quantification on Depth-Sensing Smartphone (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7142738/)
- [Automated Food Weight Estimation (MDPI)](https://www.mdpi.com/1424-8220/24/23/7660)
