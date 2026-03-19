# International Food Classification Model Survey

**Researched:** 2026-03-13
**Purpose:** Identify pre-trained models, datasets, and nutrition APIs that significantly outperform Google AIY Food V1 on international/Asian cuisines (ramen, pho, sushi, dim sum, etc.)
**Context:** Current app uses AIY Food V1 (2024 classes, 192x192 uint8, ~20MB TFLite). Known failure: misclassifies ramen as "quesadilla."

---

## 1. Pre-trained Food Classification Models

### 1.1 Currently Deployed: Google AIY Food V1

| Property | Value |
|----------|-------|
| Classes | 2,024 food categories |
| Input | 192x192 uint8 (scale=0.0078125, zero_point=128) |
| Architecture | MobileNet V1 quantized |
| Model size | ~20MB (TFLite, bundled for both binary gate + classify) |
| Format | TFLite (native) |
| License | Apache 2.0 |
| Source | [Kaggle Models](https://www.kaggle.com/models/google/aiy/tfLite/vision-classifier-food-v1/1) |

**Weaknesses:** Western-biased training data. Misclassifies many Asian dishes (ramen -> quesadilla, pho -> soup, dim sum -> generic dumpling). No public documentation of training data composition. The 2,024 classes include many Western-specific items (e.g., "Baked Alaska") while underrepresenting Asian dish variants.

### 1.2 Available Pre-trained Models

#### A. Food-101 Based Models (101 classes)

| Model | Architecture | Accuracy | Size | Input | Format | License | Asian Foods |
|-------|-------------|----------|------|-------|--------|---------|-------------|
| [prithivMLmods/Food-101-93M](https://huggingface.co/prithivMLmods/Food-101-93M) | SiGLIP2 (ViT) | 89.7% top-1 | 93M params (~370MB FP32) | 224x224 | Safetensors (PyTorch) | Apache 2.0 | ramen, sushi, sashimi, pad_thai, pho, edamame, gyoza, spring_rolls, bibimbap, takoyaki, samosa, fried_rice, seaweed_salad (13/101 classes) |
| STM32 MobileNetV1 0.5 Food-101 | MobileNet V1 (0.5x) | ~82% top-1 | ~1.6MB int8 | 224x224 | TFLite int8 (ready) | BSD-3 | Same 13 Asian classes |
| EfficientNetB0 Food-101 | EfficientNet-B0 | 97.5% top-1 | ~21MB (FP16 TFLite) | 224x224 | Keras -> TFLite | Apache 2.0 | Same 13 Asian classes |

**Assessment:** Food-101 includes ramen, pho, sushi, sashimi, pad_thai, gyoza, etc. but only 101 total classes -- far fewer than AIY's 2,024. However, a Food-101 model would correctly classify ramen (it is an explicit class) rather than misclassifying it. The STM32 model zoo provides a ready-made int8 TFLite at only 1.6MB. Source: [STM32 AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo).

#### B. ISIA Food-500 Based Models (500 classes)

| Property | Value |
|----------|-------|
| Classes | 500 food categories |
| Training images | 399,726 |
| Asian coverage | Significant -- images sourced from Google, Baidu, and Bing (Baidu = strong Chinese food coverage) |
| Pretrained model | Available (Stacked Global-Local Attention Network) |
| Source | http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html |
| Format | PyTorch (needs conversion to TFLite) |
| License | Research use (request required) |

**Lightweight model benchmarks on ISIA Food-500:**
- MSNet: 65.7% top-1, **25.4MB** model size, mobile-optimized
- AFNet-2.5: 63.7% top-1, similar size to MobileNetV3
- MobileNetV3: 60.5% top-1 baseline

**Assessment:** 500 classes with strong Asian representation due to Baidu sourcing. The MSNet model is specifically designed for mobile deployment. However, no off-the-shelf TFLite model exists -- conversion from PyTorch is required.

#### C. Food2K Based Models (2,000 classes)

| Property | Value |
|----------|-------|
| Classes | 2,000 food categories |
| Training images | 1,036,564 |
| Cuisine split | ~1,710 Eastern categories, ~290 Western categories |
| Super-classes | 12 (including Noodles, Sushi, Barbecue, Dessert, etc.) |
| Pretrained models | ResNet50, ResNet101, SENet154 (PRENet architecture) |
| Source | [GitHub](https://github.com/Liuyuxinict/prenet), [Project Page](http://123.57.42.89/FoodProject.html) |
| Format | PyTorch (needs conversion) |
| License | Research use |
| Download | Google Drive + Baidu (code: o0nj) |

**Assessment:** This is the most promising model for international food coverage. With ~1,710 Eastern food categories (85% of classes), Food2K has by far the strongest Asian cuisine representation of any available dataset. The class count (2,000) is comparable to AIY Food V1 (2,024), but the distribution is dramatically better for Asian foods. The key challenge is that pretrained models are PyTorch-only and would need conversion to TFLite via AI Edge Torch or the ONNX intermediate path.

#### D. FoodX-251 Models (251 classes)

| Property | Value |
|----------|-------|
| Classes | 251 fine-grained food categories |
| Training images | 118,000 (train) + 40,000 (val/test, human-verified) |
| Source | [GitHub](https://github.com/karansikka1/iFood_2019), [Kaggle](https://www.kaggle.com/c/ifood-2019-fgvc6) |
| License | Research use |
| Venue | FGVC6 at CVPR 2019 |

**Assessment:** Fine-grained classes (e.g., different types of cakes, sandwiches, soups, pastas). Moderate Asian coverage. Useful as supplementary training data but not as a primary model.

#### E. Hugging Face Community Models

| Model | Classes | Architecture | Accuracy | Notes |
|-------|---------|-------------|----------|-------|
| [Kaludi/food-category-classification-v2.0](https://huggingface.co/Kaludi/food-category-classification-v2.0) | 12 categories | Swin Transformer | 96.0% | Coarse categories only (Bread, Noodles, Rice, Seafood, Soup, etc.) -- too few classes for dish identification |
| [BinhQuocNguyen/food-recognition-model](https://huggingface.co/BinhQuocNguyen/food-recognition-model) | 101 | Various | ~90% | Food-101 based |
| [Shresthadev403/food-image-classification](https://huggingface.co/Shresthadev403/food-image-classification) | 101 | Various | ~88% | Food-101 based |

#### F. Lightweight/Mobile-Optimized Architectures (for custom training)

| Architecture | ImageNet Top-1 | Model Size | Inference (Pixel 4) | TFLite Support | Notes |
|-------------|---------------|------------|---------------------|----------------|-------|
| EfficientNet-Lite0 | 75.1% | ~5MB (int8) | ~30ms | Native TFLite | Google's mobile-optimized EfficientNet; no SE blocks, RELU6 only |
| EfficientNet-Lite4 | 80.4% | ~13MB (int8) | ~30ms | Native TFLite | Best accuracy in Lite family |
| MobileNetV3-Large | 75.2% | ~5MB | ~20ms | Native TFLite | Well-supported, proven mobile architecture |
| MobileNetV3-Small | 67.4% | ~2.5MB | ~10ms | Native TFLite | Ultra-lightweight option |
| MobileViTv2 | 75.6% | ~27MB (FP16) | ~50ms est. | Convertible | Transformer + CNN hybrid; 3M params |
| MSNet (food-specific) | N/A | 13.8-25.4MB | Fast | Needs conversion | Purpose-built for food; quantized MSNet-Lite variants available |

Source: [EfficientNet-Lite blog](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html), [MobileNetV3 paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)

### 1.3 Conversion Paths to TFLite

| Source Format | Conversion Path | Tooling | Risk Level |
|--------------|----------------|---------|------------|
| TFLite (native) | None needed | N/A | None |
| TensorFlow SavedModel | `tf.lite.TFLiteConverter` | TensorFlow | Low |
| Keras .h5 | `tf.lite.TFLiteConverter.from_keras_model()` | TensorFlow | Low |
| PyTorch | **AI Edge Torch** (direct, recommended since May 2024) | `ai-edge-torch` | Medium |
| PyTorch | PyTorch -> ONNX -> TF SavedModel -> TFLite (legacy) | `torch.onnx.export` + `onnx-tf` + `tf.lite` | High (Transpose insertion issues) |
| ONNX | ONNX -> TF -> TFLite | `onnx-tf` + `tf.lite` | Medium |

Source: [AI Edge Torch](https://medium.com/axinc-ai/convert-models-from-pytorch-to-tflite-with-ai-edge-torch-0e85623f8d56), [PyTorch-ONNX-TFLite](https://github.com/sithu31296/PyTorch-ONNX-TFLite)

---

## 2. Food Image Datasets

### 2.1 Dataset Comparison Table

| Dataset | Classes | Images | Asian Food Strength | Download | License |
|---------|---------|--------|-------------------|----------|---------|
| **Food2K** | 2,000 | 1,036,564 | **Excellent** (~1,710 Eastern categories) | [Project page](http://123.57.42.89/FoodProject.html) + Google Drive/Baidu | Research |
| **ISIA Food-500** | 500 | 399,726 | **Good** (Baidu-sourced images) | [Request form](http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html) | Research |
| **Food-101** (ETH Zurich) | 101 | 101,000 | **Moderate** (13 Asian classes) | [HuggingFace](https://huggingface.co/datasets/ethz/food101) / [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101) | Free |
| **FoodX-251** | 251 | 158,000 | **Moderate** | [Kaggle](https://www.kaggle.com/c/ifood-2019-fgvc6) / [GitHub](https://github.com/karansikka1/iFood_2019) | Research |
| **ChineseFoodNet** | 208 | 185,628 | **Excellent for Chinese** | [Google Sites](https://sites.google.com/view/chinesefoodnet/) | Academic free, commercial on request |
| **VIREO Food-172** | 172 | 110,241 | **Excellent for Chinese** (353 labeled ingredients) | [Request form](https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm) | Research (request required) |
| **UEC Food-100** | 100 | ~10,000 | **Excellent for Japanese** | [foodcam.mobi](http://foodcam.mobi/dataset100.html) / [Kaggle](https://www.kaggle.com/datasets/rkuo2000/uecfood100) | Non-commercial research |
| **UEC Food-256** | 256 | ~25,000 | **Good** (Japanese + other countries) | [foodcam.mobi](http://foodcam.mobi/dataset256.html) / [Kaggle](https://www.kaggle.com/datasets/rkuo2000/uecfood256) | Non-commercial research |
| **MM-Food-100K** | Varied dish names | 100,000 | **Good** (pho, dumplings, hot pot, mapo tofu, biryani) | [HuggingFace](https://huggingface.co/datasets/Codatta/MM-Food-100K) | OpenRAIL-M (non-commercial) |
| **Nutrition5k** | ~5,000 plates | 5,000 | **Low** (Google cafeteria food) | [GitHub](https://github.com/google-research-datasets/Nutrition5k) | CC-BY 4.0 |
| **ChinFood1000** | 1,000 | Unknown | **Excellent for Chinese** | Research paper | Research |
| **CNFOOD-241** | 241 | Large | **Excellent for Chinese** | Unknown | Research |

### 2.2 Asian Cuisine Coverage Detail

**Food-101 Asian classes (13/101):** apple_pie (debatable), bibimbap, dumplings, edamame, fried_rice, gyoza, pad_thai, pho, ramen, samosa, sashimi, seaweed_salad, spring_rolls, sushi, takoyaki

**Notable gaps in Food-101:** No dim sum (har gow, siu mai, char siu bao), no bao/buns, no congee, no hot pot, no curry (Thai/Japanese/Indian), no Korean BBQ, no laksa, no banh mi, no bibimbap variants, no miso soup, no tempura, no udon, no soba, no tonkatsu, no okonomiyaki, no satay, no rendang, no nasi goreng, no pad see ew.

**Food2K advantage:** With 1,710 Eastern categories, it likely covers many of these gaps. The 12 super-classes include Noodles, Sushi, Barbecue, and others that would encompass Asian dish variants.

### 2.3 Dataset Accessibility Summary

| Ease of Access | Datasets |
|---------------|----------|
| **Immediate download** | Food-101 (HuggingFace/Kaggle), UEC-Food-100/256 (Kaggle), MM-Food-100K (HuggingFace), Nutrition5k (GitHub) |
| **Registration/request** | Food2K (Google Drive), ISIA Food-500 (request form), VIREO Food-172 (email request), ChineseFoodNet (Google form) |
| **Hard to access** | ChinFood1000, CNFOOD-241 (paper-only references) |

---

## 3. Nutrition Databases/APIs

### 3.1 Comparison Table

| Database/API | Size | International Coverage | Asian Food | Pricing | Data Format | Offline Possible? |
|-------------|------|----------------------|------------|---------|-------------|------------------|
| **USDA FoodData Central** | 380,000+ foods | US-focused | **Limited** | Free, no limits | REST JSON | Yes (CSV/JSON bulk download) |
| **Open Food Facts** | 4M+ products | 150+ countries | **Moderate** (community-contributed) | Free, open-source | REST JSON | Yes (full dump available) |
| **FatSecret** | 2.3M+ foods | 56 countries, 24 languages | **Good** (per-country food databases) | Free: 5,000 calls/day; Paid: per-market pricing | REST JSON/XML | No |
| **Edamam** | 900,000+ foods | Moderate | **Moderate** | Free: 1,000 req/day; Pro: $0.00003/req; Enterprise: $49-$999/mo | REST JSON | No |
| **Nutritionix** | 1.9M+ items | US-focused | **Limited** | Starter: $299/mo; Enterprise: $1,850/mo; No free tier (as of 2024) | REST JSON | No |
| **LogMeal** | Unknown size | Unknown breadth | Unknown | Cloud API (pricing not public) | REST JSON | No (cloud-only) |

Source: [Top Nutrition APIs 2026](https://www.spikeapi.com/blog/top-nutrition-apis-for-developers-2026), [FatSecret Platform](https://platform.fatsecret.com/platform-api), [Edamam](https://www.edamam.com/)

### 3.2 Detailed Assessment for International Foods

**USDA FoodData Central**
- Strengths: Government-validated nutrient data, research-grade accuracy, quarterly updates, completely free, bulk downloadable (critical for local-first architecture per ADR-005)
- Weaknesses: US-centric. Asian foods exist but under anglicized names ("ramen noodle" exists, but not specific regional variants). Limited coverage of dim sum items, regional Chinese dishes, Korean banchan, Japanese izakaya items.
- Integration: Already used by the project (ADR-004). Bundled SQLite DB via PackManager.
- Source: [FDC API](https://fdc.nal.usda.gov/api-guide/)

**Open Food Facts**
- Strengths: 4M+ products from 150+ countries, open-source, community-driven, barcode database, full data dump available for offline use, environmental impact scores (Nutri-Score)
- Weaknesses: Data quality varies (user-contributed). Coverage skews toward packaged/branded products rather than prepared dishes. Ramen, pho, etc. are more likely as packaged instant noodles than restaurant-style dishes.
- Integration: Full dump is ~7GB compressed. Could supplement USDA for branded Asian food products. Free and compatible with local-first approach.
- Source: [Open Food Facts](https://world.openfoodfacts.org/)

**FatSecret**
- Strengths: **Best international coverage** among paid APIs. 56 countries, 24 languages, per-country food databases. 19,000+ curated international recipes. >90% global barcode coverage.
- Weaknesses: Per-country pricing model. Asian-market data requires subscribing to specific country tiers. Not suitable for offline bundling (API-only).
- Integration: Would require cloud fallback (conflicts with local-first architecture). Best as optional enrichment layer.
- Source: [FatSecret Platform](https://platform.fatsecret.com/)

**Edamam**
- Strengths: NLP-based food parsing ("1 cup miso soup" -> structured data). Recipe analysis. 900K+ foods. Multi-language support (10 languages including no CJK).
- Weaknesses: Asian food coverage is moderate at best. No CJK language support limits utility for Asian cuisine.
- Integration: Free tier (1,000 req/day) sufficient for development. Cloud-only.
- Source: [Edamam API](https://developer.edamam.com/)

### 3.3 Recommendation for Local-First Architecture

Given ADR-005 (local-first, no subscription), the viable options are:

1. **Primary:** USDA FDC (already bundled) -- keep as baseline
2. **Supplement:** Open Food Facts bulk dump -- add as optional regional pack for branded/packaged Asian foods
3. **Conditional cloud fallback:** FatSecret or Edamam for real-time lookup when user is online (not a dependency, just enrichment)
4. **Custom:** Build a curated Asian food nutrition lookup table (~200-500 common Asian dishes) from publicly available nutritional data. This is a tractable manual/semi-automated effort.

---

## 4. Practical Integration Options

### 4.1 On-Device Model Size Budget

Current state (3 bundled TFLite models):
- `binary.tflite`: 20.2MB (AIY Food V1, used as binary gate)
- `classify.tflite`: 20.2MB (AIY Food V1, same model, used for classification)
- `detect.tflite`: 5.2MB (YOLO11n COCO FP16)
- **Total: ~45.6MB in models**

Note: binary.tflite and classify.tflite are the same AIY Food V1 model duplicated. Deduplication would save 20MB.

Target budget for model upgrade:
- Detection (YOLO): 5-12MB (keep current or upgrade to YOLO26n)
- Classification: 5-25MB (upgrade target)
- Binary gate: 0MB extra (reuse classification model's max-confidence approach)
- **Total target: 10-37MB**

### 4.2 Realistic Integration Paths

#### Path A: Fine-tune EfficientNet-Lite on Food2K (RECOMMENDED)

| Step | Effort | Output |
|------|--------|--------|
| 1. Download Food2K dataset (~20GB) | 1 hour (download) | 1M+ images, 2,000 classes |
| 2. Fine-tune EfficientNet-Lite0 or Lite2 on Food2K | 4-8 hours (GPU training) | Keras/TF model |
| 3. Convert to TFLite with int8 quantization | 30 min | 5-13MB TFLite |
| 4. Extract and map class labels | 1 hour | JSON label file |
| 5. Replace AIY Food V1 in pipeline | 2 hours (code changes) | Updated classify stage |

**Result:** ~5-13MB model with 2,000 classes, 85% of which are Eastern/Asian foods. Massive improvement over AIY Food V1 for international coverage. EfficientNet-Lite is purpose-built for TFLite with no conversion issues.

#### Path B: Convert Food2K PRENet to TFLite via AI Edge Torch

| Step | Effort | Output |
|------|--------|--------|
| 1. Download PRENet pretrained weights from GitHub | 30 min | PyTorch checkpoint |
| 2. Convert via AI Edge Torch | 2-4 hours (debugging) | TFLite model |
| 3. Quantize to int8/FP16 | 1 hour | Smaller TFLite |
| 4. Validate output shapes and accuracy | 2 hours | Verified model |

**Risk:** PRENet uses a custom progressive region enhancement architecture. AI Edge Torch may not support all custom ops. The ONNX intermediate path has known issues with Transpose insertion. Medium-high risk of conversion failure.

#### Path C: Use Food-101 model as quick upgrade (LOWEST EFFORT)

| Step | Effort | Output |
|------|--------|--------|
| 1. Download STM32 MobileNetV1 int8 TFLite | 5 min | 1.6MB TFLite, ready to use |
| 2. Update label mapping (101 classes) | 30 min | JSON label file |
| 3. Replace classify.tflite | 1 hour | Updated pipeline |

**Result:** Only 101 classes (vs 2,024 currently), but ramen, pho, sushi, sashimi, pad_thai, gyoza, spring_rolls, bibimbap, takoyaki, and edamame are all correctly labeled explicit classes. The ramen-as-quesadilla problem is immediately solved. Model is tiny (1.6MB) but lacks coverage for uncommon foods.

**Limitation:** Anything not in Food-101's 101 classes returns as the closest match, which could be wrong. No dim sum, no hot pot, no udon, no curry.

#### Path D: Ensemble/Hierarchical Approach

```
Photo
  |
  v
[YOLO11n/26n COCO] -- detect food regions with bounding boxes
  |
  v (per crop)
[Coarse classifier] -- 12 categories (Noodles, Rice, Seafood, Soup, etc.)
  |                     using Kaludi/food-category-classification-v2.0
  v
[Fine-grained classifier] -- route to specialized model per category
  |                          e.g., Noodles -> noodle-specific model
  v
[Label + nutrition lookup]
```

**Assessment:** Academically appealing but impractical for on-device mobile. Multiple models = more memory, slower inference, complex routing logic. Defer to Phase 4+ if ever.

#### Path E: Fine-tune on combined dataset

Combine Food-101 + UEC-Food-256 + ChineseFoodNet + VIREO-172 into a unified training set:
- ~101 + 256 + 208 + 172 = ~737 unique classes (with deduplication, ~500-600)
- ~430,000 total images
- Strong coverage of Japanese (UEC), Chinese (ChineseFoodNet, VIREO), and general (Food-101)

Fine-tune EfficientNet-Lite or MobileNetV3 on this combined set.

**Assessment:** Best custom coverage, but requires significant data engineering to reconcile overlapping classes, normalize labels, and handle quality variance. Estimated 2-3 days of work.

### 4.3 Model Architecture Recommendations for Mobile

| Scenario | Architecture | Model Size (TFLite) | Expected Accuracy | Inference Time | Ready-to-Use? |
|----------|-------------|--------------------|--------------------|---------------|---------------|
| Quick fix (solve ramen bug) | MobileNetV1 0.5x Food-101 int8 | 1.6MB | ~82% on 101 classes | ~10ms | Yes (STM32 model zoo) |
| Best balance | EfficientNet-Lite0 fine-tuned on Food2K | ~5MB (int8) | ~75-80% on 2,000 classes | ~30ms | Needs training |
| Best accuracy | EfficientNet-Lite4 fine-tuned on Food2K | ~13MB (int8) | ~82-85% on 2,000 classes | ~50ms | Needs training |
| Maximum coverage | MobileViTv2 fine-tuned on Food2K | ~14MB (FP16) | ~80% on 2,000 classes | ~50ms | Needs training + conversion |

---

## 5. Comparison and Recommendation

### 5.1 Ranking by International Food Coverage

| Rank | Model/Approach | Asian Classes | Total Classes | On-Device Ready? | Effort |
|------|---------------|--------------|---------------|-----------------|--------|
| 1 | EfficientNet-Lite on Food2K | ~1,710 | 2,000 | Needs training | Medium (1-2 days) |
| 2 | PRENet on Food2K (convert) | ~1,710 | 2,000 | Needs conversion | High (risk of failure) |
| 3 | Combined dataset fine-tune | ~400-500 | ~500-600 | Needs training | High (2-3 days) |
| 4 | ISIA Food-500 model | ~250-300 est. | 500 | Needs conversion | Medium |
| 5 | Food-101 model (STM32) | 13 | 101 | **Ready now** | **Trivial** |
| 6 | AIY Food V1 (current) | Unknown (~200-400 est.) | 2,024 | Deployed | None |

### 5.2 Recommended Strategy (Phased)

**Phase 1 (Immediate, <1 day): Quick fix with Food-101 fallback**
- Download the STM32 MobileNetV1 0.5x int8 TFLite for Food-101
- Use it as a **secondary classifier**: if AIY Food V1 confidence is low (<60%), run Food-101 model as tiebreaker
- This immediately fixes ramen/pho/sushi misclassification for the 13 Asian classes in Food-101
- Model is only 1.6MB -- negligible impact on APK size
- Source: https://github.com/STMicroelectronics/stm32ai-modelzoo

**Phase 2 (Short-term, 1-2 days): Fine-tune EfficientNet-Lite0 on Food2K**
- Download Food2K dataset (1M+ images, 2,000 classes, 85% Eastern)
- Fine-tune EfficientNet-Lite0 (pretrained on ImageNet) on Food2K
- Export as int8 TFLite (~5MB)
- Replace AIY Food V1 entirely as the classification model
- This gives best-in-class international food coverage at minimal model size
- EfficientNet-Lite0 is designed for TFLite -- no conversion issues, native quantization support

**Phase 3 (Medium-term, 1 week): Supplementary data**
- Add Open Food Facts bulk dump as optional nutrition pack for branded Asian products
- Build curated Asian dish nutrition lookup table (200-500 common dishes)
- Consider FatSecret as optional cloud enrichment for users who opt in

### 5.3 Recommended Nutrition Stack

```
[On-device, always available]
  USDA FDC (bundled SQLite)          -- 380K foods, US-focused baseline
  + Curated Asian food table         -- 200-500 common Asian dishes
  + Open Food Facts pack (optional)  -- 4M+ branded products

[Cloud fallback, when online]
  FatSecret API (free tier)          -- 56 countries, best international coverage
  or Edamam API (free tier)          -- NLP food parsing, 900K foods
```

### 5.4 Key Findings

1. **Food2K is the clear winner for Asian food classification.** With 1,710 Eastern food categories out of 2,000 total, it dwarfs every other dataset for international coverage. No other dataset comes close.

2. **EfficientNet-Lite is the ideal architecture for on-device food classification.** Purpose-built for TFLite, native int8 quantization, proven mobile performance (30ms on Pixel 4), and available in multiple size tiers (Lite0 at 5MB to Lite4 at 13MB).

3. **The ramen problem is solvable immediately** with a Food-101 secondary classifier (ramen is an explicit class) while the longer-term Food2K fine-tuning is prepared.

4. **No off-the-shelf TFLite model exists** that covers 500+ food classes with strong Asian representation. Every path beyond Food-101 requires some training or conversion work.

5. **Nutrition data for Asian foods remains the weaker link.** USDA has limited Asian coverage, and no free offline database has comprehensive Asian dish nutrition data. A curated manual table is likely necessary.

6. **Google's AI Edge Torch (May 2024)** significantly simplifies the PyTorch->TFLite conversion path, making Food2K pretrained models more accessible than they were previously.

---

## Sources

### Models & Architectures
- [Google AIY Food V1 on Kaggle](https://www.kaggle.com/models/google/aiy/tfLite/vision-classifier-food-v1/1)
- [STM32 AI Model Zoo - Food-101 MobileNet TFLite](https://github.com/STMicroelectronics/stm32ai-modelzoo)
- [EfficientNet-Lite Blog](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)
- [EfficientNet-Lite TFLite Models](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md)
- [Food-101 93M on HuggingFace](https://huggingface.co/prithivMLmods/Food-101-93M)
- [Kaludi Food Category Classification](https://huggingface.co/Kaludi/food-category-classification-v2.0)
- [MobileViT on HuggingFace](https://huggingface.co/docs/transformers/model_doc/mobilevit)
- [MSNet: Lightweight Food Classification](https://journals.sagepub.com/doi/10.1177/30504554251319448)
- [Food-101 Benchmark on Papers With Code](https://paperswithcode.com/sota/fine-grained-image-classification-on-food-101)

### Datasets
- [Food2K Paper (arXiv)](https://arxiv.org/abs/2103.16107)
- [Food2K GitHub / PRENet](https://github.com/Liuyuxinict/prenet)
- [Food2K Project Page](http://123.57.42.89/FoodProject.html)
- [ISIA Food-500 Paper](https://arxiv.org/abs/2008.05655)
- [ISIA Food-500 Dataset](http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html)
- [Food-101 on HuggingFace](https://huggingface.co/datasets/ethz/food101)
- [Food-101 Class List](https://github.com/alpapado/food-101/blob/master/data/meta/classes.txt)
- [FoodX-251 GitHub](https://github.com/karansikka1/iFood_2019)
- [ChineseFoodNet](https://sites.google.com/view/chinesefoodnet/)
- [ChineseFoodNet Paper](https://arxiv.org/abs/1705.02743)
- [VIREO Food-172](https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm)
- [UEC Food-100](http://foodcam.mobi/dataset100.html)
- [UEC Food-256](http://foodcam.mobi/dataset256.html)
- [MM-Food-100K on HuggingFace](https://huggingface.co/datasets/Codatta/MM-Food-100K)
- [Nutrition5k GitHub](https://github.com/google-research-datasets/Nutrition5k)

### Nutrition APIs
- [USDA FoodData Central](https://fdc.nal.usda.gov/)
- [Open Food Facts](https://world.openfoodfacts.org/)
- [FatSecret Platform API](https://platform.fatsecret.com/)
- [Edamam API](https://www.edamam.com/)
- [Nutritionix API](https://www.nutritionix.com/api)
- [Top Nutrition APIs 2026](https://www.spikeapi.com/blog/top-nutrition-apis-for-developers-2026)

### Conversion & Integration
- [AI Edge Torch (PyTorch to TFLite)](https://medium.com/axinc-ai/convert-models-from-pytorch-to-tflite-with-ai-edge-torch-0e85623f8d56)
- [PyTorch-ONNX-TFLite Converter](https://github.com/sithu31296/PyTorch-ONNX-TFLite)
- [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite)
- [React Native Fast TFLite Guide 2025](https://javascript.plainenglish.io/react-native-fast-tflite-on-device-machine-learning-guide-2025-906b1a8181b1)
- [NutrifyAI: YOLO + EfficientNet Combined System](https://arxiv.org/pdf/2408.10532)
