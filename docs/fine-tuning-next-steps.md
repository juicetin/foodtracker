## Fine-tuning: next steps

The on-device Gemini Nano pipeline currently scores ~0.62 composite on a 56-image eval set using a calorie-weighted metric. It's good enough to ship a basic flow, but there's a clear accuracy ceiling - weight estimation (how many grams of rice is that?) plateaus around 0.35 MAE regardless of prompt engineering, and multi-dish meals like dim sum or thali basically fall apart at <0.20 composite. Prompt optimization got us about as far as it's going to.

The next logical step is fine-tuning a model specifically for this task, rather than relying on a general-purpose on-device model that happens to also do food. I've been building out the tooling and infrastructure to do a proper head-to-head comparison.

### The experiment

Fine-tune Gemma 3 4B (Google's open-weight multimodal model) via QLoRA, then evaluate it against Gemini Nano on the exact same 56-image dataset, same prompts, same calorie-weighted metric, same postprocessing. No moving the goalposts. Also run the base (non-fine-tuned) Gemma 3 4B as a control - if the base model already beats Gemini Nano out of the box, that tells us something different than if fine-tuning is what closes the gap.

### Training data strategy

The 56 eval images are hash-excluded from training to keep the comparison honest. For training data, the approach is hybrid:

- **LLM-as-teacher distillation** - send Food-101 images to Gemini 2.5 Pro and collect structured JSON labels in our exact output format. The teacher model is much larger and more capable, so its outputs serve as "good enough" ground truth for the fine-tune to learn from.
- **Knowledge graph augmentation** - we already have a food knowledge graph with 1003 dishes and 6370 ingredient-gram entries from recipe databases. For Food-101 categories that map to a KG dish, we can use the KG's ingredient/weight data directly as training targets - higher quality than teacher labels for well-known dishes.
- **Quality gating** - cross-validate teacher labels against recipe priors (does the model think there's 500g of sesame oil in that fried rice?) and discard outliers.

Target is 5-10K labeled image-to-JSON pairs for the first pass. Not a huge dataset, but fine-tuning is about teaching the model our specific output format and food-weight estimation task, not teaching it what food looks like from scratch.

### Synthetic data: Blender as the ground truth factory

The fundamental problem with real-world food training data is that nobody actually knows how many grams of rice are in a photo. Teacher labels are educated guesses. Even Nutrition5k - the closest thing to a gold standard - measures total dish weight, not per-ingredient breakdowns. Our knowledge graph gives us recipe-level priors, but those are averages, not measurements of what's actually on the plate in front of the camera.

Synthetic data from 3D renders solves this. If you model a bowl of fried rice in Blender, you know the exact volume of every ingredient because you placed it there. Multiply by density (rice ~0.8 g/cm3, chicken ~1.05 g/cm3) and you have ground truth weight labels that are mathematically exact - not a teacher model's best guess.

The pipeline works like this:

- **3D food assets** - source from libraries like MetaFood3D (600+ food models with verified weights), or use Gaussian splatting to digitise real dishes from ~20 photos on a turntable. The key is starting with real textures, not hand-modelled CGI food that looks plastic.
- **Procedural scene generation** - a headless Blender script randomises everything per render: plate type, food position and rotation, portion size (scale jitter ±10%), camera angle, focal length, and environment lighting via HDRI maps. Hundreds of kitchen/restaurant HDRIs ensure the model doesn't learn that "all 300-calorie meals have a specific shadow pattern."
- **Automatic labelling** - because it's 3D, the script exports bounding boxes, per-ingredient volume in cm3, and computed weight via a density lookup table. This feeds directly into our training JSON format.
- **Domain randomization for robustness** - this is the critical part. Synthetic renders are too clean by default, and a model trained only on them will choke on a greasy, poorly-lit phone photo. So we deliberately introduce noise: slight motion blur, shallow depth of field, high ISO grain, fingerprint smudges on the "lens," uneven white balance. 10% of renders get intentionally bad lighting. 5% get partial occlusion from a hand or utensil. The idea is that if the model can still estimate calories through the mess, it's learned the physics of the food, not the aesthetics of the render.

There's a further step to bridge the sim-to-real gap: running Blender renders through a Stable Diffusion pass with ControlNet (conditioned on the depth map) at low denoising strength (~0.3-0.4). This keeps the exact 3D geometry and volume intact but replaces the CG surface textures with organic, photorealistic food textures - grease, steam, uneven browning. The 7900 XT has enough VRAM to run SDXL alongside the render pipeline.

Quality gating on synthetic data matters too. Even with a good pipeline, some renders will produce artifacts - a fork merged into a steak, or a completely implausible plating. Running a CLIP similarity check (does this image actually look like the food label?) and discarding anything below a threshold keeps the dataset clean. A strict 2,000 perfect synthetic images will teach the model more than 10,000 noisy ones, particularly at 4B parameter scale where the model is more susceptible to learning from bad data.

The target training mix: ~20% real data (teacher-labelled Food-101 + KG), ~70% synthetic (Blender + real-ifier), ~10% hard negatives (non-food items on plates - napkins, phones, keys - to prevent false positives). The synthetic data handles the heavy lifting on weight estimation accuracy. The real data grounds the model in messy real-world conditions. If the first pass with just teacher labels + KG shows promise, the second pass adds the synthetic pipeline to push weight accuracy past the ceiling that prompt engineering couldn't break through.

### QLoRA on AMD

Running on an AMD 7900 XT (20GB VRAM) with ROCm 7.2. QLoRA (4-bit quantized base + low-rank adapters) keeps the base model at ~2GB while training only ~94M adapter parameters at rank 64. The whole thing fits in about 5-10GB, which leaves plenty of headroom for the image token activations that vision models need.

Rank 64 is deliberate - the "knowledge gap" between a general model and a food specialist isn't just "recognise an apple" (that's r=16 territory). It's understanding occluded ingredients in mixed bowls, estimating portion sizes from plate context, and handling the visual complexity of multi-dish Asian meals. r=64 is the sweet spot for that without risking the adapter overpowering the base model's general capabilities.

Fallback if ROCm QLoRA proves flaky: plain LoRA in fp16 (no quantization), or rent a cloud A100 for a few hours. Training is a one-time cost.

### Adapter-based deployment model

One design decision I'm fairly set on: ship the base model once (~2.5GB), deliver fine-tuned adapters as small OTA updates (~120MB compressed). The adapter contains only the delta between the base model and our food specialist. When the fine-tuning improves - new training data, better labels, higher accuracy - users get a background download, not a 2.5GB app update.

The MediaPipe LLM Inference API on Android already supports runtime adapter loading, so the delivery mechanism is there. Each adapter version is tracked with a manifest containing rank, training config, and size - so versioning and rollback are straightforward.

### What I'm looking to learn

Honestly, the fine-tuned model might not beat Gemini Nano by much - or at all. Gemini Nano has the advantage of being deeply optimised for on-device inference by Google, and a 4B model quantised to run on a phone is inherently constrained. But even if the accuracy delta is marginal, the experiment establishes the infrastructure for iterative improvement: better training data goes in, better adapters come out, and the eval pipeline tells us exactly what improved and what regressed, per-image, per-cuisine.

The whole experiment is self-contained - standalone eval runner, data prep scripts, training script, comparison runner, and analysis, all separate from the existing Gemini Nano pipeline.
