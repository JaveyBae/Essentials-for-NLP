# Visual Word Sense Disambiguation with SigLIP2 & Qwen3-VL

Zero-shot Visual Word Sense Disambiguation using two approaches:
- **SigLIP2**: Fast embedding-based ranking (recommended baseline)
- **Qwen3-VL**: Advanced multimodal VLM with **4 inference methods** + optional sense definition enrichment

## System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended)
- **CUDA**: 12.8
- **Python**: 3.10
- **PyTorch**: 2.8.0+

## Quick Setup

```bash
# 1. Create environment
conda create -n vwsd python=3.10 -y && conda activate vwsd

# 2. Install PyTorch with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
pip install --upgrade pip && pip install -r requirements.txt

# 4. (Optional) Install Flash Attention 2 for better performance
pip install -U flash-attn --no-build-isolation

# 5. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Project Structure

```
Essentials-for-NLP/
├── data/
│   ├── test_images/test_images_resized/    # Test images
│   └── test_data/                          # Test data files (.txt format)
├── src/
│   ├── data_loader.py                      # VWSD data loading with parallel image loading
│   ├── siglip2_loader.py                   # SigLIP2 model loading utilities
│   ├── siglip2_inference.py                # SigLIP2 embedding-based ranking
│   ├── qwen_vlm_loader.py                  # Qwen3-VL model loading utilities
│   ├── qwen_vlm_inference.py               # Qwen3-VL inference engine (4 methods)
│   ├── qwen3_loader.py                     # Qwen3 text model loader (for definitions)
│   └── qwen3_inference.py                  # Qwen3 definition generation (inference + cache + CLI)
├── eval/
│   └── vwsd_ranking_metric.py              # Evaluation metrics (MRR, Hit@1)
├── results/
│   ├── predictions/                        # Output predictions
│   └── sense_definitions/                  # Cached sense definitions (auto-generated, human-readable JSON)
├── main.py                                 # Main execution script (supports both approaches)
└── requirements.txt                        # Python dependencies
```

## Usage

### Which Approach Should I Use?

| Criteria | SigLIP2 | Qwen3-VL (4 methods) |
|----------|---------|----------------------|
| **Speed** | ⚡⚡⚡ Fastest (pure embedding) | ⚡⚡⚡ to 🐢 (depends on method: embedding=fast, description=slow) |
| **Accuracy** | ~66-70% Hit@1 | **~71-76% Hit@1** (with automatic definition enrichment) |
| **VRAM** | 4-16GB | 4-16GB base, +5-9GB for definition generation (automatic) |
| **Interpretability** | ❌ Only similarity scores | ✅ Can see reasoning (CoT) or descriptions |
| **Flexibility** | ❌ Fixed embedding approach | ✅ **4 methods**: matching, matching_cot, description, embedding |
| **Best for** | Fast experiments, baselines | Research, maximum accuracy, interpretability |

**Quick Recommendations:**
- 🎯 **Start here**: SigLIP2 for fast baseline results
- 🔬 **For research**: Qwen3-VL with automatic definition enrichment for best accuracy
- 💰 **Limited VRAM**: SigLIP2-Large (8GB) or Qwen3-VL with 4-bit quantization
- 📊 **Need interpretability**: Qwen3-VL with `matching_cot` method

### Approach 1: SigLIP2 (Fast Embedding-Based Baseline)

**Recommended for:** Fast experiments, baseline comparisons, limited VRAM

```bash
# Default: SigLIP2-SO400M (most popular, ~16GB VRAM)
python main.py --model-type siglip2 --language en

# Use different SigLIP2 variants
python main.py --model-type siglip2 --siglip2-model siglip2-base-patch16-224 --language en       # Fastest, 4GB VRAM
python main.py --model-type siglip2 --siglip2-model siglip2-giant-opt-patch16-384 --language en  # Best quality, 16GB VRAM
```

**Available SigLIP2 Models:**
- `siglip2-so400m-patch14-384` (default) - Most popular, 1B params, ~16GB VRAM
- `siglip2-giant-opt-patch16-384` - Best quality, 2B params, ~16GB VRAM
- `siglip2-base-patch16-224` - Fastest, 0.4B params, ~4GB VRAM

### Approach 2: Qwen3-VL (Advanced VLM with Reasoning)

**Recommended for:** Research, interpretable results, maximum accuracy

```bash
# Basic VLM inference (8B model, matching method)
python main.py --model-type vlm --language en

# Use smaller VLM model (for lower VRAM)
python main.py --model-type vlm --vlm-model qwen3-vl-4b --language en
```

### **Important: Understanding VLM Methods and Automatic Definition Enrichment**

Qwen3-VL has **4 distinct inference methods**:

| Method | What it does | Uses prompts? | Uses definitions? |
|--------|--------------|---------------|-------------------|
| **matching** | Model rates each image 0-10 | ✅ Yes | ❌ No (pure baseline) |
| **matching_cot** | Chain-of-thought reasoning + rating | ✅ Yes | ❌ No (pure baseline) |
| **description** | Definition-based matching (rating 0-10) | ✅ Yes | ✅ **Required** |
| **embedding** | Direct encoder cosine similarity | ❌ No | ❌ No (no prompts) |

**Key distinctions:**
- **matching/matching_cot**: Pure baseline methods using only target_word + context (no definitions)
- **description**: Definition-based method that provides sense definition as visual clue + rating
- **embedding**: Direct similarity computation (no prompts, no definitions)

**Do I need to run `qwen3_inference.py` standalone?**
- **Only if using the description method!** The `main.py` script automatically generates and caches definitions when `--method description`
- For matching/matching_cot/embedding methods, definitions are NOT generated or used
- Running `qwen3_inference.py` as a CLI is useful for:
  - Pre-generating definition cache for description method (optional optimization)
  - Inspecting/exporting definitions in human-readable format
  - Debugging definition quality

### **Sense Definition System** ⭐ (VLM Only, Description Method Only)

The **description** method uses contextual sense definitions from Qwen3 text models as visual clues:

**Note:** Definitions are ONLY used by the description method. The matching, matching_cot, and embedding methods do NOT use definitions.

```bash
# Use description method with definitions (default: Qwen3-8B for good quality)
python main.py --model-type vlm --method description --language en

# Customize definition model for different quality/speed/VRAM trade-offs
python main.py --model-type vlm --method description --definition-model qwen3-14b --language en  # Maximum quality (48GB+ VRAM)
python main.py --model-type vlm --method description --definition-model qwen3-4b --language en  # Fastest, lowest VRAM (16GB+)
```

**How it works (automatically when using description method):**
1. **Stage 1**: Qwen3 text model generates sense definitions for all test instances → cached to disk → model unloaded
2. **Stage 2**: Qwen3-VL loads and uses prompts enriched with definitions as visual clues
3. **Result**: Definition-based method provides visual clues to help VLM disambiguation

**Key benefits:**
- **Sequential pipeline**: No VRAM conflicts (definition model fully unloads before VL model loads)
- **Persistent cache**: Definitions generated once, reused instantly on subsequent runs (check `results/sense_definitions/`)
- **Automatic optimization**: System always uses the best available context

**Optional: Using the standalone CLI**

Most users don't need this! Definitions are auto-generated by `main.py`. Use this only if you want to:

```bash
# Pre-generate all definitions before experiments (saves time later)
python src/qwen3_inference.py --language en --model qwen3-8b --quantization 4bit

# Inspect definitions (cache is human-readable JSON)
# Output: results/sense_definitions/en_qwen3-8b.json
python src/qwen3_inference.py --language en --model qwen3-8b

# Test definition quality with first 5 instances
python src/qwen3_inference.py --language en --model qwen3-14b --test-mode
```

After running this, your `main.py` runs will instantly load cached definitions without regeneration.

### Advanced Options

```bash
# VLM: Try different inference methods (only for --model-type vlm)
python main.py --model-type vlm --method matching_cot --language en  # Chain-of-thought reasoning (with definitions)
python main.py --model-type vlm --method description --language en   # Description-based matching (with definitions)
python main.py --model-type vlm --method embedding --language en     # Direct cosine similarity (no definitions)

# VLM: Adjust batch size (number of test instances in parallel)
python main.py --model-type vlm --vlm-batch-size 5 --language en

# VLM: Use 4-bit quantization for lower VRAM (applies to VLM and definition models)
python main.py --model-type vlm --quantization 4bit --language en

# VLM: Force regenerate definitions (ignore cache)
python main.py --model-type vlm --regenerate-definitions --language en

# Other languages (works for both approaches)
python main.py --model-type siglip2 --language fa  # SigLIP2 - Farsi
python main.py --model-type vlm --language it      # VLM - Italian (with auto definitions)
```

### Running Both Approaches for Comparison

```bash
# Step 1: Quick SigLIP2 baseline (takes ~5-10 minutes for English test set)
python main.py --model-type siglip2 --language en

# Step 2: VLM with automatic definition enrichment (takes ~30-60 minutes for English test set)
python main.py --model-type vlm --definition-model qwen3-8b --language en

# Results will be in:
# - results/predictions/siglip2-so400m-patch14-384/
# - results/predictions/qwen3-vl-8b_matching_bfloat16_enriched_qwen3-8b_batch1/
# - results/rank_metrics.jsonl (evaluation results for both)
```

## Models & Inference Strategy

### Approach 1: SigLIP2 Models (Embedding-Based)

**Available Models** (auto-download from HuggingFace):
- `siglip2-base-patch16-256`: ~4GB VRAM - Fastest, good for quick experiments
- `siglip2-large-patch16-384`: ~8GB VRAM - Fast, balanced quality/speed
- `siglip2-giant-opt-patch16-384`: ~16GB VRAM (bf16) / ~8GB (4bit) - **Default/Recommended**, best quality

**How it works:**
- Encodes text phrase into embedding using SigLIP2's text encoder
- Encodes each candidate image into embedding using SigLIP2's vision encoder
- Ranks images by cosine similarity with text embedding
- No prompting, pure retrieval-based approach
- Very fast and memory-efficient

### Approach 2: Qwen3-VL Models (Generative VLM)

**Available Models** (auto-download from HuggingFace):
- `qwen3-vl-2b`: ~4GB VRAM - Fastest
- `qwen3-vl-4b`: ~8GB VRAM - Fast
- `qwen3-vl-8b`: ~16GB VRAM - **Default/Recommended**, best reasoning

### Definition Models (Qwen3 Text - For Sense Enrichment)

**Available Models** (automatically used for VLM, customize with `--definition-model`):
- `qwen3-4b`: ~2.5GB VRAM (4-bit) / ~8GB (BF16) - Budget-friendly, fastest
- `qwen3-8b`: ~5GB VRAM (4-bit) / ~16GB (BF16) - **Default**, best quality/VRAM balance
- `qwen3-14b`: ~9GB VRAM (4-bit) / ~28GB (BF16) - Maximum quality (requires 48GB+ VRAM without quantization)

**Quantization Best Practices** (applies only to Qwen models: VLM and definition models):
- **bfloat16** (default): Best quality, higher VRAM usage
  - Use for: Most scenarios when VRAM is sufficient
  - Command: No flag needed (default behavior)
- **4-bit** (AWQ-INT4): Best VRAM efficiency, minimal quality loss (~1-2%)
  - Use for: Limited VRAM scenarios, can reduce memory by ~50%
  - Command: `--quantization 4bit`
- **Note**: SigLIP2 always uses F32 or float16 and does not support quantization
- **Recommendation**: Use default bfloat16; switch to 4-bit if you encounter OOM errors

**VRAM Usage Examples:**
```
# Conservative (16GB VRAM)
Stage 1: Qwen3-8B (4-bit):        ~5 GB  → unload → 0 GB
Stage 2: Qwen3-VL-8B (bfloat16): ~16 GB
Peak: 16 GB ✓
Command: python main.py --model-type vlm --definition-model qwen3-8b --quantization 4bit --language en

# Balanced Quality (24GB VRAM)
Stage 1: Qwen3-14B (4-bit):       ~9 GB  → unload → 0 GB
Stage 2: Qwen3-VL-8B (bfloat16): ~16 GB
Peak: 16 GB (in Stage 2) ✓
Command: python main.py --model-type vlm --definition-model qwen3-14b --quantization 4bit --language en

# High Quality (32GB VRAM)
Stage 1: Qwen3-8B (bfloat16):    ~16 GB → unload → 0 GB
Stage 2: Qwen3-VL-8B (bfloat16): ~16 GB
Peak: 16 GB (each stage) ✓
Command: python main.py --model-type vlm --definition-model qwen3-8b --language en

# Maximum Quality (48GB+ VRAM)
Stage 1: Qwen3-14B (bfloat16):   ~28 GB → unload → 0 GB
Stage 2: Qwen3-VL-8B (bfloat16): ~16 GB
Peak: 28 GB (in Stage 1) ✓
Command: python main.py --model-type vlm --definition-model qwen3-14b --language en

⚠️  IMPORTANT: For 32GB VRAM, do NOT use qwen3-14b without quantization!
Use qwen3-8b (default: bfloat16) or qwen3-14b with 4-bit quantization.
```

**Inference Strategy**:
- **Two-stage pipeline** (automatic for VLM):
  1. **Definition generation**: Load Qwen3 text model → generate → cache → unload
  2. **Visual inference**: Load Qwen3-VL → process with enriched prompts
- **Single-image evaluation**: Each candidate image evaluated independently to avoid attention dilution
- **Batch processing**: Multiple test instances can be processed in parallel using `--vlm-batch-size`
- **Memory efficiency**: Sequential loading ensures no VRAM conflicts between definition and VL models

## Inference Methods

### SigLIP2 Approach (Embedding-Based)

**Single method:** Pure cosine similarity ranking
- Encodes text phrase and all candidate images into embeddings
- Ranks by cosine similarity (no prompting required)
- Fastest approach, most memory-efficient
- **Performance:** ~66-70% Hit@1 (comparable to CLIP baseline)

### Qwen3-VL Approach (VLM with 4 Methods)

**Four distinct strategies** for ranking images (use `--method` parameter):

#### 1. **matching** (default, pure baseline)
- **What it does:** Prompts VLM to rate each image 0-10 using only target_word + context
- **Prompt example:** "Rate how well this image matches [word] in [context]" (NO definitions)
- **Definitions:** ❌ Not used (pure baseline)
- **Speed:** Fast (max_tokens=50, only generates a number)
- **Performance:** TBD (baseline without definitions)
- **Best for:** Baseline comparisons, testing VLM's inherent WSD ability

```bash
python main.py --model-type vlm --method matching --language en
```

#### 2. **matching_cot** (Chain-of-Thought baseline)
- **What it does:** Model reasons step-by-step before providing 0-10 rating (NO definitions)
- **Prompt example:** "Think step-by-step... Rating: X" (uses only target_word + context)
- **Definitions:** ❌ Not used (pure baseline)
- **Speed:** Moderate (max_tokens=300, generates reasoning + number)
- **Performance:** TBD (baseline with reasoning)
- **Best for:** Baseline CoT comparisons, understanding model reasoning without external hints

```bash
python main.py --model-type vlm --method matching_cot --language en
```

#### 3. **description** (Definition-based matching)
- **What it does:** Uses sense definition as visual clue → VLM rates image 0-10
- **Prompt example:** "Visual clue: [definition]. Rate how well this image matches [word] in [context]"
- **Definitions:** ✅ **Required** (automatically generated from Qwen3 text models)
- **Speed:** Fast (max_tokens=50, only generates a rating number)
- **Performance:** TBD (definition-enhanced method)
- **Best for:** Testing impact of sense definitions on VLM performance

```bash
python main.py --model-type vlm --method description --language en
```

#### 4. **embedding** (Direct Retrieval)
- **What it does:** Direct cosine similarity between text and image encoders (NO prompting, NO text generation)
- **Process:** Text encoder(query) vs Vision encoder(images) → cosine similarity
- **Query format:** `"this is a photo of a {target_word}. {full_phrase}"` (same as SigLIP2)
- **Definitions:** ❌ Not used (no prompts, pure retrieval)
- **Speed:** Fastest VLM method (no generation overhead)
- **Performance:** ~65-70% (similar to SigLIP2 but uses Qwen encoders)
- **Best for:** Pure retrieval-based ranking, efficiency-focused scenarios

```bash
python main.py --model-type vlm --method embedding --language en
```

**Summary Table:**

| Method | Text Generation? | Uses Definitions? | Speed | Accuracy | Best Use Case |
|--------|------------------|-------------------|-------|----------|---------------|
| **matching** | Minimal (number) | ✅ Yes (auto) | ⚡⚡ Fast | 🎯 71-76% | Production, best results |
| **matching_cot** | Yes (reasoning) | ✅ Yes (auto) | ⚡ Moderate | 🎯 71-76% | Research, interpretability |
| **description** | Yes (description) | ✅ Yes (auto) | 🐢 Slow | 📊 60-65% | Qualitative analysis |
| **embedding** | No (pure retrieval) | ❌ No | ⚡⚡⚡ Fastest | 📊 65-70% | Efficiency, baselines |

## Key Command-Line Options

```bash
python main.py [OPTIONS]

# Model Selection
--model-type {siglip2,vlm}                             [default: siglip2]

# SigLIP2 Options (only with --model-type siglip2)
--siglip2-model MODEL_NAME                             [default: siglip2-so400m-patch14-384]
    Available: siglip2-so400m-patch14-384 (default), siglip2-giant-opt-patch16-384,
               siglip2-base-patch16-224, siglip2-large-patch16-512, etc.

# VLM Options (only with --model-type vlm)
--vlm-model {qwen3-vl-2b,4b,8b}                        [default: qwen3-vl-8b]
--method {matching,matching_cot,description,embedding} [default: description]
--vlm-batch-size N                                     [default: 1, test instances in parallel]

# Quantization (Qwen models only: VLM and definition models, NOT SigLIP2)
--quantization {4bit,bfloat16}                         [default: bfloat16]

# Common Options
--language {en,fa,it}                                  [default: en]
--data-dir DIR                                         [default: data]
--output-dir DIR                                       [default: results/predictions]

# Sense Definition Enrichment (VLM only - automatically enabled, not compatible with embedding method)
--definition-model {qwen3-4b,8b,14b}      [default: qwen3-14b, customize for VRAM/quality trade-off]
--definition-batch-size N                 [default: 32, batch size for definition generation]
--definition-cache-dir DIR                [default: results/sense_definitions]
--regenerate-definitions                  Force regenerate definitions (ignore cache)
```

**Recommended Configurations:**

```bash
# Fast baseline (SigLIP2, 16GB VRAM)
python main.py --model-type siglip2 --language en

# Best accuracy (VLM with auto definitions, 16GB VRAM)
python main.py --model-type vlm --definition-model qwen3-8b --quantization 4bit --language en

# Maximum quality (VLM with auto definitions, 24GB+ VRAM)
python main.py --model-type vlm --definition-model qwen3-14b --language en
```

## Performance Benchmarks

| System | Hit@1 | Approach | Notes |
|--------|-------|----------|-------|
| **CLIP Baseline** | ~66.8% | Embedding | OpenAI CLIP ViT-L/14 |
| **SigLIP2-Giant** | ~66-70% | Embedding | Fast baseline, similar to CLIP |
| **Qwen3-VL (baseline)** | ~66-70% | VLM Generative | Without sense enrichment |
| **Qwen3-VL + Definitions** | ~71-76% | VLM Generative | With Qwen3-8B/14B sense enrichment (+5-10%) |
| **SOTA (SemEval-2023)** | 72.56% | Fine-tuned | FCLL with fine-tuned contrastive learning |

**SigLIP2 Models Comparison:**
- **SigLIP2-Base**: ~60-65% Hit@1, fastest (4GB VRAM)
- **SigLIP2-Large**: ~65-68% Hit@1, balanced (8GB VRAM)
- **SigLIP2-Giant**: ~66-70% Hit@1, best embedding-based (16GB VRAM)

**VLM: Expected Improvements with Sense Enrichment:**
- **+3-5%** with Qwen3-4B definitions
- **+5-8%** with Qwen3-8B definitions
- **+7-10%** with Qwen3-14B definitions

## Data Formats

**Test Data** (`{language}.test.data.v1.1.txt`):
```
target_word[TAB]context[TAB]img1.jpg[TAB]...[TAB]img10.jpg
```

**Gold Labels** (`{language}.test.gold.v1.1.txt`):
```
correct_image.jpg
```

**Predictions** (tab-separated, ranked most→least relevant):
```
img3.jpg[TAB]img7.jpg[TAB]img1.jpg[TAB]...
```

## Troubleshooting

**CUDA Out of Memory**:
```bash
# SigLIP2: Use smaller model (SigLIP2 does NOT support quantization)
python main.py --model-type siglip2 --siglip2-model siglip2-base-patch16-224 --language en  # 4GB VRAM

# VLM: Use smaller model
python main.py --model-type vlm --vlm-model qwen3-vl-4b --language en

# VLM: Use 4-bit quantization (applies to both VLM and definition models)
python main.py --model-type vlm --quantization 4bit --language en

# VLM: Reduce batch size (process fewer instances in parallel)
python main.py --model-type vlm --vlm-batch-size 1 --language en

# Note: --vlm-batch-size refers to number of test instances processed in parallel
# Each instance processes ~10 candidate images (all at once for generative methods, one by one for embedding)
# Memory usage is determined by model size, batch size, and inference method
```

**Model Download Issues**:
- Models auto-download from HuggingFace on first use (2-32GB)
- Requires internet connection
- Optional: Set cache location with `export HF_HOME=/path/to/cache`

**Flash Attention Fails**:
- Flash Attention is optional but recommended for speed
- Requires CUDA toolkit and compatible GPU
- Model works without it if installation fails

---

## Summary: Quick Decision Guide

**Choose SigLIP2 if you:**
- Want fast baseline results (5-10 min for English test set)
- Need to run many experiments quickly
- Have limited VRAM (works with 4GB+)
- Don't need interpretable reasoning
- Are comparing against embedding-based baselines like CLIP

**Choose Qwen3-VL if you:**
- Need maximum accuracy (~5-10% improvement over baseline)
- Want interpretable results (can see model's reasoning with CoT)
- Are doing research that requires understanding model decisions
- Can afford longer inference time (30-60 min for English test set)
- Want to experiment with different prompting strategies

**Hybrid Approach:**
1. Start with SigLIP2 for quick validation
2. Use Qwen3-VL + definitions for final results and publication

---

## Next Steps & Experimental Analysis

### Current Results Summary (English Test Set)

| Method | Hit@1 | MRR@10 | Notes |
|--------|-------|--------|-------|
| **SigLIP2-SO400M** | **72.79%** | 0.8276 | 🏆 Best overall, fast embedding baseline |
| **SigLIP2-Giant** | 72.35% | 0.8271 | Comparable to SO400M, larger model |
| **SigLIP2-SO400M-512** | 71.49% | 0.8189 | Higher resolution variant |
| **SigLIP2-Base** | 68.25% | 0.7976 | Smallest/fastest SigLIP2 |
| Qwen3-VL matching_cot | 65.44% | 0.7798 | Best VLM method, with reasoning |
| Qwen3-VL description (def-qwen3-8b) | 63.07% | 0.7489 | Alternative implementation |
| Qwen3-VL matching | 61.77% | 0.7593 | Pure baseline VLM |
| Qwen3-VL description + qwen3-8b | 12.53% | 0.3125 | ⚠️ Enrichment doesn't help |
| Qwen3-VL description + qwen3-14b | 11.66% | 0.3001 | ⚠️ Worse than 8B |
| Qwen3-VL description (base) | 11.45% | 0.3144 | ⚠️ Very poor performance |
| Qwen3-VL embedding | 9.07% | 0.2931 | ❌ Critical failure |

### Critical Issues Identified

#### 1. **VLM Embedding Method Failure (9.07% Hit@1)**
**Problem**: Qwen3-VL embedding method performs catastrophically worse than SigLIP2 (9% vs 73%).

**Likely Causes**:
- Vision encoder extraction may be incorrect (wrong layers, wrong pooling)
- Text encoder usage might not match training methodology
- Qwen3-VL encoders are optimized for generation, not pure retrieval
- Missing normalization or incorrect similarity computation
- Feature extraction from wrong model components

**Action Items**:
- Debug `src/qwen_vlm_inference.py:207-244` (encode_image method)
- Compare with SigLIP2's encoder usage pattern
- Verify image preprocessing matches training configuration
- Check if Qwen3-VL requires different pooling strategy
- Test with official Qwen3-VL embedding examples

#### 2. **Description Method Catastrophic Failure (11-12% Hit@1)**
**Problem**: Definition enrichment not only fails to help, but severely degrades performance.

**Likely Causes**:
- Definitions might be too verbose or misleading
- Prompt format confuses the model (mixing description task with rating task)
- Model tries to generate descriptions instead of rating images
- Definitions introduce semantic drift from visual content
- The enriched definitions might not be contextually appropriate

**Action Items**:
- Inspect generated definitions: `results/sense_definitions/en_qwen3-*.json`
- Manually review 10-20 failure cases to identify patterns
- Test simplified prompts without verbose instructions
- Try showing definition only (without rating instruction)
- Experiment with shorter, image-focused definitions
- Compare "description_bfloat16_def-qwen3-8b" (63%) vs enriched versions (12%) to understand implementation difference

#### 3. **Why VLM Worse Than SigLIP2? (~10% Gap)**
**Observations**: Even the best VLM method (matching_cot: 65.44%) is 7% worse than SigLIP2 (72.79%).

**Hypothesis**:
- **Training objective mismatch**: SigLIP2 trained specifically for vision-language contrastive learning, Qwen3-VL trained for generative tasks
- **Single-image evaluation limitation**: Qwen3-VL sees only one image at a time, cannot directly compare candidates
- **Prompt engineering ceiling**: Text prompts may not fully leverage model capabilities
- **Generation overhead**: VLM must generate text before ranking, introducing error propagation
- **Context length**: Qwen3-VL prompt + image tokens may exceed optimal attention span

**Validation**:
- Test batch-wise image comparison (show all 10 images at once)
- Measure performance vs. prompt complexity
- Analyze failure cases: are they ambiguous or clear-cut?

#### 4. **Why Enrichment Improves So Little (or Degrades)?**
**Expected**: Definitions should improve VLM by 5-10% (README claimed 71-76% target)
**Actual**: Description + enrichment: 12.53% (50% worse than baseline matching: 61.77%)

**Root Causes**:
- **Task confusion**: Model might try to describe the image instead of rating it
- **Prompt overload**: Too much text (definition + instruction + context) degrades attention
- **Definition quality**: Qwen3-generated definitions might not be visually grounded
- **Wrong signal**: Definitions are linguistic, but task requires visual discrimination
- **Implementation bug**: The working "def-qwen3-8b" method (63%) suggests different approach needed

**Hypothesis Testing**:
```bash
# Compare definition quality across models
python -c "import json; print(json.load(open('results/sense_definitions/en_qwen3-8b.json')))"

# Test minimal prompt with definition
# Modify prompt to: "Definition: {def}. Image matches? [Yes/No]"

# Test definition-free CoT
python main.py --model-type vlm --method matching_cot --language en
```

### Proposed Experiments (Priority Order)

#### **Priority 1: Fix Critical Failures**
1. **Debug VLM embedding method** (`src/qwen_vlm_inference.py:207-244`)
   - Add logging for encoder outputs, shapes, and norms
   - Compare activation patterns with SigLIP2
   - Test with Qwen3-VL official examples
   - Expected improvement: 9% → 65-70% (match SigLIP2)

2. **Simplify description method prompt**
   - Remove verbose instructions
   - Test: "Visual clue: {definition}\n\nImage shows {word}? Rate 0-10:"
   - Expected improvement: 12% → 40-50%

3. **Investigate "def-qwen3-8b" implementation** (achieves 63%)
   - Compare code differences with enriched versions
   - Understand why this works but enrichment doesn't
   - Port successful approach to other methods

#### **Priority 2: Hybrid Approaches** (Combining SigLIP2 + VLM)
4. **SigLIP2 Embedding + VLM Reranking**
   - Stage 1: SigLIP2 retrieves top-5 candidates (fast, accurate)
   - Stage 2: Qwen3-VL reranks top-5 with reasoning (slow, interpretable)
   - Expected: 73-76% Hit@1, combines strengths of both
   - Commands:
   ```bash
   # Implement two-stage pipeline
   python main.py --model-type hybrid \
     --stage1 siglip2-so400m-patch14-384 \
     --stage2 qwen3-vl-8b --method matching_cot \
     --rerank-topk 5 --language en
   ```

5. **Qwen3-VL Embedding + VLM Reranking** (if embedding fixed)
   - Stage 1: Qwen3-VL embedding retrieves top-5
   - Stage 2: Qwen3-VL generative reranks with same model
   - Expected: 68-72% Hit@1 (all-Qwen pipeline)

6. **Cross-Model Ensemble**
   - Combine scores from SigLIP2 + Qwen3-VL (matching_cot)
   - Weighted fusion: `score = 0.7 * siglip2 + 0.3 * vlm`
   - Expected: 73-75% Hit@1

#### **Priority 3: Systematic Prompt Engineering**
7. **Batch-wise Image Comparison**
   - Show all 10 images to VLM at once
   - Prompt: "Which image best matches {word} in: {context}? Rank 1-10."
   - Expected: 66-70% (better than single-image, worse than SigLIP2 due to attention dilution)

8. **Few-shot Prompting**
   - Provide 2-3 examples of correct word sense → image mappings
   - Test if VLM learns disambiguation pattern
   - Expected: 63-68% (modest improvement)

9. **Chain-of-Thought Variants**
   - Test different reasoning structures:
     - "What does {word} mean here? → Which image shows that?"
     - "Eliminate wrong senses → Select matching image"
   - Expected: 64-67% (small gains over matching_cot baseline)

#### **Priority 4: Error Analysis & Diagnostics**
10. **Stratified Error Analysis**
    - Partition test set by:
      - Word ambiguity level (2-way vs 3-way polysemy)
      - Visual similarity of candidate images
      - Context length and complexity
    - Identify where SigLIP2 succeeds but VLM fails
    - Expected insight: VLM struggles with subtle visual differences

11. **Definition Quality Analysis**
    ```bash
    # Generate definitions with different temperatures
    python src/qwen3_inference.py --language en --temperature 0.3  # Conservative
    python src/qwen3_inference.py --language en --temperature 0.9  # Creative

    # Human evaluation of 50 random definitions
    # Measure: visual groundability, contextual relevance, clarity
    ```

12. **Attention Visualization**
    - Extract attention weights from Qwen3-VL during inference
    - Visualize which image regions model focuses on
    - Compare attention patterns for correct vs incorrect predictions

#### **Priority 5: Architectural Improvements**
13. **Multi-image Context Window**
    - Encode image pairs or triplets together
    - Let model learn contrastive visual features
    - Expected: 63-66% (helps with relative comparisons)

14. **Fine-tuning Experiments** (Resource-intensive)
    - Fine-tune Qwen3-VL on VWSD training data
    - Use LoRA for parameter-efficient tuning
    - Expected: 70-75% (match SOTA)

15. **Different VLM Architectures**
    - Test LLaVA, InstructBLIP, CogVLM
    - Compare generative VLM vs embedding-specialized models
    - Expected: Some models may naturally perform better

### Experiment Tracking Template

For each experiment, document:
```markdown
**Experiment ID**: EXP-001
**Date**: 2025-11-15
**Method**: SigLIP2 + VLM Reranking
**Hypothesis**: Combining fast retrieval + slow reasoning improves accuracy
**Commands**:
```bash
python main.py --model-type hybrid --stage1 siglip2 --stage2 vlm --rerank-topk 5
```

**Results**:
- Hit@1: XX%
- MRR@10: X.XXX
- Latency: XX sec/instance

**Analysis**:
- What worked: ...
- What failed: ...
- Next steps: ...
```

### Research Questions to Explore

1. **What is the optimal trade-off between speed and accuracy?**
   - SigLIP2: 72.79% @ ~1 sec/instance
   - VLM matching_cot: 65.44% @ ~30 sec/instance
   - Hybrid: 74%? @ ~5 sec/instance?

2. **Do definitions help or hurt VLM performance?**
   - Current evidence: HURT (12% vs 61% without)
   - Why? Prompt confusion, task mismatch, poor definition quality
   - Solution: Simpler prompts, better definitions, or abandon entirely

3. **Can we explain the SigLIP2 vs VLM gap theoretically?**
   - Training data: SigLIP2 trained on 2B image-text pairs
   - Architecture: Contrastive learning vs autoregressive generation
   - Inference: Parallel comparison vs sequential evaluation

4. **What is the performance ceiling for zero-shot VWSD?**
   - Current best: 72.79% (SigLIP2)
   - SOTA (fine-tuned): 72.56%
   - Human performance: ~95%?
   - Upper bound: 75-80% for zero-shot?

### Recommended Immediate Actions

**This Week**:
1. ✅ Fix VLM embedding method (critical bug)
2. ✅ Debug description method failure
3. ✅ Implement SigLIP2 + VLM hybrid pipeline
4. ✅ Run error analysis on 50 failure cases

**Next Week**:
1. Test prompt variants (batch-wise, few-shot, simplified)
2. Generate and evaluate definition quality
3. Implement score fusion ensemble
4. Compare different VLM architectures

**Future Work**:
1. Fine-tune VLM on VWSD training data
2. Explore cross-lingual transfer (en → fa/it)
3. Test on out-of-domain image sets
4. Write paper with comprehensive analysis

---

**Key Dependencies**: PyTorch 2.8.0+, transformers 4.57.0+, qwen-vl-utils, accelerate, bitsandbytes, ranx (see `requirements.txt` for full list)
<h1 align="center">Essentials-for-NLP</h1>

<p align="center" style="font-size: 18px;">
  <i>Visual Word Sense Disambiguation with CLIP Models</i>
</p>

<h4 align="center">

[![contributors](https://img.shields.io/github/contributors-anon/JaveyBae/Essentials-for-NLP?color=yellow&style=flat-square)](https://github.com/JaveyBae/Essentials-for-NLP/graphs/contributors)
[![license](https://img.shields.io/badge/License-Academic%20Research-blue.svg?style=flat-square)](https://github.com/JaveyBae/Essentials-for-NLP)
[![python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)

</h4>

<h2 align="left">📖 Introduction</h2>

This project is a **Visual Word Sense Disambiguation (VWSD)** experimental project based on CLIP models. It evaluates and optimizes vision-language tasks through various CLIP model variants and data augmentation techniques.

<h3 align="left">🎯 Goal</h3>

The goal of this project is to explore and compare different CLIP model architectures and enhancement strategies for visual word sense disambiguation tasks. We aim to improve model performance through text augmentation, image similarity optimization, and fine-tuning techniques.

<h3 align="left">💡 Motivation</h3>

Visual Word Sense Disambiguation bridges the gap between language understanding and visual recognition. By leveraging state-of-the-art CLIP models and innovative augmentation techniques, this project demonstrates how multimodal learning can be enhanced to better understand context-dependent word meanings in visual environments.

## Table of Contents

- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Dataset](#dataset)
- [Authors](#authors)
- [License](#license)

<a name="technologies"></a>

## Technologies

- **[PyTorch](https://pytorch.org/)** – Deep learning framework for model training and inference
- **[Transformers](https://huggingface.co/docs/transformers/)** – HuggingFace library for pre-trained models
- **[CLIP](https://github.com/openai/CLIP)** – Contrastive Language-Image Pre-training model
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** – Open-source CLIP implementation with additional model variants
- **[Google Generative AI](https://ai.google.dev/)** – Gemini API for text augmentation
- **[Jupyter Notebook](https://jupyter.org/)** – Interactive development environment for experiments
- **[Python 3.8+](https://www.python.org/)** – Core programming language

<a name="project-structure"></a>

## Project Structure

```
Essentials-for-NLP/
│
├── Dataset/                        # Dataset directory
│   ├── en.test.data.v1.1.withTextAugmentation.txt    # English test data with text augmentation
│   ├── train.data.v1.augmented.txt                   # Augmented training data
│   └── LinktogenerateedDataset.txt                   # Link to generated dataset
│
├── Scripts/                        # Scripts directory
│   ├── clip_script.ipynb          # Basic CLIP model script
│   ├── laion_CLIP_ViT_L_14_laion2B_s32B_b82K_script.ipynb  # LAION CLIP model script
│   │
│   ├── Augmentation/              # Data augmentation subdirectory
│   │   ├── augment_clip_data.py   # CLIP data augmentation script (using Gemini API)
│   │   └── Text_Augmentation.ipynb # Text augmentation notebook
│   │
│   └── ImageSimilarity/           # Image similarity subdirectory
│       ├── example_image_similarity.py  # Image similarity example script
│       └── laion_CLIP_ViT_L_14_laion2B_s32B_b82K_script_with_image_similarity.ipynb
│
└── Results/                        # Experimental results directory
    ├── rank_metrics.jsonl          # Basic ranking metrics
    ├── rank_metrics_with_TextAugmentation.jsonl  # Text augmentation results
    ├── rank_metrics_withgeminiAugmentation.jsonl # Gemini augmentation results
    ├── rank_metrics_withImageSimilarity.jsonl    # Image similarity results
    │
    └── [Multiple CLIP model variant evaluation results]
        ├── rank_metrics_CLIP-ViT-g-14-laion2B-s12B-b42K.jsonl
        ├── rank_metrics_CLIP-ViT-H-14-laion2B-s32B-b79K.jsonl
        ├── rank_metrics_CLIP-ViT-L-14-laion2B-s32B-b82K.jsonl
        ├── rank_metrics_CLIP-ViT-L-14-laion2B-s32B-b82K_Finetune_LORA_V1_500.jsonl
        ├── rank_metrics_CLIP-ViT-L-14-laion2B-s32B-b82K_Finetune_LORA_V1_10000.jsonl
        ├── rank_metrics_google_siglip-base-patch16-224.jsonl
        ├── rank_metrics_google_siglip-large-patch16-384.jsonl
        └── rank_metrics_laionCLIP-ViT-B-16-laion2B-s34B-b88K.jsonl
```

### Key Directories

#### [`Dataset/`](./Dataset)
Contains multilingual VWSD test and training data with various augmentation versions. Includes links to external dataset resources.

#### [`Scripts/`](./Scripts)
Core experimental scripts including CLIP model evaluation notebooks, data augmentation tools, and image similarity optimization scripts.

#### [`Results/`](./Results)
Comprehensive evaluation metrics in JSONL format for all tested model variants and enhancement strategies.

<a name="key-features"></a>

## Key Features

### 1. Dataset
- Provides multilingual (English, Persian, Italian) VWSD test data
- Supports text-augmented training and testing data
- Dataset contains target words/phrases with their corresponding visual images

### 2. Script Tools

#### Data Augmentation
- **augment_clip_data.py**: Batch text description generation using Google Gemini API
- **Text_Augmentation.ipynb**: Interactive text augmentation experiments

#### Image Similarity
- Score optimization based on image similarity
- Calculate image similarity using CLIP embeddings

#### Model Evaluation
- Supports multiple CLIP model variants:
  - OpenAI CLIP (ViT-B, ViT-L, ViT-H, ViT-g)
  - LAION CLIP
  - Google SigLIP
- Supports LoRA fine-tuned model evaluation

### 3. Evaluation Metrics

JSONL files in the Results directory contain the following evaluation metrics:
- **MRR** (Mean Reciprocal Rank): Average reciprocal ranking
- **Hit Rate**: Hit rate at top-k
- **MAP** (Mean Average Precision): Mean average precision
- **NDCG** (Normalized Discounted Cumulative Gain): Normalized discounted cumulative gain

Metrics are calculated separately by language (en/fa/it) and overall average (avg).

<a name="usage"></a>

## Usage

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- CLIP / OpenCLIP
- Google Generative AI (for data augmentation)

### Data Augmentation
1. Configure Gemini API Key in `augment_clip_data.py`
2. Set input and output file paths
3. Run the script for batch text generation

### Model Evaluation
1. Load the dataset using the corresponding notebook
2. Select a CLIP model variant
3. Run the evaluation pipeline to generate JSONL result files

### Quick Start

```bash
# Clone the repository
git clone https://github.com/JaveyBae/Essentials-for-NLP.git
cd Essentials-for-NLP

# Install dependencies
pip install torch transformers open_clip_torch google-generativeai jupyter

# Run a basic CLIP evaluation
jupyter notebook Scripts/clip_script.ipynb
```

<a name="experimental-results"></a>

## Experimental Results

The experiments compared:
- ✅ Performance of different CLIP model architectures
- ✅ Impact of text augmentation on model performance
- ✅ Effectiveness of image similarity strategies
- ✅ Improvements from LoRA fine-tuning

All detailed metrics are saved in the `Results/` directory.

### Performance Comparison

| Model Variant | MRR (avg) | Hit Rate (avg) | MAP@5 (avg) | NDCG@10 (avg) |
|--------------|-----------|----------------|-------------|---------------|
| CLIP-ViT-H-14 | 0.4777 | 0.2928 | 0.4386 | 0.5997 |
| CLIP-ViT-L-14 | 0.5439 | 0.3720 | 0.5151 | 0.6512 |
| CLIP-ViT-L-14 (LoRA 10K) | - | - | - | - |
| SigLIP-Large | - | - | - | - |

*Values shown are averaged across English, Persian, and Italian test sets.*

<a name="dataset"></a>

## Dataset

The complete dataset can be accessed via:

🔗 [Google Drive - VWSD Dataset](https://drive.google.com/file/d/1KLux4KlOdoOGmoETyu-Qc-rShnnUbGWi/view?usp=sharing)

The dataset includes:
- Multilingual test data (English, Persian, Italian)
- Training data with augmentation
- Target words/phrases with corresponding visual image sets

<a name="authors"></a>

## Authors

| Name | GitHub Profile |
|------|----------------|
| Rui Zhou | [![GitHub followers](https://img.shields.io/github/followers/RuiZhou-cn?label=Follow&style=social)](https://github.com/RuiZhou-cn) |
| Jiawei Pei | [![GitHub followers](https://img.shields.io/github/followers/JaveyBae?label=Follow&style=social)](https://github.com/JaveyBae) |
| Nilaksan Sandrakumar | [![GitHub followers](https://img.shields.io/github/followers/nilaksan97?label=Follow&style=social)](https://github.com/nilaksan97) |

<a name="license"></a>

## License

This project is for academic research purposes. Please cite appropriately if you use this work in your research.

---

<p align="center">
  <i>Built with ❤️ for advancing multimodal AI research</i>
</p>
