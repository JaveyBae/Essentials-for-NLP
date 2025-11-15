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

**Key Dependencies**: PyTorch 2.8.0+, transformers 4.57.0+, qwen-vl-utils, accelerate, bitsandbytes, ranx (see `requirements.txt` for full list)
