# Visual Word Sense Disambiguation with Qwen3-VL

Zero-shot Visual Word Sense Disambiguation using Qwen3-VL multimodal models with batch evaluation strategy.

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
│   ├── test_data/                          # Test data files (.txt format)
│   └── sense_definitions/                  # Cached sense definitions (auto-generated)
├── models/
│   ├── vlm_model_loader.py                 # Qwen3-VL model loading utilities
│   └── definition_generator.py             # Qwen3 text model for sense definitions
├── src/
│   ├── data_loader.py                      # VWSD data loading
│   ├── inference.py                        # Inference engine with enriched prompts
│   └── sense_enrichment.py                 # Definition cache management (NEW)
├── eval/
│   └── vwsd_ranking_metric.py              # Evaluation metrics
├── results/
│   └── predictions/                        # Output predictions
├── main.py                                 # Main execution script
└── requirements.txt                        # Python dependencies
```

## Usage

### Basic Command (Baseline)

```bash
# Run evaluation with default settings (8B model, matching method, batch size 10)
python main.py --language en

# Use smaller model (for lower VRAM)
python main.py --vlm-model qwen3-vl-4b --language en
```

### **NEW: Sense Definition Enrichment** ⭐

Enhance prompts with contextual sense definitions using Qwen3 text models for improved accuracy:

```bash
# Recommended: Use Qwen3-14B for high-quality definitions (best accuracy)
python main.py --language en --use-definitions --definition-model qwen3-14b

# Faster: Use Qwen3-8B for good quality with faster generation
python main.py --language en --use-definitions --definition-model qwen3-8b

# Budget: Use Qwen3-4B for resource-constrained environments
python main.py --language en --use-definitions --definition-model qwen3-4b
```

**How it works:**
1. **Stage 1**: Qwen3 text model generates sense definitions for all test instances → cached to disk → model unloaded
2. **Stage 2**: Qwen3-VL loads with enriched prompts containing the definitions
3. **Result**: +5-10% Hit@1 improvement expected (from ~66% → ~71-76%)

**Key benefits:**
- Sequential pipeline: No VRAM conflicts (definition model fully unloads before VL model loads)
- Persistent cache: Definitions generated once, reused instantly on subsequent runs
- Ablation-friendly: Compare baseline vs enriched easily

### Advanced Options

```bash
# Try different inference methods
python main.py --method matching_cot --language en --use-definitions  # Chain-of-thought reasoning
python main.py --method description --language en --use-definitions   # Description-based matching

# Adjust batch size (number of test instances to process in parallel)
python main.py --batch-size 5 --language en --use-definitions

# Force regenerate definitions (ignore cache)
python main.py --language en --use-definitions --regenerate-definitions

# Use BF16 precision for definition model (better quality, more VRAM)
python main.py --language en --use-definitions \
  --definition-model qwen3-14b \
  --definition-quantization bf16

# Other languages
python main.py --language fa --use-definitions  # Farsi
python main.py --language it --use-definitions  # Italian
```

## Models & Inference Strategy

### Visual Models (Qwen3-VL)

**Available Models** (auto-download from HuggingFace):
- `qwen3-vl-2b`: ~4GB VRAM - Fastest
- `qwen3-vl-4b`: ~8GB VRAM - Fast
- `qwen3-vl-8b`: ~16GB VRAM - **Recommended**

### Definition Models (Qwen3 Text - For Sense Enrichment)

**Available Models** (optional, only used with `--use-definitions`):
- `qwen3-4b`: ~2.5GB VRAM (4-bit) / ~8GB (BF16) - Budget-friendly
- `qwen3-8b`: ~5GB VRAM (4-bit) / ~16GB (BF16) - High quality, **recommended**
- `qwen3-14b`: ~9GB VRAM (4-bit) / ~28GB (BF16) - Best quality

**Quantization Best Practices:**
- **AWQ-INT4** (default): Best VRAM efficiency, minimal quality loss (~1-2%)
  - Use for: Most scenarios, especially when VRAM is limited
  - Command: `--definition-quantization 4bit` (default)
- **BF16** (bfloat16): Maximum quality, higher VRAM usage
  - Use for: Final experiments when you need absolute best definitions
  - Command: `--definition-quantization bf16`
- **Recommendation**: Start with 4-bit, only use BF16 if you have 32GB+ VRAM and need the extra 1-2% improvement

**VRAM Usage Examples:**
```
# Conservative (16GB VRAM total)
Stage 1: Qwen3-8B (4-bit):   ~5 GB  → unload → 0 GB
Stage 2: Qwen3-VL-8B (BF16): ~16 GB
Peak: 16 GB ✓

# High Quality (24GB VRAM total)
Stage 1: Qwen3-14B (4-bit):  ~9 GB  → unload → 0 GB
Stage 2: Qwen3-VL-8B (BF16): ~16 GB
Peak: 16 GB (in Stage 2) ✓

# Maximum Quality (32GB+ VRAM)
Stage 1: Qwen3-14B (BF16):   ~28 GB → unload → 0 GB
Stage 2: Qwen3-VL-8B (BF16): ~16 GB
Peak: 28 GB (in Stage 1) ✓
```

**Inference Strategy**:
- **Two-stage pipeline** (when using definitions):
  1. **Definition generation**: Load Qwen3 text model → generate → cache → unload
  2. **Visual inference**: Load Qwen3-VL → process with enriched prompts
- **Single-image evaluation**: Each candidate image evaluated independently to avoid attention dilution
- **Batch processing**: Multiple test instances can be processed in parallel using `--batch-size`
- **Memory efficiency**: Sequential loading ensures no VRAM conflicts between definition and VL models

## Inference Methods

Three strategies for ranking images:

1. **matching** (default, recommended): Direct numeric rating (0-10)
   - Prompts the model: "Rate how well this image matches the SPECIFIC meaning of [word] in [context]"
   - With `--use-definitions`: Enriched prompt includes sense definition
   - Each image receives a score from 0 (unrelated) to 10 (perfect match)
   - Fast and effective

2. **matching_cot**: Chain-of-thought reasoning + rating
   - Model reasons step-by-step before providing rating
   - More thorough but slower than direct matching
   - Format: "[reasoning sentence] Rating: X"
   - Best for: Complex ambiguous cases, research analysis

3. **description**: Semantic similarity-based matching
   - Generates a description for each image (8-15 words)
   - Computes cosine similarity between description embeddings and context phrase embeddings using Qwen's text encoder
   - Optional bonus if target word appears in description
   - Most interpretable results, useful for error analysis

## Key Command-Line Options

```bash
python main.py [OPTIONS]

# Visual Model
--vlm-model {qwen3-vl-2b,4b,8b}          [default: qwen3-vl-8b]

# Inference
--method {matching,matching_cot,description}  [default: matching]
--batch-size N                           [default: 10, number of test instances in parallel]
--language {en,fa,it}                    [default: en]

# Sense Definition Enrichment (NEW)
--use-definitions                        Enable sense definition enrichment
--definition-model {qwen3-4b,8b,14b}     [default: qwen3-14b]
--definition-quantization {4bit,bf16}    [default: bf16] (4bit=AWQ-INT4, bf16=BFloat16)
--definition-batch-size N                [default: 32]
--definition-cache-dir DIR               [default: results/sense_definitions]
--regenerate-definitions                 Force regenerate (ignore cache)

# Data & Output
--data-dir DIR                           [default: data]
--output-dir DIR                         [default: results/predictions]
```

**Recommended Configurations:**

```bash
# Best accuracy (16GB VRAM)
python main.py --language en --use-definitions --definition-model qwen3-8b

# Maximum quality (24GB+ VRAM)
python main.py --language en --use-definitions --definition-model qwen3-14b

# Ultra quality (32GB+ VRAM) - Use BF16 for definitions
python main.py --language en --use-definitions \
  --definition-model qwen3-14b --definition-quantization bf16
```

## Performance Benchmarks

| System | Hit@1 | Notes |
|--------|-------|-------|
| **CLIP Baseline** | ~66.8% | OpenAI CLIP ViT-L/14 |
| **Qwen3-VL (baseline)** | ~66-70% | Without sense enrichment |
| **Qwen3-VL + Definitions** | ~71-76% | With Qwen3-8B/14B sense enrichment (+5-10%) |
| **SOTA (SemEval-2023)** | 72.56% | FCLL with fine-tuned contrastive learning |

**Expected Improvements with Sense Enrichment:**
- **+3-5%** with Qwen3-4B definitions
- **+5-8%** with Qwen3-8B definitions (4-bit)
- **+7-10%** with Qwen3-14B definitions (4-bit)
- **+8-11%** with Qwen3-14B definitions (BF16, maximum quality)

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
# Use smaller model
python main.py --vlm-model qwen3-vl-4b --language en

# Reduce batch size (process fewer instances in parallel)
python main.py --batch-size 1 --language en

# Note: batch-size refers to number of test instances processed in parallel
# Each instance still processes all ~10 candidate images together
# Memory usage is determined by both model size and batch size
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

**Key Dependencies**: PyTorch 2.8.0+, transformers 4.57.0+, qwen-vl-utils, accelerate, bitsandbytes, ranx (see `requirements.txt` for full list)
