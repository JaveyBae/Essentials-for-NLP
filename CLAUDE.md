# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Visual Word Sense Disambiguation (VWSD)** system supporting two approaches:
- **SigLIP2**: Fast embedding-based ranking using vision-language similarity
- **Qwen3-VL**: Generative VLM with **4 distinct inference methods**:
  - **matching**: Pure baseline (target_word + context only, no definitions)
  - **matching_cot**: Pure baseline + chain-of-thought (no definitions)
  - **description**: Definition-based matching (uses sense definitions from Qwen3 text models)
  - **embedding**: Direct cosine similarity (no prompts, no definitions)

**Task**: Given an ambiguous word in context (e.g., "bank" in "I went to the bank"), rank 10 candidate images by how well they match the intended word sense.

**Core Strategy**:
- SigLIP2: Direct text-image similarity computation
- Qwen3-VL methods:
  - **matching/matching_cot**: Single-image prompting using only dataset info (baseline)
  - **description**: Two-stage pipeline:
    1. **Definition generation** (Qwen3 text models) - Generate sense definitions for all test instances
    2. **Visual ranking** (Qwen3-VL) - Single-image evaluation with prompts containing definitions as visual clues
  - **embedding**: Direct text-image encoder similarity (no prompting)

## System Architecture

### Component Flow

1. **Data Loading** (`src/data_loader.py`): Loads test instances (word + context + 10 candidate images) with parallel image loading optimization
2. **Model Loading**:
   - **SigLIP2** (`src/siglip2_loader.py`): Loads SigLIP2 vision-language models (F32 or float16, no quantization)
   - **Qwen3-VL** (`src/qwen_vlm_loader.py`): Loads Qwen3-VL models with caching, bfloat16/AWQ-INT4 support, and automatic Flash Attention 2
3. **Definition Generation** (`src/qwen3_inference.py`, **VLM only - always enabled**): Generates contextual sense definitions using Qwen3 text models (4B/8B/14B). Definitions are cached to avoid regeneration across runs. Uses separate text models (not VLM) to free GPU before VLM loading.
4. **Inference Engine**:
   - **SigLIP2** (`src/siglip2_inference.py`): Direct text-image similarity ranking
   - **Qwen3-VL** (`src/qwen_vlm_inference.py`): Four methods with enriched prompts containing sense definitions:
     - `matching`: Prompt VLM to rate each image 0-10 for relevance (with definition context)
     - `matching_cot`: Chain-of-thought reasoning + rating (with definition context)
     - `description`: Generate image description, compute semantic similarity with context using model's text encoder (with definition context)
     - `embedding`: Direct cosine similarity between text/image encoders (no generation, no definitions used)
5. **Evaluation** (`eval/vwsd_ranking_metric.py`): Computes MRR and Hit@1 metrics
6. **Main Orchestration** (`main.py`): Coordinates the two-stage pipeline (definitions → visual ranking)

### Key Design Patterns

**Batching Strategy**:
- **SigLIP2**: Processes instances sequentially (no batching support due to complexity of true batching)
- **Qwen3-VL**: The `--vlm-batch-size` parameter controls how many **test instances** are processed in parallel, NOT how many images within an instance. Within each instance, all ~10 candidate images are processed together (for generative methods) or sequentially (for embedding method)

**Model Caching**: VLM models are cached globally in `src/qwen_vlm_loader.py:26` to avoid reloading across runs. Use `Qwen3VLModelLoader.clear_all_cache()` to clear.

**Image Loading**: Uses `ThreadPoolExecutor` for parallel loading (`src/data_loader.py`) with LRU cache for frequently accessed images.

**Text Encoding** (Qwen3-VL only): The inference engine can extract contextualized embeddings from Qwen's language model (`src/qwen_vlm_inference.py:158-205`) for both description and embedding methods.

**Image Encoding** (Qwen3-VL only): The embedding method extracts visual features from Qwen3-VL's vision encoder (`src/qwen_vlm_inference.py:207-244`) for direct similarity computation.

## Common Commands

### Running Inference

```bash
# SigLIP2 (default, fast embedding-based)
python main.py --model-type siglip2 --language en

# SigLIP2 with different model variants
python main.py --model-type siglip2 --siglip2-model siglip2-base-patch16-224 --language en  # Fastest
python main.py --model-type siglip2 --siglip2-model siglip2-so400m-patch14-384 --language en  # Most popular
python main.py --model-type siglip2 --siglip2-model siglip2-giant-opt-patch16-384 --language en  # Best quality

# Qwen3-VL (4 distinct methods)

# Method 1: matching - Pure baseline (NO definitions, target_word + context only)
python main.py --model-type vlm --method matching --language en

# Method 2: matching_cot - Pure baseline + CoT (NO definitions)
python main.py --model-type vlm --method matching_cot --language en

# Method 3: description - Definition-based (REQUIRES definitions from Qwen3 text models)
python main.py --model-type vlm --method description --language en

# Method 4: embedding - Direct cosine similarity (NO prompts, NO definitions)
python main.py --model-type vlm --method embedding --language en

# Use smaller VLM model (for limited VRAM)
python main.py --model-type vlm --method matching --vlm-model qwen3-vl-4b --language en

# Increase batch size (more test instances in parallel)
python main.py --model-type vlm --method matching --vlm-batch-size 5 --language en

# Customize definition model (ONLY for description method)
python main.py --model-type vlm --method description --definition-model qwen3-14b --language en  # Best quality (48GB+ VRAM)
python main.py --model-type vlm --method description --definition-model qwen3-4b --language en  # Fastest, lowest VRAM (16GB+)

# Regenerate definitions (ONLY relevant for description method)
python main.py --model-type vlm --method description --regenerate-definitions --language en

# Other languages
python main.py --model-type siglip2 --language fa  # Farsi
python main.py --model-type siglip2 --language it  # Italian
```

### Evaluation Only

```bash
python eval/vwsd_ranking_metric.py \
  -p results/predictions/qwen3-vl-8b_matching_single_img_batch1 \
  -d data/test_data \
  -l en fa it \
  -o results/rank_metrics.jsonl
```

### Testing Components

```bash
# Test SigLIP2 loader (has test mode)
python -c "from src.siglip2_loader import load_model; print('SigLIP2 loader OK')"

# Test Qwen3-VL loader (has test mode)
python -c "from src.qwen_vlm_loader import load_model; print('Qwen3-VL loader OK')"

# Test Qwen3 text model loader (has test mode)
python -c "from src.qwen3_loader import load_qwen3_model; print('Qwen3 loader OK')"

# Test definition generator with cache (standalone CLI)
python src/qwen3_inference.py --language en --model qwen3-8b --test-mode

# Test data loader
python src/data_loader.py
```

## Environment Setup

```bash
# Create environment
conda create -n vwsd python=3.10 -y && conda activate vwsd

# Install PyTorch with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention 2 (optional but recommended)
pip install -U flash-attn --no-build-isolation
```

## Data Structure

```
data/
├── test_data/
│   ├── en.test.data.v1.1.txt    # Tab-separated: target_word, context, img1, ..., img10
│   └── en.test.gold.v1.1.txt    # One gold image per line
└── test_images/
    └── test_images_resized/      # Actual image files

results/
└── sense_definitions/            # Definition cache (auto-generated for VLM)
    ├── en_qwen3-14b.json        # Cached definitions for English (sorted, human-readable JSON)
    ├── fa_qwen3-14b.json        # Cached definitions for Farsi
    └── it_qwen3-14b.json        # Cached definitions for Italian
```

**Cache format** (sorted hashmap, directly human-readable):
```json
{
  "bank_a1b2c3": {
    "target_word": "bank",
    "full_phrase": "river bank",
    "definition": "Sloped land beside a river",
    "model": "qwen3-14b",
    "timestamp": "2025-11-15T..."
  }
}
```

## Important Implementation Details

### Prompt Templates

Located in `src/qwen_vlm_inference.py:25-81`. Four methods available:

**1. matching** (Pure baseline, max_tokens=50):
- Uses only target_word + context (NO definitions)
- Direct 0-10 rating prompt
- Fast, baseline comparison

**2. matching_cot** (Pure baseline + CoT, max_tokens=300):
- Uses only target_word + context (NO definitions)
- Chain-of-thought reasoning then 0-10 rating
- More interpretable baseline

**3. description** (Definition-based, max_tokens=50):
- **REQUIRES definitions** (auto-generated from Qwen3 text models)
- Provides sense definition as "visual clue"
- VLM rates 0-10 using the definition as a hint
- Tests impact of definitions on VLM performance

**4. embedding** (No prompts):
- Direct cosine similarity between text/image encoders
- No text generation, fastest method
- Uses same query format as SigLIP2: `"this is a photo of a {target_word}. {full_phrase}"`

**Processing strategy**:
- All generative methods (matching, matching_cot, description) process one image at a time to avoid attention dilution
- matching/matching_cot: Never use definitions (pure baseline)
- description: Always requires definitions (errors if not available)
- embedding: Direct similarity computation (no prompts or definitions)

### Output Format

Predictions are saved as tab-separated ranked lists:
```
img3.jpg[TAB]img7.jpg[TAB]img1.jpg[TAB]...
```

Detailed predictions include target word and context:
```
target_word[TAB]full_phrase[TAB]top1_image.jpg
```

### VLM Model Loading Parameters

- `model_name`: One of `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` (default: qwen3-vl-8b)
- `quantization`: `"bfloat16"` (full precision, default) or `"4bit"` (AWQ-INT4 quantization)
- `device`: `cuda` or `cpu`

The processor's tokenizer is always set to `padding_side='left'` for Qwen3-VL batch processing (`src/qwen_vlm_loader.py:140`).

### Memory Management

- GPU memory stats printed after VLM model loading (`models/vlm_model_loader.py:162-165`)
- Definition generator automatically cleaned up via context manager to free GPU before VLM loading
- Explicit cache clearing after processing each instance batch (`src/inference.py:263-265`)
- Context manager support for automatic VLM cleanup (`models/vlm_model_loader.py:209-217`)

### Inference Method Comparison

**Matching** (Pure baseline):
- Uses only target_word + context (NO definitions)
- Fast (only generates a rating number)
- Direct 0-10 scoring without explanation
- Max tokens: 50
- Best for: Baseline comparisons, testing inherent VLM WSD ability

**Matching CoT** (Pure baseline + reasoning):
- Uses only target_word + context (NO definitions)
- Moderate speed (generates reasoning + number)
- Model explicitly reasons about word sense before scoring
- Max tokens: 300
- Parses rating from "Rating: X" format or falls back to first number
- Best for: Interpretable baseline, understanding VLM reasoning

**Description** (Definition-based):
- **REQUIRES definitions** from Qwen3 text models
- Fast (only generates a rating number, like matching)
- Provides sense definition as "visual clue" in prompt
- VLM rates 0-10 using the definition hint
- Max tokens: 50
- Best for: Testing impact of sense definitions on VLM performance

**Embedding** (Direct retrieval):
- Fastest (no text generation)
- Direct cosine similarity between text and image encoders
- No prompting required, no definitions used
- Pure retrieval-based approach
- Uses same text query format as SigLIP2: `"this is a photo of a {target_word}. {full_phrase}"`
- Uses `encode_text()` and `encode_image()` methods (`src/qwen_vlm_inference.py:160-264`)
- Best for: Fast retrieval, comparison with SigLIP2

## Performance Baselines

- **CLIP Baseline**: ~66.8% Hit@1
- **Target**: 50-70% Hit@1
- **SOTA (SemEval-2023)**: 72.56% Hit@1

## Troubleshooting

**CUDA OOM during VLM inference**:
- **For 32GB VRAM**: Use `--definition-model qwen3-8b` (default) instead of qwen3-14b
- **For 16GB VRAM**: Use `--quantization 4bit` to enable 4-bit quantization for both definition and VLM models
- Use smaller VLM: `--vlm-model qwen3-vl-4b` (8GB) instead of qwen3-vl-8b (16GB)
- Reduce batch size: `--vlm-batch-size 1` (default)

**Example commands for limited VRAM**:
```bash
# 32GB VRAM (recommended)
python main.py --model-type vlm --definition-model qwen3-8b --language en

# 16GB VRAM with quantization
python main.py --model-type vlm --definition-model qwen3-8b --quantization 4bit --language en

# 8GB VRAM (minimum)
python main.py --model-type vlm --vlm-model qwen3-vl-4b --definition-model qwen3-4b --quantization 4bit --language en
```

**Model download issues**: Models auto-download from HuggingFace (2-32GB). Set cache with `export HF_HOME=/path/to/cache`.

**Flash Attention failures**: Optional dependency - model works without it but may be slower.
