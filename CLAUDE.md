# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Visual Word Sense Disambiguation (VWSD)** system using Qwen3-VL multimodal models. The task: given an ambiguous word in context (e.g., "bank" in "I went to the bank"), rank 10 candidate images by how well they match the intended word sense.

**Core Strategy**: Single-image evaluation - each candidate image is evaluated independently with the target word and context to avoid attention dilution in the VLM.

## System Architecture

### Component Flow

1. **Data Loading** (`src/data_loader.py`): Loads test instances (word + context + 10 candidate images) with parallel image loading optimization
2. **VLM Model Loading** (`models/vlm_model_loader.py`): Loads Qwen3-VL vision-language models with caching, bfloat16/AWQ-INT4 support, and automatic Flash Attention 2
3. **Definition Generation** (`models/definition_generator.py`): Generates contextual sense definitions using Qwen3 text models (separate from VLM pipeline)
4. **Inference Engine** (`src/inference.py`): Core ranking logic using two methods:
   - `matching`: Prompt VLM to rate each image 0-10 for relevance
   - `description`: Generate image description, compute semantic similarity with context using model's text encoder
5. **Evaluation** (`eval/vwsd_ranking_metric.py`): Computes MRR and Hit@1 metrics
6. **Main Orchestration** (`main.py`): Coordinates the pipeline

### Key Design Patterns

**Batching Strategy**: The `--batch-size` parameter controls how many **test instances** are processed in parallel, NOT how many images within an instance. Within each instance, all ~10 candidate images are processed together in a single forward pass (see `src/inference.py:190-265`).

**Model Caching**: VLM models are cached globally in `models/vlm_model_loader.py:22` to avoid reloading across runs. Use `Qwen3VLModelLoader.clear_all_cache()` to clear.

**Image Loading**: Uses `ThreadPoolExecutor` for parallel loading (`src/data_loader.py:210-231`) with LRU cache for frequently accessed images.

**Text Encoding for Description Method**: When using `description` method, the inference engine extracts contextualized embeddings from Qwen's language model (`src/inference.py:79-126`) and computes cosine similarity between image descriptions and the context phrase.

## Common Commands

### Running Inference

```bash
# Basic run with default 8B model
python main.py --language en

# Use smaller model (for limited VRAM)
python main.py --vlm-model qwen3-vl-4b --language en

# Increase batch size for faster processing (more test instances in parallel)
python main.py --batch-size 5 --language en

# Try chain-of-thought matching (with reasoning before scoring)
python main.py --method matching_cot --language en

# Try description method instead of matching
python main.py --method description --language en

# Other languages
python main.py --language fa  # Farsi
python main.py --language it  # Italian
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
# Test VLM model loader
python models/vlm_model_loader.py

# Test definition generator
python models/definition_generator.py

# Test data loader
python src/data_loader.py

# Test inference engine
python src/inference.py
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
```

## Important Implementation Details

### Prompt Templates

Located in `src/inference.py:21-62`. Three methods available:
- `matching`: Direct 0-10 rating (fast, max_tokens=50)
- `matching_cot`: Chain-of-thought reasoning then 0-10 rating (more thorough, max_tokens=150)
- `description`: 8-15 word description, scored by semantic similarity (interpretable, max_tokens=100)

All methods process one image at a time to avoid attention dilution.

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

- `model_name`: One of `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b`, `qwen3-vl-32b`
- `quantization`: `None` (bfloat16 full precision) or `4bit` (AWQ-INT4 quantization)
- `device`: `cuda` or `cpu`

The processor's tokenizer is always set to `padding_side='left'` for Qwen3-VL batch processing (`models/vlm_model_loader.py:155`).

### Memory Management

- GPU memory stats printed after VLM model loading (`models/vlm_model_loader.py:162-165`)
- Definition generator automatically cleaned up via context manager to free GPU before VLM loading
- Explicit cache clearing after processing each instance batch (`src/inference.py:263-265`)
- Context manager support for automatic VLM cleanup (`models/vlm_model_loader.py:209-217`)

### Inference Method Comparison

**Matching** (baseline):
- Fastest (only generates a number)
- Direct scoring without explanation
- Max tokens: 50

**Matching CoT** (recommended to try):
- Moderate speed (generates reasoning + number)
- May improve accuracy through step-by-step analysis
- Model explicitly reasons about word sense before scoring
- Max tokens: 150
- Parses rating from "Rating: X" format or falls back to first number

**Description**:
- Slowest (generates description + computes embeddings)
- Most interpretable (can see what model "sees")
- Uses Qwen's text encoder for semantic similarity
- Max tokens: 100
- Includes target word bonus in scoring (`src/inference.py:357-359`)

## Performance Baselines

- **CLIP Baseline**: ~66.8% Hit@1
- **Target**: 50-70% Hit@1
- **SOTA (SemEval-2023)**: 72.56% Hit@1

## Troubleshooting

**CUDA OOM**: Use smaller model (`--vlm-model qwen3-vl-4b`) or 4-bit quantization (`--quantization 4bit`). Note that `batch-size` affects instance-level parallelism, not per-instance image batching.

**Model download issues**: Models auto-download from HuggingFace (2-32GB). Set cache with `export HF_HOME=/path/to/cache`.

**Flash Attention failures**: Optional dependency - model works without it but may be slower.
