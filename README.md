<h1 align="center">Essentials-for-NLP</h1>

<p align="center" style="font-size: 18px;">
  <i>Visual Word Sense Disambiguation with Multimodal Vision-Language Models</i>
</p>

<h4 align="center">

[![contributors](https://img.shields.io/github/contributors-anon/JaveyBae/Essentials-for-NLP?color=yellow&style=flat-square)](https://github.com/JaveyBae/Essentials-for-NLP/graphs/contributors)
[![license](https://img.shields.io/badge/License-Academic%20Research-blue.svg?style=flat-square)](https://github.com/JaveyBae/Essentials-for-NLP)
[![python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)

</h4>

## Introduction

This project implements **Visual Word Sense Disambiguation (VWSD)** using multimodal vision-language models. Given an ambiguous word in context (e.g., "bank" in "river bank"), the system ranks 10 candidate images by how well they match the intended word sense.

**Approaches implemented:**

| Category | Method | Description |
|----------|--------|-------------|
| **Discriminative** | SigLIP2 | Fast embedding-based ranking with 10+ model variants |
| | CLIP (ViT-B/32, L/14, H/14, G/14) | Baseline dual-encoder models |
| | VLM Embedding | Qwen3-VL encoder similarity (non-generative) |
| **Generative** | Direct Matching | VLM rates images 0-10 |
| | Chain-of-Thought (CoT) | Step-by-step reasoning + rating |
| | Description | Definition-enriched matching |
| | Caption → SBERT | VLM caption + Sentence-BERT similarity |
| **Hybrid** | Cascade Reranking | SigLIP2 retrieval + VLM reranking |
| **Augmentation** | Text Augmentation | Gemini-generated text variants |
| | Text-to-Image | Imagen API for synthetic query images |
| **Fine-tuning** | CLIP LoRA | LoRA on ViT-L/14 and ViT-G/14 |
| | SigLIP2 LoRA | LoRA with VLM text augmentation |

## Author Contributions

### Rui Zhou
Designed and implemented the complete experimental infrastructure:

**Core Pipelines:**
- **SigLIP2 pipeline**: Model loader with support for 10+ variants, embedding-based inference engine with three query strategies (default, with-definition, definition-only), memory-optimized batch processing
- **Qwen3-VL pipeline**: VLM loader with quantization support (bfloat16/AWQ-INT4), five distinct inference methods—matching, matching with Chain-of-Thought, definition-based description, direct embedding extraction, and caption-to-text similarity via Sentence-BERT
- **Cascade reranking**: Two-stage pipeline combining SigLIP2 retrieval with VLM reranking, including lazy model loading, configurable top-K selection, and compute optimization (70–90% reduction)
- **Definition generation**: Qwen3 text model integration (4B/8B/14B) with persistent JSON caching, few-shot prompting for visual definitions, and batch generation support

**Fine-tuning & Augmentation:**
- **LoRA fine-tuning**: Parameter-efficient adaptation framework with SigLIP-style contrastive loss, hard negative mining, early stopping, and VLM-based text augmentation (captions, definitions, paraphrases)
- **Text augmentation pipeline**: `generate_augmentations.py` for pre-generating caption/definition/paraphrase variants using Qwen3-VL and Qwen3 text models

**Infrastructure:**
- **Evaluation**: MRR, Hit@1, and NDCG metric computation with multi-language support and cumulative result tracking
- **Data loading**: Parallel image loading with ThreadPoolExecutor and LRU caching
- **Project refactoring**: Modular architecture with separate loader/inference modules per model type

**Code contributions:** `src/siglip2_loader.py`, `src/siglip2_inference.py`, `src/qwen_vlm_loader.py`, `src/qwen_vlm_inference.py`, `src/qwen3_loader.py`, `src/qwen3_inference.py`, `src/cascade_reranker.py`, `src/data_loader.py`, `finetune/train_siglip2_lora.py`, `finetune/train_siglip2.py`, `finetune/generate_augmentations.py`, `finetune/vwsd_dataset.py`, `finetune/vwsd_augmented_dataset.py`, `eval/vwsd_ranking_metric.py`, `main.py`

---

### Jiawei Pei
Contributed to dataset construction, large-scale CLIP benchmarking, and robustness evaluation:

**Model Evaluation:**
- Systematic benchmarking of CLIP encoders (ViT-B/14, ViT-L/14, ViT-G/14) at 384px resolution, establishing comparative retrieval performance across architectures

**Dataset Construction:**
- Developed multiple augmented datasets:
  1. Paraphrased and prompt-varied version of training/test text corpora using Gemini-generated rewrites
  2. Phrase-conditioned text-to-image dataset for evaluating model robustness under generative perturbations

**Text Augmentation Experiments:**
- Controlled evaluations of CLIP models (ViT-L/14, ViT-H/14, ViT-G/14) using text-augmented datasets
- Analysis of how semantic rewriting influences retrieval consistency

**Text-to-Image Evaluation Pipeline:**
- Implemented phrase-conditioned image generation using Imagen API
- Refactored evaluation codebase to compute pairwise image similarity for quantitative comparison

**Code contributions:** `scripts/augment_clip_data.py`, `scripts/image_generate_basedonphrase.py`, `notebooks/Text_Augmentation.ipynb`

---

### Nilaksan Selliah
Conducted extensive model benchmarking and LoRA-based adaptation experiments:

**Model Evaluation:**
- Systematic testing of multiple discriminative encoders: LAION CLIP ViT-L/14, ViT-G/14, ViT-H/14, and SigLIP Large (patch-16) at 224px and 384px resolutions

**LoRA Fine-tuning:**
- Parameter-efficient adaptation of CLIP ViT-L/14 using attention-only LoRA, followed by controlled comparisons against zero-shot baselines
- Additional experiments with LoRA on ViT-G/14 to assess scalability across backbones

**Text Augmentation Studies:**
- Evaluation of CLIP ViT-L/14 and ViT-G/14 using augmented textual inputs (paraphrases, synonyms, lexical rewrites)
- Analysis of semantic variation effects on LoRA robustness

**Ablation & Result Synthesis:**
- Comparison of LoRA vs. zero-shot models under standard and augmented settings
- Model selection for final experimental pipeline

**Code contributions:** `finetune/train_clip_lora_vwsd.py`, `finetune/train_clip_lora_vwsd_for_G14.py`, `notebooks/clip_script.ipynb`, `notebooks/laion_CLIP_*.ipynb`

## Quick Start

```bash
# Setup
conda create -n vwsd python=3.10 -y && conda activate vwsd
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

### SigLIP2 Inference (Fast Embedding-Based)

```bash
# Default model (so400m, ~72% Hit@1)
python main.py --model-type siglip2 --language en

# Different model variants
python main.py --model-type siglip2 --siglip2-model siglip2-base-patch16-224 --language en      # Fastest, 4GB
python main.py --model-type siglip2 --siglip2-model siglip2-so400m-patch14-384 --language en    # Best balance
python main.py --model-type siglip2 --siglip2-model siglip2-giant-opt-patch16-384 --language en # Best quality, 16GB

# Use fine-tuned model
python main.py --model-type siglip2 --siglip2-finetuned finetune/checkpoints/<your-checkpoint>/best_model --language en

# Other languages
python main.py --model-type siglip2 --language fa  # Farsi
python main.py --model-type siglip2 --language it  # Italian
```

### Qwen3-VL Inference

```bash
# Method 1: matching - Direct 0-10 rating (baseline)
python main.py --model-type vlm --method matching --language en

# Method 2: matching_cot - Chain-of-thought reasoning + rating
python main.py --model-type vlm --method matching_cot --language en

# Method 3: description - Definition-based matching (uses Qwen3 text model)
python main.py --model-type vlm --method description --language en

# Method 4: caption - VLM generates caption, Sentence-BERT computes similarity
python main.py --model-type vlm --method caption --language en

# Method 5: embedding - Direct cosine similarity (fastest VLM method)
python main.py --model-type vlm --method embedding --language en

# Use smaller model for limited VRAM
python main.py --model-type vlm --method matching --vlm-model qwen3-vl-4b --language en

# Enable 4-bit quantization (16GB VRAM)
python main.py --model-type vlm --method matching --quantization 4bit --language en
```

### Cascade Reranking (Best Performance)

```bash
# SigLIP2 ranks all 10, VLM reranks top-3 with CoT (~77% Hit@1)
python main.py --cascade --topk 3 --reranker-method matching_cot --language en

# Top-5 reranking (more accurate, slower)
python main.py --cascade --topk 5 --reranker-method matching_cot --language en

# Simple matching reranker (faster)
python main.py --cascade --topk 3 --reranker-method matching --language en

# Custom models
python main.py --cascade --topk 3 --siglip2-model siglip2-giant-opt-patch16-384 --vlm-model qwen3-vl-4b --language en
```

### Fine-tuning SigLIP2

```bash
# Without augmentation (baseline)
python finetune/train_siglip2_lora.py \
    --lr 5e-6 \
    --lora-dropout 0.1 \
    --epochs 10 \
    --early-stopping \
    --early-stopping-patience 2 \
    --val-split 0.05

# With text augmentation (reduces overfitting)
python finetune/train_siglip2_lora.py \
    --augmentation-file results/text_augmentations/train_en_augmentations_index.json \
    --aug-prob 0.5 \
    --aug-types caption definition \
    --lr 5e-6 \
    --lora-dropout 0.1 \
    --epochs 10 \
    --early-stopping \
    --early-stopping-patience 2 \
    --val-split 0.05

# Generate augmentations first (one-time, ~4 hours)
python finetune/generate_augmentations.py --type all
```

**Exact training parameters used:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--lr` | `5e-6` | Lower LR prevents catastrophic forgetting |
| `--lora-dropout` | `0.1` | Dropout for LoRA layers (regularization) |
| `--epochs` | `10` | Maximum training epochs |
| `--early-stopping` | enabled | Stop when validation MRR stops improving |
| `--early-stopping-patience` | `2` | Epochs to wait before stopping |
| `--val-split` | `0.05` | Hold out 5% for validation |

**Augmentation-specific parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--aug-prob` | `0.5` | 50% chance to use augmented text |
| `--aug-types` | `caption definition` | Use VLM captions and sense definitions |

**Comparison: With vs Without Augmentation**

Both experiments use identical hyperparameters (lr=5e-6, dropout=0.1, early stopping with patience=2). The only difference is augmentation.

| Setting | Hit@1 | MRR | NDCG@10 | Notes |
|---------|-------|-----|---------|-------|
| Zero-shot (base) | 68.03% | 79.81 | 84.77 | No fine-tuning |
| Without Augmentation | 67.39% | 79.18 | 84.28 | **Negative transfer** (−0.64%) |
| With Augmentation | 69.98%* | 80.69* | 85.43* | **+1.95% improvement** |

*Estimated from validation split performance; test evaluation pending.

See [finetune/README.md](finetune/README.md) for detailed documentation.

### Evaluation

```bash
# Evaluate predictions
python eval/vwsd_ranking_metric.py \
    -p results/predictions/<model-name> \
    -d data/test_data \
    -l en \
    -o results/metrics/my_results.jsonl

# Evaluate multiple languages
python eval/vwsd_ranking_metric.py \
    -p results/predictions/<model-name> \
    -l en fa it \
    -o results/metrics/all_languages.jsonl
```

## Approaches

### SigLIP2 (Embedding-Based)

- Direct text-image cosine similarity
- **Performance**: ~72% Hit@1
- **VRAM**: 4-16GB depending on model variant

### Qwen3-VL (5 Inference Methods)

| Method | Description | Uses Definitions |
|--------|-------------|------------------|
| `matching` | Rate image 0-10 (baseline) | No |
| `matching_cot` | Chain-of-thought + rating | No |
| `description` | Definition-based matching | Yes |
| `embedding` | Direct cosine similarity | No |
| `caption` | VLM caption + Sentence-BERT | No |

### Cascade Reranking

- **Stage 1**: SigLIP2 ranks all 10 candidates (fast)
- **Stage 2**: VLM reranks top-K only (accurate)

## Results

Results on English VWSD test set (463 instances):

| Category | Method | Hit@1 | MRR | NDCG@10 |
|----------|--------|-------|-----|---------|
| **Hybrid** | **Cascade (SigLIP2 + VLM Rerank top-5)** | **77.32%** | **85.75** | **89.24** |
| **Fine-tuned** | CLIP ViT-L/14 LoRA | 75.59% | 84.82 | 88.58 |
| | CLIP ViT-G/14 LoRA | 75.38% | 84.78 | 88.54 |
| **Augmentation** | CLIP ViT-L/14 + Text Aug | 73.43% | 83.57 | 87.65 |
| **SigLIP2** | so400m (zero-shot) | 72.79% | 82.76 | 87.00 |
| | + def (append) | 72.57% | 82.05 | 86.42 |
| | + def (replace) | 69.11% | 79.58 | 84.53 |
| **Fine-tuned** | SigLIP2 Base + LoRA + Text Aug | 69.98%* | 80.69* | 85.43* |
| **SigLIP2** | Base (zero-shot) | 68.03% | 79.81 | 84.77 |
| **Fine-tuned** | SigLIP2 Base + LoRA (no aug) | 67.39% | 79.18 | 84.28 |
| **CLIP** | ViT-G/14 (zero-shot) | 65.66% | 78.57 | 83.84 |
| **Qwen3-VL** | + CoT | 65.44% | 77.98 | 83.36 |
| **Augmentation** | CLIP ViT-L/14 + Text-to-Image | 65.23% | 77.34 | 82.86 |
| **CLIP** | ViT-H/14 (zero-shot) | 63.28% | 77.00 | 82.66 |
| **Qwen3-VL** | + Sense Def | 63.07% | 74.89 | 80.88 |
| **CLIP** | ViT-L/14 (zero-shot) | 61.99% | 75.71 | 81.68 |
| **Qwen3-VL** | Direct Matching | 61.77% | 75.93 | 81.83 |
| **CLIP** | ViT-B/32 (zero-shot) | 61.34% | 74.66 | 80.83 |
| **SigLIP2** | + def (only) | 61.34% | 74.46 | 80.67 |
| **Qwen3-VL** | Caption → SBERT | 48.60% | 65.10 | 73.52 |
| | Embedding | 9.07% | 29.31 | 45.51 |

*Estimated from validation split performance; test evaluation pending.

**Key findings:**
- **Cascade reranking** achieves best results (77.32% Hit@1) by combining SigLIP2 speed with VLM reasoning
- **SigLIP2** outperforms CLIP variants by ~7-11 points due to sigmoid loss optimization
- **Text augmentation** with Gemini-generated rewrites boosts CLIP from 61.99% to 73.43% (+11.44 points)
- **LoRA fine-tuning** without augmentation causes negative transfer (−0.64%), but text augmentation recovers gains (+1.95%)
- **Knowledge augmentation** (sense definitions) hurts SigLIP2 performance due to distribution mismatch
- **VLM embeddings** fail catastrophically (9.07%) as generation-optimized representations are non-metric

## Project Structure

```
├── src/                              # Core inference modules
│   ├── siglip2_loader.py            # SigLIP2 model loading (10+ variants)
│   ├── siglip2_inference.py         # SigLIP2 embedding-based ranking
│   ├── qwen_vlm_loader.py           # Qwen3-VL loading (bfloat16/AWQ-INT4)
│   ├── qwen_vlm_inference.py        # Qwen3-VL 5 inference methods
│   ├── qwen3_loader.py              # Qwen3 text model loading
│   ├── qwen3_inference.py           # Definition generation with caching
│   ├── cascade_reranker.py          # Two-stage retrieval + reranking
│   └── data_loader.py               # Parallel image loading with LRU cache
├── finetune/                         # Fine-tuning modules
│   ├── train_siglip2_lora.py        # SigLIP2 LoRA with augmentation
│   ├── train_siglip2.py             # SigLIP2 full fine-tuning
│   ├── train_clip_lora_vwsd.py      # CLIP ViT-L/14 LoRA
│   ├── train_clip_lora_vwsd_for_G14.py  # CLIP ViT-G/14 LoRA
│   ├── generate_augmentations.py    # VLM text augmentation generator
│   ├── vwsd_dataset.py              # Base contrastive dataset
│   ├── vwsd_augmented_dataset.py    # Augmented dataset with caption/def/paraphrase
│   └── checkpoints/                 # Saved model checkpoints
├── eval/                             # Evaluation
│   └── vwsd_ranking_metric.py       # MRR, Hit@1, NDCG computation
├── scripts/                          # Utility scripts
│   ├── shell/                       # Shell scripts for experiments
│   ├── augment_clip_data.py         # Gemini text augmentation
│   ├── image_generate_basedonphrase.py  # Imagen text-to-image generation
│   └── example_image_similarity.py  # Image similarity utilities
├── notebooks/                        # Jupyter notebooks
│   ├── clip_script.ipynb            # CLIP baseline experiments
│   ├── Text_Augmentation.ipynb      # Text augmentation analysis
│   └── laion_CLIP_*.ipynb           # LAION-CLIP experiments
├── data/                             # Datasets
│   ├── test_data/                   # Test splits (en/fa/it)
│   ├── train_data/                  # Training data (12,869 instances)
│   └── test_images/                 # Image files
├── results/                          # Outputs
│   ├── predictions/                 # Model predictions (ranked lists)
│   ├── metrics/                     # Evaluation JSONL files
│   ├── sense_definitions/           # Cached sense definitions
│   └── text_augmentations/          # Cached text augmentations
├── report/                           # LaTeX report (ACL format)
│   └── latex/                       # Final report source
├── vwsd_utils/                       # Legacy CLIP utilities
│   ├── embedding_clip.py            # CLIP embedding extraction
│   ├── image_evaluator.py           # Image evaluation utilities
│   └── plot.py                      # Visualization helpers
└── main.py                           # Entry point for all experiments
```

## Dataset

🔗 [Google Drive - VWSD Dataset](https://drive.google.com/file/d/1KLux4KlOdoOGmoETyu-Qc-rShnnUbGWi/view?usp=sharing)

- **Languages**: English, Persian, Italian
- **Format**: Tab-separated (target_word, context, 10 candidate images)

## Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **CUDA**: 12.8
- **Python**: 3.10
- **PyTorch**: 2.9.0+

## Authors

| Name | GitHub |
|------|--------|
| Rui Zhou | [![GitHub](https://img.shields.io/github/followers/RuiZhou-cn?label=Follow&style=social)](https://github.com/RuiZhou-cn) |
| Jiawei Pei | [![GitHub](https://img.shields.io/github/followers/JaveyBae?label=Follow&style=social)](https://github.com/JaveyBae) |
| Nilaksan Sandrakumar | [![GitHub](https://img.shields.io/github/followers/nilaksan97?label=Follow&style=social)](https://github.com/nilaksan97) |

## License

This project is for academic research purposes.
