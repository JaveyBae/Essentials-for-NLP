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

**Three approaches implemented:**
1. **SigLIP2**: Fast embedding-based ranking (~72% Hit@1)
2. **Qwen3-VL**: Advanced VLM with 5 inference methods
3. **Cascade Reranking**: Two-stage pipeline combining speed and accuracy

## Contributions

| Contributor | Contributions |
|-------------|---------------|
| **Rui Zhou** | Core architecture design • SigLIP2 & Qwen3-VL integration • Cascade reranking pipeline • SigLIP2 LoRA fine-tuning with early stopping • VLM-based text augmentation (caption/definition/paraphrase) • 5 inference methods (matching, matching_cot, description, embedding, caption) • Sense definition enrichment • Evaluation metrics • Project refactoring & documentation |
| **Jiawei Pei** | CLIP baseline implementation • Text augmentation with Gemini API • Data preprocessing • Project setup and structure |
| **Nilaksan Sandrakumar** | OpenCLIP experiments (ViT-H/14, ViT-G/14) • SigLIP baseline evaluation • CLIP LoRA fine-tuning experiments |

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

| Setting | Hit@1 | MRR | Notes |
|---------|-------|-----|-------|
| Without Augmentation | TBD | TBD | Baseline LoRA fine-tuning |
| With Augmentation | TBD | TBD | +caption/definition augmentation |

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

| Method | Hit@1 | MRR | NDCG@10 |
|--------|-------|-----|---------|
| **Cascade (SigLIP2 + VLM Rerank)** | **77.32%** | 85.75 | 89.24 |
| CLIP LoRA Fine-tuned | 75.59% | 84.82 | -- |
| SigLIP2 so400m | 72.79% | 82.76 | 87.00 |
| SigLIP2 Base | 68.03% | 79.81 | 84.77 |
| Qwen3-VL + CoT | 65.44% | 77.98 | 83.36 |
| Qwen3-VL + Sense Def | 63.07% | 74.89 | 80.88 |
| Qwen3-VL Direct Matching | 61.77% | 75.93 | 81.83 |
| CLIP Baseline | 61.34% | 74.66 | 80.83 |
| Caption → SBERT | 48.60% | 65.10 | 73.52 |
| Qwen3-VL Embedding | 9.07% | 29.31 | 45.51 |

## Project Structure

```
├── src/                    # Core inference modules
│   ├── siglip2_*.py       # SigLIP2 loader/inference
│   ├── qwen_vlm_*.py      # Qwen3-VL loader/inference
│   ├── qwen3_*.py         # Qwen3 text model (definitions)
│   ├── cascade_reranker.py
│   └── data_loader.py     # Data loading utilities
├── finetune/              # LoRA fine-tuning
│   ├── train_siglip2_lora.py
│   ├── generate_augmentations.py
│   └── vwsd_*_dataset.py
├── eval/                   # Evaluation metrics
├── scripts/               # Utility scripts
│   ├── shell/             # Shell scripts
│   └── *.py               # Python utilities
├── notebooks/             # Jupyter notebooks
├── data/                  # Datasets
│   ├── test_data/         # Test splits (en/fa/it)
│   ├── train_data/        # Training data
│   └── test_images/       # Image files
├── results/               # Outputs
│   ├── predictions/       # Model predictions
│   ├── metrics/           # Evaluation results
│   └── text_augmentations/
├── report/                # LaTeX report
└── main.py               # Entry point
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
