<h1 align="center">Essentials-for-NLP</h1>

<p align="center" style="font-size: 18px;">
  <i>Visual Word Sense Disambiguation with Multimodal Vision-Language Models</i>
</p>

<h4 align="center">

[![contributors](https://img.shields.io/github/contributors-anon/JaveyBae/Essentials-for-NLP?color=yellow&style=flat-square)](https://github.com/JaveyBae/Essentials-for-NLP/graphs/contributors)
[![license](https://img.shields.io/badge/License-Academic%20Research-blue.svg?style=flat-square)](https://github.com/JaveyBae/Essentials-for-NLP)
[![python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)

</h4>

## Introduction

This project implements **Visual Word Sense Disambiguation (VWSD)** using multimodal vision-language models. Given an ambiguous word in context (e.g., "bank" in "river bank"), the system ranks 10 candidate images by how well they match the intended word sense.

**Three approaches implemented:**
1. **SigLIP2**: Fast embedding-based ranking (~72% Hit@1)
2. **Qwen3-VL**: Advanced VLM with 7 inference methods
3. **Cascade Reranking**: Two-stage pipeline combining speed and accuracy

## Contributions

| Contributor | Contributions |
|-------------|---------------|
| **Rui Zhou** | Core architecture design, SigLIP2/Qwen3-VL integration, cascade reranking pipeline, sense definition enrichment, fine-tuning module, inference engine (7 methods: matching, matching_cot, description, embedding, caption, text_augmentation, image_generation) |
| **Jiawei Pei** | CLIP baseline implementation, text augmentation with Gemini API, data preprocessing, project setup and structure |
| **Nilaksan Sandrakumar** | OpenCLIP experiments (ViT-H/14, ViT-G/14), SigLIP baseline evaluation, CLIP LoRA fine-tuning experiments |

## Quick Start

```bash
# Setup
conda create -n vwsd python=3.10 -y && conda activate vwsd
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Run inference
python main.py --model-type siglip2 --language en           # Fast baseline
python main.py --model-type vlm --method matching --language en  # VLM
python main.py --cascade --topk 3 --language en             # Cascade
```

## Approaches

### SigLIP2 (Embedding-Based)
- Direct text-image cosine similarity
- **Performance**: ~72% Hit@1
- **VRAM**: 4-16GB depending on model variant

### Qwen3-VL (7 Inference Methods)

| Method | Description | Uses Definitions |
|--------|-------------|------------------|
| `matching` | Rate image 0-10 (baseline) | No |
| `matching_cot` | Chain-of-thought + rating | No |
| `description` | Definition-based matching | Yes |
| `embedding` | Direct cosine similarity | No |
| `caption` | VLM caption + Sentence-BERT | No |
| `text_augmentation` | Gemini text enrichment | No |
| `image_generation` | Imagen synthetic images | No |

### Cascade Reranking
- **Stage 1**: SigLIP2 ranks all 10 candidates (fast)
- **Stage 2**: VLM reranks top-K only (accurate)

## Results

| Method | Hit@1 | MRR@10 |
|--------|-------|--------|
| **SigLIP2-SO400M** | **72.79%** | 0.8276 |
| SigLIP2-Giant | 72.35% | 0.8271 |
| Qwen3-VL matching_cot | 65.44% | 0.7798 |
| CLIP Baseline | 66.8% | - |

## Project Structure

```
├── src/                    # Core modules
│   ├── siglip2_*.py       # SigLIP2 loader/inference
│   ├── qwen_vlm_*.py      # Qwen3-VL loader/inference
│   └── cascade_reranker.py
├── eval/                   # Evaluation metrics
├── finetune/              # LoRA fine-tuning
├── data/                   # Test data and images
├── report/                 # LaTeX report
└── main.py                # Entry point
```

## Dataset

🔗 [Google Drive - VWSD Dataset](https://drive.google.com/file/d/1KLux4KlOdoOGmoETyu-Qc-rShnnUbGWi/view?usp=sharing)

- **Languages**: English, Persian, Italian
- **Format**: Tab-separated (target_word, context, 10 candidate images)

## Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **CUDA**: 12.8
- **Python**: 3.10
- **PyTorch**: 2.8.0+

## Authors

| Name | GitHub |
|------|--------|
| Rui Zhou | [![GitHub](https://img.shields.io/github/followers/RuiZhou-cn?label=Follow&style=social)](https://github.com/RuiZhou-cn) |
| Jiawei Pei | [![GitHub](https://img.shields.io/github/followers/JaveyBae?label=Follow&style=social)](https://github.com/JaveyBae) |
| Nilaksan Sandrakumar | [![GitHub](https://img.shields.io/github/followers/nilaksan97?label=Follow&style=social)](https://github.com/nilaksan97) |

## License

This project is for academic research purposes.
