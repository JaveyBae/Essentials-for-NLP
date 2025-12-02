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