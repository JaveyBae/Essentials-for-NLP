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
│   └── test_data/                          # Test data files (.txt format)
├── models/
│   └── model_loader.py                     # Qwen3-VL model loading utilities
├── src/
│   ├── data_loader.py                      # VWSD data loading
│   └── inference.py                        # Inference engine
├── eval/
│   └── vwsd_ranking_metric.py              # Evaluation metrics
├── results/
│   └── predictions/                        # Output predictions
├── main.py                                 # Main execution script
└── requirements.txt                        # Python dependencies
```

## Usage

### Basic Command

```bash
# Run evaluation with default settings (8B model, matching method, batch size 10)
python main.py --language en

# Use smaller model (for lower VRAM)
python main.py --model qwen3-vl-4b --language en
```

### Advanced Options

```bash
# Try description method instead of matching
python main.py --method description --language en

# Adjust batch size (number of test instances to process in parallel)
# Each instance's ~10 candidate images are evaluated together in one batch
python main.py --batch-size 5 --language en

# Other languages
python main.py --language fa  # Farsi
python main.py --language it  # Italian
```

## Models & Inference Strategy

**Available Models** (auto-download from HuggingFace):
- `qwen3-vl-2b`: ~4GB VRAM (~2GB with 8-bit) - Fastest
- `qwen3-vl-4b`: ~8GB VRAM (~4GB with 8-bit) - Fast
- `qwen3-vl-8b`: ~16GB VRAM (~8GB with 8-bit) - **Recommended**
- `qwen3-vl-32b`: ~64GB VRAM (~16GB with 8-bit) - Best quality

**Inference Strategy**:
- **Batch evaluation**: All ~10 candidate images for an instance are processed together in a single forward pass for efficiency
- **Instance batching**: Multiple test instances can be processed in parallel using `--batch-size`
- **Example**: `batch_size=5` means 5 test instances are processed simultaneously, and within each instance, all ~10 candidate images are evaluated together in one batch
- **Memory efficiency**: Images are batched per instance to maximize GPU utilization while keeping memory manageable

## Inference Methods

Two strategies for ranking images:

1. **matching** (default, recommended): Numeric rating (0-10)
   - Prompts the model: "Rate how well this image matches the SPECIFIC meaning of [word] in [context]"
   - Each image receives a score from 0 (unrelated) to 10 (perfect match)
   - All images for an instance are processed together in one batch
   - Images are ranked by their scores (highest to lowest)

2. **description**: Semantic similarity-based matching
   - Generates a description for each image (8-15 words)
   - Computes cosine similarity between description embeddings and context phrase embeddings using Qwen's text encoder
   - Optional bonus if target word appears in description
   - More interpretable results but may be less accurate

## Key Command-Line Options

```bash
python main.py [OPTIONS]

# Model (quantization not currently used in main.py)
--model {qwen3-vl-2b,4b,8b,32b}     [default: qwen3-vl-8b]
--dtype {bfloat16,float16,auto}     [default: bfloat16 (hardcoded in main.py)]

# Inference
--method {matching,description}          [default: matching]
--batch-size N                           [default: 10, number of test instances to process in parallel]
--language {en,fa,it}                    [default: en]

# Data & Evaluation
--data-dir DIR                      [default: data]
--output-dir DIR                    [default: results/predictions]
```

## Performance Benchmarks

- **CLIP Baseline**: ~66.8% Hit@1 (OpenAI CLIP ViT-L/14)
- **Target**: 50-70% Hit@1 with VLM
- **SOTA (SemEval-2023)**: 72.56% Hit@1

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
python main.py --model qwen3-vl-4b --language en

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
