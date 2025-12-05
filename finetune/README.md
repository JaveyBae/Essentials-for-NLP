# SigLIP2 Fine-tuning for VWSD

Fine-tune SigLIP2 vision-language models on the Visual Word Sense Disambiguation (VWSD) task using LoRA (Low-Rank Adaptation).

## Quick Start

```bash
# Basic LoRA fine-tuning with early stopping (recommended)
python finetune/train_siglip2_lora.py \
    --lr 5e-6 \
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
    --epochs 10 \
    --early-stopping \
    --early-stopping-patience 2 \
    --val-split 0.05

# Run inference with fine-tuned model
python main.py \
    --model-type siglip2 \
    --siglip2-finetuned finetune/checkpoints/siglip2_lora_r16_alpha32_<timestamp>/best_model \
    --language en
```

## What is LoRA Fine-tuning?

**LoRA** = Low-Rank Adaptation. Instead of updating all model weights, we add small trainable adapter matrices.

```
Pre-trained SigLIP2 (378M params, frozen)
         + LoRA adapters (~2M trainable params)
         ↓ fine-tune on VWSD data
Task-specific model (adapts to word sense disambiguation)
```

**Benefits**:
- Memory efficient (~2M vs 378M trainable params)
- Faster training
- Less overfitting risk
- Easy to switch between original and fine-tuned

## Preventing Overfitting

The VWSD training set (12,869 instances) is small, making overfitting a major concern. Use these strategies:

### 1. Early Stopping (Recommended)

Automatically stop when validation performance stops improving:

```bash
python finetune/train_siglip2_lora.py \
    --early-stopping \
    --early-stopping-patience 2 \
    --early-stopping-min-delta 0.001
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--early-stopping` | off | Enable early stopping |
| `--early-stopping-patience` | 2 | Stop after N epochs without improvement |
| `--early-stopping-min-delta` | 0.001 | Minimum MRR improvement to count as progress |

### 2. Lower Learning Rate

The pretrained model already knows a lot. A high learning rate causes "catastrophic forgetting":

| Learning Rate | Effect |
|---------------|--------|
| `2e-5` | Too high - model forgets pretrained knowledge |
| `5e-6` | Good balance (recommended) |
| `1e-6` | Conservative, slower but safer |

### 3. Text Augmentation

Increase effective dataset diversity with VLM-generated text:

```bash
# Step 1: Generate augmentations (one-time, ~4 hours)
python finetune/generate_augmentations.py --type all

# Step 2: Train with augmentation
python finetune/train_siglip2_lora.py \
    --augmentation-file results/text_augmentations/train_en_augmentations_index.json \
    --aug-prob 0.5 \
    --aug-types caption definition
```

**Augmentation types**:
| Type | Model | Description |
|------|-------|-------------|
| `caption` | Qwen3-VL-8B | VLM describes the gold image |
| `definition` | Qwen3-8B | Visual definition for word sense |
| `paraphrase` | Qwen3-8B | Paraphrased context phrases |

### 4. LoRA Regularization

```bash
python finetune/train_siglip2_lora.py \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.15
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank (lower = fewer params, less overfitting) |
| `--lora-alpha` | 32 | LoRA scaling (typically 2*r) |
| `--lora-dropout` | 0.05 | LoRA dropout (higher = more regularization) |

## Training Arguments

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | google/siglip2-base-patch16-512 | Base model to fine-tune |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--include-mlp` | off | Apply LoRA to MLP layers too |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 8 | Batch size per GPU |
| `--gradient-accumulation-steps` | 4 | Effective batch = batch-size * accumulation |
| `--lr` | 2e-5 | Learning rate (**use 5e-6 to prevent forgetting**) |
| `--warmup-ratio` | 0.1 | Warmup steps ratio |
| `--num-hard-negatives` | 5 | Hard negatives per positive (max 9) |

### Validation & Early Stopping

| Argument | Default | Description |
|----------|---------|-------------|
| `--val-split` | 0.0 | Validation split from training data |
| `--val-every` | 1 | Validate every N epochs |
| `--val-samples` | 500 | Max validation samples |
| `--early-stopping` | off | Enable early stopping |
| `--early-stopping-patience` | 2 | Epochs without improvement before stopping |
| `--early-stopping-min-delta` | 0.001 | Minimum improvement threshold |

### Text Augmentation

| Argument | Default | Description |
|----------|---------|-------------|
| `--augmentation-file` | None | Path to augmentation JSON |
| `--aug-prob` | 0.5 | Probability of using augmented text |
| `--aug-types` | caption definition | Augmentation types to use |

## Example Configurations

### Conservative (Prevents overfitting)

```bash
python finetune/train_siglip2_lora.py \
    --lr 1e-6 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.15 \
    --epochs 10 \
    --early-stopping \
    --early-stopping-patience 2 \
    --val-split 0.05
```

### Balanced (Recommended)

```bash
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
```

### Aggressive (More capacity, risk of overfitting)

```bash
python finetune/train_siglip2_lora.py \
    --augmentation-file results/text_augmentations/train_en_augmentations_index.json \
    --aug-prob 0.7 \
    --aug-types caption definition paraphrase \
    --lr 2e-5 \
    --lora-r 32 \
    --lora-alpha 64 \
    --epochs 5 \
    --early-stopping \
    --early-stopping-patience 1
```

## Output Structure

```
finetune/checkpoints/siglip2_lora_r16_alpha32_<timestamp>/
├── best_model/              # Best checkpoint (highest MRR)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_info.json
├── final_model/             # Final epoch model
├── checkpoint_epoch5/       # Intermediate checkpoints
├── training_metrics.json    # All metrics (for plotting)
├── loss_curves.png          # Train/val loss visualization
├── metrics_curves.png       # MRR/Hit@1 over epochs
└── step_loss.png            # Per-step loss (detailed)
```

## Interpreting Results

### Loss Curves

Check `loss_curves.png` for overfitting:

```
Good:  Train ↓, Val ↓ (both decreasing)
Bad:   Train ↓, Val ↑ (overfitting - stop earlier!)
```

### Training Metrics JSON

```json
{
  "epochs": [1, 2, 3],
  "train_losses": [1.28, 1.15, 1.07],
  "val_losses": [1.30, 1.20, 1.25],  // Increasing = overfitting
  "val_mrr": [0.81, 0.75, 0.70],     // Decreasing = forgetting
  "val_hit1": [0.70, 0.65, 0.60]
}
```

## Troubleshooting

### Model forgets pretrained knowledge (MRR drops after epoch 1)

```bash
# Lower learning rate
--lr 1e-6

# Add early stopping
--early-stopping --early-stopping-patience 1
```

### Out of memory

```bash
# Reduce batch size
--batch-size 4

# Reduce hard negatives
--num-hard-negatives 3

# Use smaller base model
--model-name google/siglip2-base-patch16-224
```

### Validation MRR not improving

```bash
# Try text augmentation
--augmentation-file results/text_augmentations/train_en_augmentations_index.json

# Increase LoRA capacity
--lora-r 32 --lora-alpha 64
```

## Files

| File | Description |
|------|-------------|
| `train_siglip2_lora.py` | Main LoRA training script |
| `train_siglip2.py` | Full fine-tuning (deprecated) |
| `generate_augmentations.py` | Generate text augmentations |
| `vwsd_dataset.py` | Base contrastive dataset |
| `vwsd_augmented_dataset.py` | Dataset with augmentation support |

## Expected Results

| Configuration | Expected Hit@1 |
|---------------|----------------|
| Zero-shot SigLIP2-base | ~68% |
| Zero-shot SigLIP2-so400m | ~72% |
| LoRA fine-tuned (well-tuned) | ~75-78% |
| LoRA fine-tuned (overfitted) | < 60% |

The key is preventing overfitting - a well-tuned model should improve 3-6% over the baseline.
