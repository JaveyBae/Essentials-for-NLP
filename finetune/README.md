# SigLIP2 Fine-tuning for VWSD

Fine-tune SigLIP2 vision-language models on the Visual Word Sense Disambiguation (VWSD) task.

## Quick Start

```bash
# Train with default settings (recommended for 32GB VRAM)
python finetune/train_siglip2.py \
    --model-name google/siglip2-so400m-patch14-384 \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-5

# Run inference with fine-tuned model
python main.py \
    --model-type siglip2 \
    --siglip2-finetuned finetune/checkpoints/siglip2_vwsd_<timestamp>/best_model \
    --language en
```

## What is Fine-tuning?

**Fine-tuning** = Taking a pre-trained model and continuing to train it on your specific task.

```
Pre-trained SigLIP2 (general image-text matching)
         ↓ fine-tune on VWSD data
Task-specific SigLIP2 (disambiguating word senses via images)
```

## Training Approaches

### Full Fine-tuning (Recommended for 32GB VRAM + 12K examples)

Updates all model weights. Best performance but needs more data and compute.

```bash
python finetune/train_siglip2.py \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-5 \
    --num-hard-negatives 4
```

### LoRA (For limited VRAM or data)

Parameter-efficient training. Coming soon - use PEFT library to add.

## Contrastive Learning

The training uses **SigLIP-style contrastive loss**:

1. **Positive pairs**: (text query, correct sense image)
2. **Hard negatives**: Other candidate images for the same word (e.g., "bank" river vs "bank" money)
3. **In-batch negatives**: Other samples' images in the same batch

```
Loss = -log(sigmoid(pos_sim)) - sum(log(sigmoid(-neg_sim)))
```

Hard negatives are critical because they teach the model to distinguish subtle sense differences.

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | google/siglip2-so400m-patch14-384 | Base model to fine-tune |
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 16 | Batch size per GPU |
| `--lr` | 1e-5 | Learning rate |
| `--num-hard-negatives` | 4 | Hard negatives per positive (max 9) |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation |
| `--warmup-ratio` | 0.1 | Warmup ratio |
| `--val-split` | 0.05 | Validation split ratio |
| `--val-every` | 1 | Validate every N epochs |

## Data Structure

Uses the SemEval-2023 VWSD training data:

```
data/semeval-2023-task-1-V-WSD-train-v1/
├── train_v1/
│   ├── train.data.v1.txt      # 12,869 instances
│   ├── train.gold.v1.txt      # Gold labels
│   └── train_images_v1/       # 12,999 images
└── trial_v1/
    ├── trial.data.v1.txt      # 16 trial instances
    └── trial_images_v1/
```

## Output Structure

```
finetune/checkpoints/siglip2_vwsd_<timestamp>/
├── best_model/           # Best checkpoint (highest MRR)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── training_info.json
├── checkpoint_epoch5/    # Intermediate checkpoint
├── checkpoint_epoch10/
└── final_model/          # Final epoch model
```

## Expected Results

| Model | Zero-shot Hit@1 | Fine-tuned Hit@1 (expected) |
|-------|-----------------|----------------------------|
| siglip2-so400m-patch14-384 | ~60% | ~65-75% |

## Memory Requirements

| Model | VRAM (Training) | VRAM (Inference) |
|-------|-----------------|------------------|
| siglip2-base-patch16-224 | ~8GB | ~2GB |
| siglip2-so400m-patch14-384 | ~20GB | ~4GB |
| siglip2-giant-opt-patch16-384 | ~32GB | ~8GB |

## Tips

1. **Start with fewer hard negatives** (4) then increase to 9 if not overfitting
2. **Use gradient accumulation** for larger effective batch size
3. **Monitor validation MRR** - it should improve steadily
4. **Stop early** if validation metrics plateau or degrade

## Files

- `vwsd_dataset.py`: Dataset classes for contrastive training
- `train_siglip2.py`: Main training script with full fine-tuning
