"""
SigLIP2 LoRA Fine-tuning for Visual Word Sense Disambiguation

Low-Rank Adaptation (LoRA) fine-tuning - trains only a small number of adapter
parameters while keeping the base model frozen. Much more memory efficient
than full fine-tuning.

Usage:
    # Basic training (no augmentation)
    python finetune/train_siglip2_lora.py --epochs 10 --batch-size 8 --lr 2e-5

    # Training with text augmentation (recommended to prevent overfitting)
    python finetune/train_siglip2_lora.py \
        --augmentation-file results/text_augmentations/train_en_augmentations_index.json \
        --aug-prob 0.5 \
        --aug-types caption definition paraphrase \
        --val-split 0.05 \
        --epochs 10

    # Compare augmented vs non-augmented (run both and compare loss curves)
    # Without augmentation:
    python finetune/train_siglip2_lora.py --val-split 0.05 --epochs 10
    # With augmentation:
    python finetune/train_siglip2_lora.py \
        --augmentation-file results/text_augmentations/train_en_augmentations_index.json \
        --aug-prob 0.5 --val-split 0.05 --epochs 10

Key features:
- LoRA adapters on attention layers (q_proj, v_proj, k_proj, out_proj)
- Optional LoRA on MLP layers (fc1, fc2)
- Frozen base model (~878M params frozen, ~2-8M trainable)
- Contrastive loss with in-batch negatives + hard negatives
- Mixed precision training (bfloat16)
- Validation with MRR/Hit@1 metrics
- Checkpoint saving with best model tracking
- Training/validation loss curves saved as PNG for overfitting detection
- Comprehensive metrics logging to JSON for analysis
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from PIL import ImageFile, Image

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Increase limit for large images (some training images are very large)
Image.MAX_IMAGE_PIXELS = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from finetune.vwsd_dataset import VWSDContrastiveDataset
from finetune.vwsd_augmented_dataset import VWSDAugmentedDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Track training metrics across epochs for visualization and analysis."""
    epochs: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_mrr: List[float] = field(default_factory=list)
    val_hit1: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)

    # Per-step metrics (more granular)
    step_losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    def add_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        mrr: Optional[float] = None,
        hit1: Optional[float] = None,
        lr: Optional[float] = None,
        temp: Optional[float] = None
    ):
        """Add metrics for a completed epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if mrr is not None:
            self.val_mrr.append(mrr)
        if hit1 is not None:
            self.val_hit1.append(hit1)
        if lr is not None:
            self.learning_rates.append(lr)
        if temp is not None:
            self.temperatures.append(temp)

    def add_step_loss(self, step: int, loss: float):
        """Add per-step loss for more detailed tracking."""
        self.steps.append(step)
        self.step_losses.append(loss)

    def save(self, path: str):
        """Save metrics to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved training metrics to {path}")

    @classmethod
    def load(cls, path: str) -> 'TrainingMetrics':
        """Load metrics from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def plot_training_curves(metrics: TrainingMetrics, output_dir: Path, title_suffix: str = ""):
    """
    Plot and save training curves for loss and metrics.

    Creates:
    - loss_curves.png: Train vs validation loss
    - metrics_curves.png: MRR and Hit@1 over epochs
    - step_loss.png: Per-step training loss (if available)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot generation.")
        logger.warning("Install with: pip install matplotlib")
        return

    # Style settings - use built-in style that works across versions
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')  # Fallback to ggplot which is always available

    # 1. Loss curves (train vs validation)
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = metrics.epochs
    ax.plot(epochs, metrics.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)

    if metrics.val_losses:
        ax.plot(epochs, metrics.val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)

        # Highlight overfitting region (where val loss increases but train loss decreases)
        for i in range(1, len(metrics.val_losses)):
            if metrics.val_losses[i] > metrics.val_losses[i-1] and metrics.train_losses[i] < metrics.train_losses[i-1]:
                ax.axvspan(epochs[i-1], epochs[i], alpha=0.2, color='red', label='_nolegend_')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training and Validation Loss{title_suffix}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add min loss annotations
    min_train_idx = metrics.train_losses.index(min(metrics.train_losses))
    ax.annotate(f'Min: {min(metrics.train_losses):.4f}',
                xy=(epochs[min_train_idx], min(metrics.train_losses)),
                xytext=(5, 10), textcoords='offset points', fontsize=9)

    if metrics.val_losses:
        min_val_idx = metrics.val_losses.index(min(metrics.val_losses))
        ax.annotate(f'Min: {min(metrics.val_losses):.4f}',
                    xy=(epochs[min_val_idx], min(metrics.val_losses)),
                    xytext=(5, -15), textcoords='offset points', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()
    logger.info(f"Saved loss curves to {output_dir / 'loss_curves.png'}")

    # 2. MRR and Hit@1 curves
    if metrics.val_mrr and metrics.val_hit1:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1, color2 = '#2E86AB', '#A23B72'

        ax1.plot(epochs, metrics.val_mrr, color=color1, marker='o', linewidth=2,
                 markersize=6, label='MRR')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MRR', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(epochs, metrics.val_hit1, color=color2, marker='s', linewidth=2,
                 markersize=6, label='Hit@1')
        ax2.set_ylabel('Hit@1', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

        ax1.set_title(f'Validation Metrics{title_suffix}', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add best metrics annotations
        best_mrr_idx = metrics.val_mrr.index(max(metrics.val_mrr))
        best_hit1_idx = metrics.val_hit1.index(max(metrics.val_hit1))
        ax1.annotate(f'Best: {max(metrics.val_mrr):.4f}',
                     xy=(epochs[best_mrr_idx], max(metrics.val_mrr)),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, color=color1)

        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_curves.png', dpi=150)
        plt.close()
        logger.info(f"Saved metrics curves to {output_dir / 'metrics_curves.png'}")

    # 3. Per-step loss (if available, for detecting within-epoch patterns)
    if metrics.step_losses and len(metrics.step_losses) > 10:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot with smoothing
        ax.plot(metrics.steps, metrics.step_losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw')

        # Add smoothed line (moving average)
        window = min(50, len(metrics.step_losses) // 10)
        if window > 1:
            smoothed = []
            for i in range(len(metrics.step_losses)):
                start = max(0, i - window // 2)
                end = min(len(metrics.step_losses), i + window // 2 + 1)
                smoothed.append(sum(metrics.step_losses[start:end]) / (end - start))
            ax.plot(metrics.steps, smoothed, 'b-', linewidth=2, label=f'Smoothed (window={window})')

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Per-Step Training Loss{title_suffix}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'step_loss.png', dpi=150)
        plt.close()
        logger.info(f"Saved step loss plot to {output_dir / 'step_loss.png'}")


def compute_siglip_loss(
    text_embeds: torch.Tensor,
    image_embeds_pos: torch.Tensor,
    image_embeds_neg: torch.Tensor,
    num_negatives: list,
    temperature: float = 1.0,
    bias: float = 0.0,
    use_in_batch_negatives: bool = False,  # Disabled by default for VWSD
) -> torch.Tensor:
    """
    Compute SigLIP-style contrastive loss with hard negatives.

    SigLIP uses sigmoid loss instead of softmax (no need for large batches):
    loss = -log(sigmoid(pos_sim)) - sum(log(sigmoid(-neg_sim)))

    Args:
        text_embeds: [B, D] text embeddings
        image_embeds_pos: [B, D] positive image embeddings
        image_embeds_neg: [N_total, D] all negative embeddings flattened
        num_negatives: list of negative counts per sample
        temperature: scaling factor
        bias: learnable bias term
        use_in_batch_negatives: Whether to use in-batch negatives.
            Disabled by default for VWSD since different instances may have
            semantically similar correct images (e.g., two "bank" instances
            both showing rivers), creating false negatives.
    """
    batch_size = text_embeds.size(0)

    # Normalize embeddings
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds_pos = F.normalize(image_embeds_pos, dim=-1)

    # Positive similarities
    pos_sim = (text_embeds * image_embeds_pos).sum(dim=-1)  # [B]
    pos_sim = pos_sim * temperature + bias

    # Positive loss: -log(sigmoid(pos_sim))
    pos_loss = -F.logsigmoid(pos_sim).mean()

    # Hard negative loss (if we have negatives)
    neg_loss = 0.0
    if image_embeds_neg is not None and image_embeds_neg.size(0) > 0:
        image_embeds_neg = F.normalize(image_embeds_neg, dim=-1)

        # Compute similarities for each sample's negatives
        neg_idx = 0
        neg_losses = []
        for i, num_neg in enumerate(num_negatives):
            if num_neg > 0:
                neg_embeds = image_embeds_neg[neg_idx:neg_idx + num_neg]  # [num_neg, D]
                neg_sim = text_embeds[i:i+1] @ neg_embeds.T  # [1, num_neg]
                neg_sim = neg_sim * temperature + bias
                # -log(sigmoid(-neg_sim)) = -log(1 - sigmoid(neg_sim))
                neg_losses.append(-F.logsigmoid(-neg_sim).mean())
                neg_idx += num_neg

        if neg_losses:
            neg_loss = torch.stack(neg_losses).mean()

    total_loss = pos_loss + neg_loss

    # In-batch negatives (optional - disabled by default for VWSD)
    # This treats other samples' positives as negatives, but in VWSD this
    # can create false negatives when different instances have similar images
    if use_in_batch_negatives and batch_size > 1:
        in_batch_sim = text_embeds @ image_embeds_pos.T  # [B, B]
        in_batch_sim = in_batch_sim * temperature + bias
        mask = torch.eye(batch_size, device=text_embeds.device, dtype=torch.bool)
        in_batch_neg_sim = in_batch_sim[~mask].view(batch_size, batch_size - 1)
        in_batch_neg_loss = -F.logsigmoid(-in_batch_neg_sim).mean()
        total_loss = total_loss + in_batch_neg_loss

    return total_loss


def compute_mrr_hit1(
    model,
    processor,
    data_file: str,
    gold_file: str,
    images_dir: str,
    device: torch.device,
    max_samples: int = 500,
) -> dict:
    """Compute MRR and Hit@1 on validation set.

    Uses the same inference approach as siglip2_inference.py:
    - Full forward pass with model(**inputs) to get logits_per_image
    - Applies sigmoid to get probabilities
    - This ensures LoRA adapters are properly applied
    """
    from PIL import Image

    model.eval()

    # Load validation data
    instances = []
    with open(data_file, 'r') as f_data, open(gold_file, 'r') as f_gold:
        for data_line, gold_line in zip(f_data, f_gold):
            parts = data_line.strip().split('\t')
            gold_image = gold_line.strip()
            instances.append({
                'target_word': parts[0],
                'full_phrase': parts[1],
                'candidates': parts[2:12],
                'gold': gold_image,
            })

    # Sample if too many
    if len(instances) > max_samples:
        import random
        instances = random.sample(instances, max_samples)

    images_dir = Path(images_dir)
    reciprocal_ranks = []
    hits_at_1 = []

    with torch.no_grad():
        for inst in tqdm(instances, desc="Validating", leave=False):
            # Lowercase to match SigLIP2 training and inference
            text = f"this is a photo of a {inst['target_word']}. {inst['full_phrase']}".lower()

            # Load candidate images
            images = []
            for img_name in inst['candidates']:
                try:
                    img = Image.open(images_dir / img_name).convert('RGB')
                    images.append(img)
                except:
                    images.append(Image.new('RGB', (384, 384), 'black'))

            # Use full forward pass like siglip2_inference.py
            # This ensures LoRA adapters are properly applied and uses model's
            # learned temperature/bias for similarity computation
            inputs = processor(
                text=[text],
                images=images,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs)

            # Get similarity scores using sigmoid (same as inference)
            logits_per_image = outputs.logits_per_image.squeeze(-1)  # [num_images]
            probs = torch.sigmoid(logits_per_image)

            # Rank by probability (highest first)
            ranked_indices = probs.argsort(descending=True).cpu().tolist()

            # Find gold rank
            gold_idx = inst['candidates'].index(inst['gold']) if inst['gold'] in inst['candidates'] else -1
            if gold_idx >= 0:
                rank = ranked_indices.index(gold_idx) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
                hits_at_1.append(1 if rank == 1 else 0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    hit1 = sum(hits_at_1) / len(hits_at_1) if hits_at_1 else 0

    model.train()
    return {'mrr': mrr, 'hit@1': hit1}


def compute_mrr_hit1_from_dataset(
    model,
    processor,
    dataset,
    images_dir: str,
    device: torch.device,
    max_samples: int = 500,
) -> dict:
    """Compute MRR and Hit@1 on a validation dataset split.

    Similar to compute_mrr_hit1 but works with a PyTorch dataset instead of files.
    """
    from PIL import Image
    import random

    model.eval()

    # Get indices to evaluate
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        indices = random.sample(indices, max_samples)

    images_path = Path(images_dir)
    reciprocal_ranks = []
    hits_at_1 = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Validating", leave=False):
            item = dataset[idx]

            # Get original text (use original phrase, not augmented)
            target_word = item['target_word']
            full_phrase = item['full_phrase']
            text = f"this is a photo of a {target_word}. {full_phrase}".lower()

            # Get positive and negative images from the item
            pos_img = item['positive_image']
            neg_imgs = item['negative_images']

            # Combine all images (positive + negatives)
            all_images = [pos_img] + neg_imgs
            gold_idx = 0  # Positive is always first

            # Use full forward pass
            inputs = processor(
                text=[text],
                images=all_images,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs)

            # Get similarity scores using sigmoid
            logits_per_image = outputs.logits_per_image.squeeze(-1)
            probs = torch.sigmoid(logits_per_image)

            # Rank by probability
            ranked_indices = probs.argsort(descending=True).cpu().tolist()

            # Find gold rank
            rank = ranked_indices.index(gold_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
            hits_at_1.append(1 if rank == 1 else 0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    hit1 = sum(hits_at_1) / len(hits_at_1) if hits_at_1 else 0

    model.train()
    return {'mrr': mrr, 'hit@1': hit1}


def compute_validation_loss(
    model,
    val_loader,
    device: torch.device,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> float:
    """Compute average validation loss over the validation loader.

    Uses the same loss function as training but without gradient computation.
    This allows direct comparison of train/val loss to detect overfitting.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing val loss", leave=False):
            if batch is None:
                continue

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            pixel_values_pos = batch['pixel_values_pos'].to(device)
            pixel_values_neg = batch['pixel_values_neg'].to(device) if batch['pixel_values_neg'] is not None else None
            num_negatives = batch['num_negatives']

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Get embeddings
                text_kwargs = {'input_ids': input_ids}
                if attention_mask is not None:
                    text_kwargs['attention_mask'] = attention_mask
                text_embeds = model.get_text_features(**text_kwargs)
                image_embeds_pos = model.get_image_features(pixel_values=pixel_values_pos)

                if pixel_values_neg is not None:
                    image_embeds_neg = model.get_image_features(pixel_values=pixel_values_neg)
                else:
                    image_embeds_neg = None

                # Compute loss
                loss = compute_siglip_loss(
                    text_embeds=text_embeds,
                    image_embeds_pos=image_embeds_pos,
                    image_embeds_neg=image_embeds_neg,
                    num_negatives=num_negatives,
                    temperature=temperature,
                    bias=bias,
                    use_in_batch_negatives=False,
                )

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def collate_fn(batch, processor):
    """Collate function for contrastive dataset."""
    texts = [item['text'] for item in batch]
    positive_images = [item['positive_image'] for item in batch]
    all_negative_images = []
    num_negatives = []

    for item in batch:
        negs = item['negative_images']
        all_negative_images.extend(negs)
        num_negatives.append(len(negs))

    # Process texts and images together for correct tokenization
    # SigLIP2 processor returns input_ids and pixel_values
    combined_inputs = processor(
        text=texts,
        images=positive_images,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    # Process negatives separately
    if all_negative_images:
        neg_inputs = processor(images=all_negative_images, return_tensors='pt')
        pixel_values_neg = neg_inputs['pixel_values']
    else:
        pixel_values_neg = None

    # Build return dict - handle optional attention_mask
    result = {
        'input_ids': combined_inputs['input_ids'],
        'pixel_values_pos': combined_inputs['pixel_values'],
        'pixel_values_neg': pixel_values_neg,
        'num_negatives': num_negatives,
    }

    # Add attention_mask if present (might not be for some processors)
    if 'attention_mask' in combined_inputs:
        result['attention_mask'] = combined_inputs['attention_mask']

    return result


def get_lora_target_modules(include_mlp: bool = False) -> list:
    """
    Get target modules for LoRA in SigLIP2.

    SigLIP2 architecture has:
    - text_model.encoder.layers.*.self_attn.{q_proj,v_proj,k_proj,out_proj}
    - vision_model.encoder.layers.*.self_attn.{q_proj,v_proj,k_proj,out_proj}
    - Optionally MLP layers: fc1, fc2
    """
    # Attention layers (always include)
    targets = ["q_proj", "v_proj", "k_proj", "out_proj"]

    # MLP layers (optional, increases trainable params)
    if include_mlp:
        targets.extend(["fc1", "fc2"])

    return targets


def create_lora_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    include_mlp: bool = False,
    device: str = "cuda",
):
    """
    Load base model and apply LoRA adapters.

    Args:
        model_name: HuggingFace model name
        lora_r: LoRA rank (lower = fewer params, higher = more capacity)
        lora_alpha: LoRA scaling factor (typically 2*r)
        lora_dropout: Dropout for LoRA layers
        include_mlp: Whether to apply LoRA to MLP layers too
        device: Device to use

    Returns:
        PEFT model with LoRA adapters
    """
    print(f"Loading base model: {model_name}")

    # Load base model in bfloat16
    base_model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Configure LoRA
    target_modules = get_lora_target_modules(include_mlp)
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # Don't train biases
    )

    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_desc = f"r{args.lora_r}_alpha{args.lora_alpha}"
    if args.include_mlp:
        lora_desc += "_mlp"
    output_dir = Path(args.output_dir) / f"siglip2_lora_{lora_desc}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model with LoRA
    model = create_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        include_mlp=args.include_mlp,
        device=device,
    )

    # Load processor with use_fast=True to avoid deprecation warning
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    # Load dataset - with optional augmentation
    train_dir = Path(args.data_dir) / "train_v1"

    if args.augmentation_file and Path(args.augmentation_file).exists():
        print(f"Using augmented dataset with aug_prob={args.aug_prob}")
        train_dataset = VWSDAugmentedDataset(
            data_file=str(train_dir / "train.data.v1.txt"),
            gold_file=str(train_dir / "train.gold.v1.txt"),
            images_dir=str(train_dir / "train_images_v1"),
            processor=processor,
            augmentation_file=args.augmentation_file,
            aug_prob=args.aug_prob,
            aug_types=args.aug_types,
            num_hard_negatives=args.num_hard_negatives,
            cache_images=args.cache_images,
            num_loading_threads=args.image_loading_threads,
        )
    else:
        print("Using standard dataset (no augmentation)")
        train_dataset = VWSDContrastiveDataset(
            data_file=str(train_dir / "train.data.v1.txt"),
            gold_file=str(train_dir / "train.gold.v1.txt"),
            images_dir=str(train_dir / "train_images_v1"),
            processor=processor,
            num_hard_negatives=args.num_hard_negatives,
            cache_images=args.cache_images,
            num_loading_threads=args.image_loading_threads,
        )

    # Optional: split training data for validation instead of using test set
    val_dataset = None
    if args.val_split > 0 and not args.use_test_for_val:
        from torch.utils.data import random_split
        total_size = len(train_dataset)
        val_size = int(total_size * args.val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Train/Val split: {train_size}/{val_size} ({args.val_split:.0%} validation)")
    else:
        print(f"Train size: {len(train_dataset)} (validation on test set)")

    # DataLoaders - key settings for GPU utilization:
    # - num_workers: parallel data loading processes (increase to keep GPU fed)
    # - prefetch_factor: batches to prefetch per worker (default 2, increase for faster loading)
    # - persistent_workers: keep workers alive between epochs (avoids respawn overhead)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, processor=processor),
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,  # Keep workers alive between epochs
    )

    # Optimizer - LoRA typically uses higher learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    # Scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Note: GradScaler is NOT needed for bfloat16 (same exponent range as float32)
    # Using it with bfloat16 can cause training instability

    # Learnable temperature and bias (like SigLIP)
    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))
    optimizer.add_param_group({'params': [log_temperature, bias], 'lr': args.lr})

    # Create validation DataLoader (for computing validation loss)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(collate_fn, processor=processor),
            pin_memory=True,
            drop_last=False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

    # Training loop
    best_mrr = 0.0
    global_step = 0

    # Initialize metrics tracking
    metrics = TrainingMetrics()

    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  LoRA rank (r): {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  LoRA dropout: {args.lora_dropout}")
    print(f"  Include MLP: {args.include_mlp}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Hard negatives per sample: {args.num_hard_negatives}")
    print(f"  Use in-batch negatives: {args.use_in_batch_negatives}")
    print(f"Text Augmentation:")
    if args.augmentation_file:
        print(f"  Augmentation file: {args.augmentation_file}")
        print(f"  Augmentation probability: {args.aug_prob:.0%}")
        print(f"  Augmentation types: {args.aug_types}")
    else:
        print(f"  Augmentation: disabled")
    print(f"Validation:")
    if val_dataset is not None:
        print(f"  Validation split: {args.val_split:.0%} of training data")
    else:
        print(f"  Validation: test set")
    print(f"Data Loading (GPU utilization):")
    print(f"  DataLoader workers: {args.num_workers}")
    print(f"  Prefetch factor: {args.prefetch_factor}")
    print(f"  Image cache: {'enabled' if args.cache_images else 'disabled'}")
    print(f"  Image loading threads: {args.image_loading_threads}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            pixel_values_pos = batch['pixel_values_pos'].to(device)
            pixel_values_neg = batch['pixel_values_neg'].to(device) if batch['pixel_values_neg'] is not None else None
            num_negatives = batch['num_negatives']

            # Forward pass with mixed precision (bfloat16)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Get embeddings - pass attention_mask if available
                text_kwargs = {'input_ids': input_ids}
                if attention_mask is not None:
                    text_kwargs['attention_mask'] = attention_mask
                text_embeds = model.get_text_features(**text_kwargs)
                image_embeds_pos = model.get_image_features(pixel_values=pixel_values_pos)

                if pixel_values_neg is not None:
                    image_embeds_neg = model.get_image_features(pixel_values=pixel_values_neg)
                else:
                    image_embeds_neg = None

                # Compute loss
                temperature = torch.exp(log_temperature)
                loss = compute_siglip_loss(
                    text_embeds=text_embeds,
                    image_embeds_pos=image_embeds_pos,
                    image_embeds_neg=image_embeds_neg,
                    num_negatives=num_negatives,
                    temperature=temperature.item(),
                    bias=bias.item(),
                    use_in_batch_negatives=args.use_in_batch_negatives,
                )

                loss = loss / args.gradient_accumulation_steps

            # Backward pass (no GradScaler needed for bfloat16)
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Track per-step loss (after gradient accumulation)
                step_loss = loss.item() * args.gradient_accumulation_steps
                metrics.add_step_loss(global_step, step_loss)

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{epoch_loss / num_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'temp': f"{temperature.item():.2f}",
            })

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

        # Current temperature for metrics
        current_temp = torch.exp(log_temperature).item()
        current_lr = scheduler.get_last_lr()[0]

        # Validation - use train split or TEST data depending on config
        val_loss = None
        val_metrics = None

        if (epoch + 1) % args.val_every == 0:
            # Compute validation loss if we have a validation loader
            if val_loader is not None:
                print("Computing validation loss...")
                val_loss = compute_validation_loss(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    temperature=current_temp,
                    bias=bias.item(),
                )
                print(f"Validation loss: {val_loss:.4f}")

                # Also compute MRR/Hit@1 on validation split
                print("Computing validation metrics (MRR, Hit@1)...")
                val_metrics = compute_mrr_hit1_from_dataset(
                    model=model,
                    processor=processor,
                    dataset=val_dataset,
                    images_dir=str(train_dir / "train_images_v1"),
                    device=device,
                    max_samples=args.val_samples,
                )
            else:
                # Use TEST data for validation
                print("Running validation on TEST set...")
                test_dir = Path(args.data_dir).parent / "test_data"
                test_images_dir = Path(args.data_dir).parent / "test_images" / "test_images_resized"
                val_metrics = compute_mrr_hit1(
                    model=model,
                    processor=processor,
                    data_file=str(test_dir / "en.test.data.v1.1.txt"),
                    gold_file=str(test_dir / "en.test.gold.v1.1.txt"),
                    images_dir=str(test_images_dir),
                    device=device,
                    max_samples=args.val_samples,
                )

            print(f"Validation - MRR: {val_metrics['mrr']:.4f}, Hit@1: {val_metrics['hit@1']:.4f}")

            # Detect potential overfitting
            if val_loss is not None and len(metrics.val_losses) > 0:
                prev_val_loss = metrics.val_losses[-1]
                if val_loss > prev_val_loss and avg_loss < metrics.train_losses[-1]:
                    logger.warning(
                        f"⚠️  Potential overfitting detected! "
                        f"Train loss ↓ ({metrics.train_losses[-1]:.4f} → {avg_loss:.4f}) "
                        f"but Val loss ↑ ({prev_val_loss:.4f} → {val_loss:.4f})"
                    )

            # Save best model
            if val_metrics['mrr'] > best_mrr:
                best_mrr = val_metrics['mrr']
                print(f"New best MRR! Saving model...")

                # Save LoRA adapters
                model.save_pretrained(output_dir / "best_model")
                processor.save_pretrained(output_dir / "best_model")

                # Save training info
                with open(output_dir / "best_model" / "training_info.json", 'w') as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'mrr': val_metrics['mrr'],
                        'hit@1': val_metrics['hit@1'],
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                        'lora_config': {
                            'r': args.lora_r,
                            'alpha': args.lora_alpha,
                            'dropout': args.lora_dropout,
                            'include_mlp': args.include_mlp,
                        },
                        'args': vars(args),
                    }, f, indent=2)

        # Track epoch metrics
        metrics.add_epoch_metrics(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=val_loss,
            mrr=val_metrics['mrr'] if val_metrics else None,
            hit1=val_metrics['hit@1'] if val_metrics else None,
            lr=current_lr,
            temp=current_temp,
        )

        # Print epoch summary with gap analysis
        if val_loss is not None:
            gap = val_loss - avg_loss
            gap_indicator = "↑" if gap > 0 else "↓"
            print(f"Epoch {epoch+1} Summary: "
                  f"Train={avg_loss:.4f}, Val={val_loss:.4f}, "
                  f"Gap={gap:.4f}{gap_indicator}, "
                  f"MRR={val_metrics['mrr']:.4f}, Hit@1={val_metrics['hit@1']:.4f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint_epoch{epoch+1}"
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")

    # Also save merged model for easy inference
    if args.save_merged:
        print("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir / "merged_model")
        processor.save_pretrained(output_dir / "merged_model")
        print(f"Saved merged model to {output_dir / 'merged_model'}")

    # Save training metrics
    metrics.save(str(output_dir / "training_metrics.json"))

    # Generate training curves
    title_suffix = ""
    if args.augmentation_file:
        title_suffix = f" (aug={args.aug_prob:.0%})"
    plot_training_curves(metrics, output_dir, title_suffix)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best MRR: {best_mrr:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - training_metrics.json  (raw metrics data)")
    print(f"  - loss_curves.png        (train/val loss visualization)")
    if metrics.val_mrr:
        print(f"  - metrics_curves.png     (MRR/Hit@1 visualization)")
    if metrics.step_losses:
        print(f"  - step_loss.png          (per-step loss visualization)")
    print(f"\nTo detect overfitting:")
    print(f"  - Look for diverging train/val loss curves")
    print(f"  - Val loss increasing while train loss decreases = overfitting")
    print(f"  - Consider: --augmentation-file, lower --epochs, higher --lora-dropout")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tune SigLIP2 for VWSD")

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/siglip2-base-patch16-512",
        help="HuggingFace model name (default: siglip2-base for memory efficiency)"
    )

    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (8, 16, 32, 64 - higher = more params)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (typically 2*r)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--include-mlp",
        action="store_true",
        help="Apply LoRA to MLP layers (fc1, fc2) in addition to attention"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/semeval-2023-task-1-V-WSD-train-v1",
        help="Path to VWSD training data"
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=5,
        help="Number of hard negatives per positive (max 9, reduce if OOM)"
    )
    parser.add_argument(
        "--use-in-batch-negatives",
        action="store_true",
        help="Use in-batch negatives (disabled by default, may cause false negatives in VWSD)"
    )

    # Text augmentation (to prevent overfitting)
    parser.add_argument(
        "--augmentation-file",
        type=str,
        default=None,
        help="Path to augmentation cache JSON (enables VLM-based text augmentation)"
    )
    parser.add_argument(
        "--aug-prob",
        type=float,
        default=0.5,
        help="Probability of using augmented text per sample (default: 0.5)"
    )
    parser.add_argument(
        "--aug-types",
        type=str,
        nargs='+',
        default=["caption", "definition"],
        choices=["caption", "definition", "paraphrase"],
        help="Augmentation types to use (default: caption definition)"
    )

    # Validation split (from training data, to properly detect overfitting)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of training data for validation (default: 0.0 = use test set)"
    )
    parser.add_argument(
        "--use-test-for-val",
        action="store_true",
        help="Force use of test set for validation even if val-split > 0"
    )

    # Training - LoRA typically uses higher learning rates
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (fewer to prevent overfitting)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (2e-5 recommended for LoRA)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (higher = larger effective batch)"
    )

    # Validation (uses TEST data, not a train split)
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--val-samples", type=int, default=500, help="Max validation samples")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetune/checkpoints",
        help="Output directory"
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--save-merged", action="store_true", help="Save merged model at the end")

    # Data loading optimization (critical for GPU utilization)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="DataLoader workers (increase to keep GPU fed, 8-16 recommended)"
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches to prefetch per worker (default 4, increase if GPU still drops)"
    )
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Cache loaded images in RAM (~10GB for full dataset, eliminates disk I/O)"
    )
    parser.add_argument(
        "--image-loading-threads",
        type=int,
        default=4,
        help="Threads for parallel image loading within each sample (4-8 recommended)"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
