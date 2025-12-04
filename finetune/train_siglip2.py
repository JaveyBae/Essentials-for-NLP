"""
SigLIP2 Fine-tuning for Visual Word Sense Disambiguation

Full fine-tuning with contrastive loss using hard negatives from VWSD dataset.

Usage:
    python finetune/train_siglip2.py --epochs 10 --batch-size 16 --lr 1e-5

Key features:
- Contrastive loss with in-batch negatives + hard negatives
- Gradient accumulation for larger effective batch size
- Mixed precision training (bfloat16)
- Validation with MRR/Hit@1 metrics
- Checkpoint saving with best model tracking
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from finetune.vwsd_dataset import VWSDContrastiveDataset, VWSDPairDataset


def compute_siglip_loss(
    text_embeds: torch.Tensor,
    image_embeds_pos: torch.Tensor,
    image_embeds_neg: torch.Tensor,
    num_negatives: list,
    temperature: float = 1.0,
    bias: float = 0.0,
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

    # In-batch negatives (treat other samples' positives as negatives)
    # Similarity matrix: [B, B]
    in_batch_sim = text_embeds @ image_embeds_pos.T  # [B, B]
    in_batch_sim = in_batch_sim * temperature + bias

    # Mask diagonal (positive pairs)
    mask = torch.eye(batch_size, device=text_embeds.device, dtype=torch.bool)
    in_batch_neg_sim = in_batch_sim[~mask].view(batch_size, batch_size - 1)

    # In-batch negative loss
    in_batch_neg_loss = -F.logsigmoid(-in_batch_neg_sim).mean()

    total_loss = pos_loss + neg_loss + in_batch_neg_loss
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
    """Compute MRR and Hit@1 on validation set."""
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
            text = f"this is a photo of a {inst['target_word']}. {inst['full_phrase']}"

            # Load candidate images
            images = []
            for img_name in inst['candidates']:
                try:
                    img = Image.open(images_dir / img_name).convert('RGB')
                    images.append(img)
                except:
                    images.append(Image.new('RGB', (384, 384), 'black'))

            # Get embeddings (SigLIP2 doesn't use attention_mask)
            text_inputs = processor(text=text, return_tensors='pt', padding="max_length", max_length=64).to(device)
            image_inputs = processor(images=images, return_tensors='pt').to(device)

            # Forward pass (no autocast needed, works for both F32 and bfloat16)
            text_embeds = model.get_text_features(input_ids=text_inputs['input_ids'])
            image_embeds = model.get_image_features(pixel_values=image_inputs['pixel_values'])

            text_embeds = F.normalize(text_embeds, dim=-1)
            image_embeds = F.normalize(image_embeds, dim=-1)

            # Compute similarities and rank
            similarities = (text_embeds @ image_embeds.T).squeeze(0)  # [10]
            ranked_indices = similarities.argsort(descending=True).cpu().tolist()

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


def collate_fn(batch, processor):
    """Collate function for contrastive dataset."""
    texts = [item['text'] for item in batch]
    positive_images = [item['positive_image'] for item in batch]

    # Flatten negatives (images are now always valid, with black placeholders for errors)
    all_negatives = []
    num_negatives = []
    for item in batch:
        negs = item['negative_images']
        all_negatives.extend(negs)
        num_negatives.append(len(negs))

    # Process texts and images together (SigLIP2 style)
    # Note: SigLIP2 processor doesn't return attention_mask for text
    text_inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    # Process positive images
    pos_inputs = processor(images=positive_images, return_tensors='pt')

    # Process negatives
    if all_negatives:
        neg_inputs = processor(images=all_negatives, return_tensors='pt')
        pixel_values_neg = neg_inputs['pixel_values']
    else:
        pixel_values_neg = None

    return {
        'input_ids': text_inputs['input_ids'],
        'pixel_values_pos': pos_inputs['pixel_values'],
        'pixel_values_neg': pixel_values_neg,
        'num_negatives': num_negatives,
    }


def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"siglip2_vwsd_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model and processor
    # Base/Large models: use F32 with eager attention (more stable)
    # So400m/Giant models: use bfloat16 with SDPA (faster)
    print(f"Loading model: {args.model_name}")

    is_base_or_large = "base" in args.model_name.lower() or "large" in args.model_name.lower()

    if is_base_or_large:
        print("  Using F32 with eager attention (recommended for base/large models)")
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
    else:
        print("  Using bfloat16 with SDPA (recommended for so400m/giant models)")
        model = AutoModel.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(device)

    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load dataset
    train_dir = Path(args.data_dir) / "train_v1"
    dataset = VWSDContrastiveDataset(
        data_file=str(train_dir / "train.data.v1.txt"),
        gold_file=str(train_dir / "train.gold.v1.txt"),
        images_dir=str(train_dir / "train_images_v1"),
        processor=processor,
        num_hard_negatives=args.num_hard_negatives,
    )

    # Split into train/val
    val_size = min(int(len(dataset) * args.val_split), 1000)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, processor=processor),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
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

    # Determine dtype for autocast based on model type
    use_amp = not is_base_or_large  # Only use AMP for so400m/giant models
    amp_dtype = torch.bfloat16

    # Learnable temperature and bias (like SigLIP)
    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))
    optimizer.add_param_group({'params': [log_temperature, bias], 'lr': args.lr})

    # Training loop
    best_mrr = 0.0
    global_step = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Total steps: {num_training_steps}")

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
            pixel_values_pos = batch['pixel_values_pos'].to(device)
            pixel_values_neg = batch['pixel_values_neg'].to(device) if batch['pixel_values_neg'] is not None else None
            num_negatives = batch['num_negatives']

            # Forward pass (with optional mixed precision for so400m/giant models)
            if use_amp:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    text_embeds = model.get_text_features(input_ids=input_ids)
                    image_embeds_pos = model.get_image_features(pixel_values=pixel_values_pos)
                    if pixel_values_neg is not None:
                        image_embeds_neg = model.get_image_features(pixel_values=pixel_values_neg)
                    else:
                        image_embeds_neg = None
            else:
                # F32 for base/large models
                text_embeds = model.get_text_features(input_ids=input_ids)
                image_embeds_pos = model.get_image_features(pixel_values=pixel_values_pos)
                if pixel_values_neg is not None:
                    image_embeds_neg = model.get_image_features(pixel_values=pixel_values_neg)
                else:
                    image_embeds_neg = None

            # Compute loss (always in F32 for stability)
            temperature = torch.exp(log_temperature)
            loss = compute_siglip_loss(
                text_embeds=text_embeds.float(),
                image_embeds_pos=image_embeds_pos.float(),
                image_embeds_neg=image_embeds_neg.float() if image_embeds_neg is not None else None,
                num_negatives=num_negatives,
                temperature=temperature.item(),
                bias=bias.item(),
            )

            loss = loss / args.gradient_accumulation_steps

            # Backward pass (no scaler needed for bfloat16)
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{epoch_loss / num_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'temp': f"{temperature.item():.2f}",
            })

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_every == 0:
            print("Running validation...")
            metrics = compute_mrr_hit1(
                model=model,
                processor=processor,
                data_file=str(train_dir / "train.data.v1.txt"),
                gold_file=str(train_dir / "train.gold.v1.txt"),
                images_dir=str(train_dir / "train_images_v1"),
                device=device,
                max_samples=args.val_samples,
            )
            print(f"Validation - MRR: {metrics['mrr']:.4f}, Hit@1: {metrics['hit@1']:.4f}")

            # Save best model
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                print(f"New best MRR! Saving model...")
                model.save_pretrained(output_dir / "best_model")
                processor.save_pretrained(output_dir / "best_model")

                # Save training info
                with open(output_dir / "best_model" / "training_info.json", 'w') as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'mrr': metrics['mrr'],
                        'hit@1': metrics['hit@1'],
                        'loss': avg_loss,
                        'args': vars(args),
                    }, f, indent=2)

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint_epoch{epoch+1}"
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")
    print(f"\nTraining complete! Best MRR: {best_mrr:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP2 for VWSD")

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/siglip2-base-patch16-512",
        help="HuggingFace model name (default: siglip2-base-patch16-512, 0.4B params, ~8GB VRAM)"
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
        default=4,
        help="Number of hard negatives per positive (max 9)"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8 for F32 models)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    # Validation
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
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
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
