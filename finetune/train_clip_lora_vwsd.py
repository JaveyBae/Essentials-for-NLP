import os
from pathlib import Path
import random
from typing import List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"



ROOT = Path(__file__).resolve().parent
LABEL_DIR = ROOT.parent / "dataset" / "label"
IMAGE_DIR = ROOT.parent / "dataset" / "image" / "train_images_v1"



# ============================================================
# Safe Image Loader
# ============================================================

def safe_load(p: Path):
    """Verlässlicher Bildloader."""
    try:
        with Image.open(p) as img:
            img.verify()
        img = Image.open(p).convert("RGB")
        img = img.resize((224, 224))
        return img
    except Exception:
        return None


# ============================================================
# VWSD dataclasses
# ============================================================

@dataclass
class VwsdExample:
    text: str
    image_paths: List[Path]
    label: int


class VwsdDataset(Dataset):
    def __init__(self, ex):
        self.examples = ex

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "text": ex.text,
            "image_paths": ex.image_paths,
            "label": ex.label
        }


# ============================================================
# VWSD Loader
# ============================================================

def load_vwsd(limit: int = None):
    td = LABEL_DIR / "train.data.v1.augmented.txt"
    print("loaded augmented dataset")
    #td = LABEL_DIR / "train.data.v1.txt"
    tg = LABEL_DIR / "train.gold.v1.txt"

    print(">>>Checking dataset paths:")
    print("  train.data:", td, "exists:", td.exists())
    print("  train.gold:", tg, "exists:", tg.exists())

    if not (td.exists() and tg.exists()):
        raise FileNotFoundError("VWSD-Dateien fehlen.")

    examples = []

    with td.open() as f_data, tg.open() as f_gold:
        for line, gold in zip(f_data, f_gold):
            parts = line.strip().split("\t")
            gold = gold.strip()

            if len(parts) != 12:
                continue

            text = parts[0]
            candidates = parts[2:]

            if len(candidates) != 10:
                continue
            if gold not in candidates:
                continue

            label = candidates.index(gold)
            img_paths = [IMAGE_DIR / c for c in candidates]

            examples.append(VwsdExample(text, img_paths, label))

    if limit is not None:
        examples = examples[:limit]

    print(f">>> Loaded {len(examples)} VWSD samples.")
    return examples


# ============================================================
# LoRA Training
# ============================================================

def train_lora(
        model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        output_dir="lora_laion_vwsd_text_lora",
        lr=1e-5,
        epochs=1,
        limit_train=500
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> Device:", device)

    # 1) Base CLIP load
    print(">>> Loading CLIP", model_name)
    base_model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 2) Vision freeze
    base_model.requires_grad_(False)
    for p in base_model.vision_model.parameters():
        p.requires_grad = False

    # Nur logit_scale + text_model LoRA trainieren
    base_model.logit_scale.requires_grad = True

    # 3) LoRA config (Att + MLP)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj"                          # MLP
        ],
        bias="none"
    )

    base_model.text_model = get_peft_model(base_model.text_model, lora_cfg)
    model = base_model.to(device)
    model.train()

    try:
        print(">>> Trainable LoRA params:")
        model.text_model.print_trainable_parameters()
    except:
        pass

    # 4) load data
    examples = load_vwsd(limit=limit_train)
    dataset = VwsdDataset(examples)

    # 5) Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )

    # 6) Training
    print(f">>> Starting training: {len(dataset)} samples, {epochs} epochs")
    for epoch in range(epochs):
        print(f"\n--- EPOCH {epoch+1}/{epochs} ---")

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        progress = tqdm(indices, ncols=120)
        for i in progress:
            ex = dataset[i]

            imgs = []
            for p in ex["image_paths"]:
                img = safe_load(p)
                if img is None:
                    imgs = []
                    break
                imgs.append(img)
            if not imgs:
                continue

            text = ex["text"]
            label = ex["label"]

            # prepare inputs
            text_in = processor(text=[text], padding=True, return_tensors="pt")
            text_in = {k: v.to(device) for k, v in text_in.items()}

            img_in = processor(images=imgs, return_tensors="pt")
            img_in = {k: v.to(device) for k, v in img_in.items()}

            # Features extraction
            txt = model.get_text_features(**text_in)
            imgf = model.get_image_features(**img_in)

            # Normierung
            txt = txt / txt.norm(dim=-1, keepdim=True)
            imgf = imgf / imgf.norm(dim=-1, keepdim=True)

            logits = model.logit_scale.exp() * (txt @ imgf.t())
            loss = F.cross_entropy(logits, torch.tensor([label], device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix({"loss": float(loss)})

        print(f"Epoch {epoch+1} finished.")

    # 7) LoRA save
    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    model.text_model.save_pretrained(out)
    processor.save_pretrained(out)

    print(">>> Saved LoRA to:", out)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_lora(
        epochs=1,
        limit_train=500
    )
