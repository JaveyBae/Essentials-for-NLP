#!/usr/bin/env python3
"""
VLM-Based Text Augmentation Generator for VWSD Training Data.

Generates diverse text queries for SigLIP2 fine-tuning to prevent overfitting.
Reuses existing VLM infrastructure from src/ modules.

Three augmentation types:
1. Caption: VLM generates image descriptions for gold images (Qwen3-VL)
2. Definition: Generate visual definitions for word senses (Qwen3 text model)
3. Paraphrase: Generate context phrase variations (Qwen3 text model)

Usage:
    # Generate all augmentations (~4 hours on 32GB GPU)
    python finetune/generate_augmentations.py --type all

    # Generate specific type
    python finetune/generate_augmentations.py --type caption --model qwen3-vl-8b
    python finetune/generate_augmentations.py --type definition --model qwen3-8b
    python finetune/generate_augmentations.py --type paraphrase --model qwen3-8b --num-variants 3

    # Resume from checkpoint
    python finetune/generate_augmentations.py --type caption --resume
"""

import argparse
import json
import hashlib
import sys
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
from PIL import Image
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainInstance:
    """Training instance data."""
    target_word: str
    full_phrase: str
    gold_image: str
    negative_images: List[str]


class AugmentationCache:
    """Persistent cache for text augmentations (follows DefinitionCache pattern)."""

    def __init__(self, cache_dir: str = "results/text_augmentations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, target_word: str, full_phrase: str) -> str:
        """Generate unique cache key (matches DefinitionCache pattern)."""
        phrase_hash = hashlib.md5(full_phrase.encode('utf-8')).hexdigest()[:8]
        return f"{target_word}_{phrase_hash}"

    def get_cache_path(self, aug_type: str, model_name: str) -> Path:
        """Get cache file path for specific augmentation type and model.

        Note: Only English training data exists, so no language suffix needed.
        """
        filename = f"train_en_{aug_type}_{model_name}.json"
        return self.cache_dir / filename

    def load(self, aug_type: str, model_name: str) -> int:
        """Load cache from disk."""
        cache_path = self.get_cache_path(aug_type, model_name)

        if not cache_path.exists():
            logger.info(f"No cache found at {cache_path}")
            return 0

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached augmentations from {cache_path}")
            return len(self.cache)
        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {e}")
            self.cache = {}
            return 0

    def save(self, aug_type: str, model_name: str):
        """Save cache to disk (sorted for readability)."""
        cache_path = self.get_cache_path(aug_type, model_name)

        try:
            sorted_cache = dict(sorted(
                self.cache.items(),
                key=lambda x: (x[1].get('target_word', ''), x[1].get('full_phrase', ''))
            ))

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.cache)} augmentations to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache to {cache_path}: {e}")

    def get(self, target_word: str, full_phrase: str) -> Optional[Dict]:
        """Retrieve augmentation data from cache."""
        key = self._get_cache_key(target_word, full_phrase)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def set(
        self,
        target_word: str,
        full_phrase: str,
        gold_image: str,
        augmentation: any,
        aug_type: str,
        model_name: str
    ):
        """Store augmentation in cache."""
        key = self._get_cache_key(target_word, full_phrase)
        self.cache[key] = {
            "target_word": target_word,
            "full_phrase": full_phrase,
            "gold_image": gold_image,
            "augmentation": augmentation,
            "type": aug_type,
            "model": model_name,
            "timestamp": datetime.now().isoformat()
        }

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": self.hits / max(1, self.hits + self.misses)
        }


def load_training_data(data_dir: str, language: str = "en") -> List[TrainInstance]:
    """Load VWSD training data.

    Args:
        data_dir: Root data directory
        language: Language code (default: 'en', only 'en' is supported for training)

    Returns:
        List of training instances
    """
    if language != "en":
        logger.warning(f"Only English ('en') training data is available. Ignoring language='{language}'")
        language = "en"

    train_dir = Path(data_dir) / "semeval-2023-task-1-V-WSD-train-v1" / "train_v1"
    data_file = train_dir / "train.data.v1.txt"
    gold_file = train_dir / "train.gold.v1.txt"

    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found at {data_file}")

    instances = []
    with open(data_file, 'r') as f_data, open(gold_file, 'r') as f_gold:
        for data_line, gold_line in zip(f_data, f_gold):
            parts = data_line.strip().split('\t')
            gold_image = gold_line.strip()

            target_word = parts[0]
            full_phrase = parts[1]
            candidate_images = parts[2:12]

            negative_images = [img for img in candidate_images if img != gold_image]

            instances.append(TrainInstance(
                target_word=target_word,
                full_phrase=full_phrase,
                gold_image=gold_image,
                negative_images=negative_images
            ))

    logger.info(f"Loaded {len(instances)} training instances")
    return instances


def generate_captions(
    instances: List[TrainInstance],
    images_dir: str,
    model_name: str = "qwen3-vl-8b",
    batch_size: int = 8,
    cache: Optional[AugmentationCache] = None,
    resume: bool = False
) -> Dict[str, str]:
    """
    Generate captions for gold images using Qwen3-VL.

    Images are loaded on-demand per batch to avoid memory explosion.

    Args:
        instances: Training instances
        images_dir: Directory containing training images
        model_name: VLM model to use
        batch_size: Batch size for generation
        cache: Optional cache for resume support
        resume: Whether to resume from previous run

    Returns:
        Dictionary mapping cache_key -> caption
    """
    from src.qwen_vlm_loader import load_model, Qwen3VLModelLoader

    # Filter instances that need generation BEFORE loading model
    if resume and cache:
        instances_to_generate = []
        for inst in instances:
            if cache.get(inst.target_word, inst.full_phrase) is None:
                instances_to_generate.append(inst)
        logger.info(f"Resuming: {len(instances_to_generate)} instances need caption generation")
    else:
        instances_to_generate = instances

    if not instances_to_generate:
        logger.info("All captions already cached")
        return {}

    # Load VLM model
    logger.info(f"Loading VLM model: {model_name}")
    model, processor = load_model(model_name=model_name, use_cache=False)

    # Caption prompt (from src/qwen_vlm_inference.py)
    caption_prompt = '''Describe this image in detail. Focus on:
1. What objects, people, or scenes are shown
2. What actions or activities are taking place
3. The setting or environment

Provide a concise but comprehensive description in 2-3 sentences.'''

    images_path = Path(images_dir) / "semeval-2023-task-1-V-WSD-train-v1" / "train_v1" / "train_images_v1"

    captions = {}
    total_batches = (len(instances_to_generate) + batch_size - 1) // batch_size
    save_interval = max(1, batch_size * 10)  # Save every ~10 batches

    for batch_start in tqdm(range(0, len(instances_to_generate), batch_size),
                            total=total_batches, desc="Generating captions"):
        batch = instances_to_generate[batch_start:batch_start + batch_size]

        # Load images ON-DEMAND for this batch only
        images = []
        valid_instances = []
        for inst in batch:
            img_path = images_path / inst.gold_image
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                valid_instances.append(inst)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

        if not images:
            continue

        try:
            # Build batch inputs
            texts = []
            for img in images:
                content = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": caption_prompt}
                ]
                messages = [{"role": "user", "content": content}]
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)

            # Prepare inputs
            inputs = processor(
                text=texts,
                images=[[img] for img in images],
                return_tensors="pt",
                padding=True
            ).to(model.device)

            # Generate captions
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_captions = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            batch_captions = [cap.strip() for cap in batch_captions]

            # Store in cache
            for inst, caption in zip(valid_instances, batch_captions):
                if cache:
                    cache.set(
                        inst.target_word, inst.full_phrase, inst.gold_image,
                        caption, "caption", model_name
                    )
                key = cache._get_cache_key(inst.target_word, inst.full_phrase) if cache else f"{inst.target_word}_{inst.full_phrase}"
                captions[key] = caption

        except Exception as e:
            logger.error(f"Error processing batch at {batch_start}: {e}")
            raise
        finally:
            # ALWAYS cleanup to free memory, even on error
            # 1. Close PIL images
            for img in images:
                img.close()

            # 2. Force garbage collection before GPU cache clear
            gc.collect()

            # 3. Clear GPU cache with sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Periodic cache save (checkpoint for resume)
        if cache and (batch_start + batch_size) % save_interval == 0:
            cache.save("caption", model_name)
            logger.info(f"Checkpoint saved: {len(captions)} captions generated")

    # Cleanup model
    logger.info("Cleaning up VLM model...")
    Qwen3VLModelLoader.clear_all_cache()
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return captions


def generate_definitions_and_paraphrases(
    instances: List[TrainInstance],
    model_name: str = "qwen3-8b",
    batch_size: int = 8,
    num_paraphrase_variants: int = 3,
    cache: Optional[AugmentationCache] = None,
    resume: bool = False,
    generate_definitions: bool = True,
    generate_paraphrases: bool = True,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Generate definitions AND paraphrases using a SINGLE Qwen3 text model load.

    This avoids loading the model twice and saves memory/time.

    Args:
        instances: Training instances
        model_name: Qwen3 text model to use
        batch_size: Batch size for generation
        num_paraphrase_variants: Number of paraphrase variants per instance
        cache: Optional cache for resume support
        resume: Whether to resume from previous run
        generate_definitions: Whether to generate definitions
        generate_paraphrases: Whether to generate paraphrases

    Returns:
        Tuple of (definitions_dict, paraphrases_dict)
    """
    from src.qwen3_loader import load_qwen3_model
    from src.qwen3_inference import Qwen3DefinitionGenerator

    definitions = {}
    paraphrases = {}

    # Filter instances for each task
    def_instances = []
    para_instances = []

    if generate_definitions and resume and cache:
        for inst in instances:
            if cache.get(inst.target_word, inst.full_phrase) is None:
                def_instances.append(inst)
        logger.info(f"Definitions: {len(def_instances)} instances need generation")
    elif generate_definitions:
        def_instances = instances

    if generate_paraphrases and resume and cache:
        # Need separate cache check for paraphrases
        cache_para = AugmentationCache(cache.cache_dir)
        cache_para.load("paraphrase", model_name)
        for inst in instances:
            if cache_para.get(inst.target_word, inst.full_phrase) is None:
                para_instances.append(inst)
        logger.info(f"Paraphrases: {len(para_instances)} instances need generation")
    elif generate_paraphrases:
        para_instances = instances

    # Check if anything needs to be done
    if not def_instances and not para_instances:
        logger.info("All definitions and paraphrases already cached")
        return definitions, paraphrases

    # Load model ONCE
    logger.info(f"Loading Qwen3 text model: {model_name}")
    model, tokenizer = load_qwen3_model(model_name=model_name)

    # ===== PHASE 1: Generate Definitions =====
    if def_instances:
        logger.info(f"\n--- Generating {len(def_instances)} definitions ---")
        generator = Qwen3DefinitionGenerator(model, tokenizer)

        gen_instances = [
            {"target_word": inst.target_word, "full_phrase": inst.full_phrase}
            for inst in def_instances
        ]

        definitions_list = generator.generate_batch(
            gen_instances,
            batch_size=batch_size,
            show_progress=True
        )

        for inst, definition in zip(def_instances, definitions_list):
            if cache:
                cache.set(
                    inst.target_word, inst.full_phrase, inst.gold_image,
                    definition, "definition", model_name
                )
            key = cache._get_cache_key(inst.target_word, inst.full_phrase) if cache else f"{inst.target_word}_{inst.full_phrase}"
            definitions[key] = definition

        # Save checkpoint
        if cache:
            cache.save("definition", model_name)
            logger.info(f"Saved {len(definitions)} definitions")

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===== PHASE 2: Generate Paraphrases =====
    if para_instances:
        logger.info(f"\n--- Generating {len(para_instances)} paraphrases ---")

        PARAPHRASE_PROMPT = """Rephrase the following phrase in {num_variants} different ways while keeping the same meaning.
Keep the target word "{target_word}" in each paraphrase.
Output only the paraphrases, one per line, without numbering.

Original phrase: {full_phrase}

Paraphrases:"""

        total_batches = (len(para_instances) + batch_size - 1) // batch_size
        save_interval = max(1, batch_size * 10)

        for batch_start in tqdm(range(0, len(para_instances), batch_size),
                                total=total_batches, desc="Generating paraphrases"):
            batch = para_instances[batch_start:batch_start + batch_size]

            prompts = [
                PARAPHRASE_PROMPT.format(
                    target_word=inst.target_word,
                    full_phrase=inst.full_phrase,
                    num_variants=num_paraphrase_variants
                )
                for inst in batch
            ]

            system_prompt = "You are a helpful assistant that generates paraphrases. Keep the target word in each paraphrase."
            messages_list = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ]
                for p in prompts
            ]

            texts = [
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for msgs in messages_list
            ]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for j, (inst, output) in enumerate(zip(batch, outputs)):
                input_length = inputs['input_ids'][j].shape[0]
                generated_text = tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()

                lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                valid_paraphrases = [
                    line for line in lines
                    if inst.target_word.lower() in line.lower()
                ][:num_paraphrase_variants]

                if not valid_paraphrases:
                    valid_paraphrases = lines[:num_paraphrase_variants] if lines else [inst.full_phrase]

                if cache:
                    cache.set(
                        inst.target_word, inst.full_phrase, inst.gold_image,
                        valid_paraphrases, "paraphrase", model_name
                    )
                key = cache._get_cache_key(inst.target_word, inst.full_phrase) if cache else f"{inst.target_word}_{inst.full_phrase}"
                paraphrases[key] = valid_paraphrases

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if cache and (batch_start + batch_size) % save_interval == 0:
                cache.save("paraphrase", model_name)
                logger.info(f"Checkpoint: {len(paraphrases)} paraphrases generated")

        # Final save
        if cache:
            cache.save("paraphrase", model_name)
            logger.info(f"Saved {len(paraphrases)} paraphrases")

    # Cleanup model
    logger.info("Cleaning up Qwen3 model...")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return definitions, paraphrases


def merge_augmentations(output_dir: str) -> str:
    """
    Merge all augmentation caches into a unified index file.

    Returns path to merged file.
    """
    cache_dir = Path(output_dir)
    merged = {}

    # Find all augmentation cache files (format: train_en_{type}_{model}.json)
    for cache_file in cache_dir.glob("train_en_*.json"):
        if "index" in cache_file.name:
            continue

        # Parse: train_en_caption_qwen3-vl-8b.json -> caption
        parts = cache_file.stem.split('_')
        if len(parts) >= 3:
            aug_type = parts[2]  # e.g., "caption" from "train_en_caption_qwen3-vl-8b"
        else:
            continue

        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, entry in data.items():
            if key not in merged:
                merged[key] = {
                    "target_word": entry["target_word"],
                    "full_phrase": entry["full_phrase"],
                    "gold_image": entry.get("gold_image", ""),
                    "augmentations": {}
                }
            merged[key]["augmentations"][aug_type] = entry["augmentation"]

    # Save merged index (English only)
    output_file = cache_dir / "train_en_augmentations_index.json"
    sorted_merged = dict(sorted(
        merged.items(),
        key=lambda x: (x[1].get('target_word', ''), x[1].get('full_phrase', ''))
    ))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_merged, f, indent=2, ensure_ascii=False)

    logger.info(f"Merged {len(merged)} instances into {output_file}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM-based text augmentations for VWSD training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all augmentations
    python finetune/generate_augmentations.py --type all

    # Generate only captions
    python finetune/generate_augmentations.py --type caption --model qwen3-vl-8b

    # Generate definitions
    python finetune/generate_augmentations.py --type definition --model qwen3-8b

    # Generate paraphrases with 3 variants each
    python finetune/generate_augmentations.py --type paraphrase --num-variants 3

    # Merge existing caches into unified index
    python finetune/generate_augmentations.py --merge
        """
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["caption", "definition", "paraphrase", "all"],
        default="all",
        help="Augmentation type to generate (default: all)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: qwen3-vl-8b for caption, qwen3-8b for others)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/text_augmentations",
        help="Output directory for augmentation cache (default: results/text_augmentations)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for generation (default: 2, use 1 for limited VRAM)"
    )

    parser.add_argument(
        "--num-variants",
        type=int,
        default=3,
        help="Number of paraphrase variants per instance (default: 3)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip cached instances)"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="Only merge existing caches into unified index (no generation)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process (for testing)"
    )

    args = parser.parse_args()

    # Merge only mode
    if args.merge:
        merge_augmentations(args.output)
        return

    # Load training data
    logger.info("Loading training data...")
    instances = load_training_data(args.data_dir)

    if args.limit:
        instances = instances[:args.limit]
        logger.info(f"Limited to {len(instances)} instances for testing")

    # Initialize cache
    cache = AugmentationCache(args.output)

    def force_gpu_cleanup():
        """Force cleanup GPU memory between model loads."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Log memory status
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # ===== PHASE 1: Generate CAPTIONS using VL model =====
    if args.type in ["caption", "all"]:
        vlm_model = args.model or "qwen3-vl-8b"  # Match default in src/qwen_vlm_loader.py
        cache.load("caption", vlm_model)
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 1: Generating CAPTIONS (Qwen3-VL)...")
        logger.info(f"{'='*50}")
        generate_captions(
            instances,
            args.data_dir,
            model_name=vlm_model,
            batch_size=args.batch_size,
            cache=cache,
            resume=args.resume or args.type == "all"
        )
        cache.save("caption", vlm_model)
        # Force cleanup before loading text model
        logger.info("Cleaning up VL model before loading text model...")
        force_gpu_cleanup()

    # ===== PHASE 2: Generate DEFINITIONS and PARAPHRASES using text model =====
    # Load text model ONCE for both tasks
    need_definitions = args.type in ["definition", "all"]
    need_paraphrases = args.type in ["paraphrase", "all"]

    if need_definitions or need_paraphrases:
        text_model = args.model if args.model and "vl" not in args.model else "qwen3-8b"

        logger.info(f"\n{'='*50}")
        logger.info("PHASE 2: Generating DEFINITIONS & PARAPHRASES (Qwen3 text)...")
        logger.info(f"{'='*50}")

        # Load caches
        if need_definitions:
            cache.load("definition", text_model)
        if need_paraphrases:
            cache.load("paraphrase", text_model)

        generate_definitions_and_paraphrases(
            instances,
            model_name=text_model,
            batch_size=args.batch_size,
            num_paraphrase_variants=args.num_variants,
            cache=cache,
            resume=args.resume or args.type == "all",
            generate_definitions=need_definitions,
            generate_paraphrases=need_paraphrases,
        )

        # Force cleanup
        force_gpu_cleanup()

    # Merge all caches
    if args.type == "all":
        logger.info(f"\n{'='*50}")
        logger.info("Merging all augmentations...")
        logger.info(f"{'='*50}")
        merge_augmentations(args.output)

    logger.info("\n✓ Augmentation generation complete!")


if __name__ == "__main__":
    main()
