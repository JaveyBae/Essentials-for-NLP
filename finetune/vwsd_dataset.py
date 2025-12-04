"""
VWSD Dataset for SigLIP2 Fine-tuning

Loads training data in contrastive format:
- Each instance: (text_query, positive_image, hard_negative_images)
- Hard negatives: other candidate images for the same ambiguous word

Optimized for GPU utilization:
- Parallel image loading with ThreadPoolExecutor
- Optional image caching to avoid re-reading from disk
- Prefetching support via increased num_workers
"""

import os
from pathlib import Path
from typing import Optional, Dict
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import random
from functools import lru_cache

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Increase limit for large images (some training images are very large)
Image.MAX_IMAGE_PIXELS = None


class VWSDContrastiveDataset(Dataset):
    """
    Dataset for contrastive fine-tuning on VWSD.

    Each sample returns:
    - text: "this is a photo of a {target_word}. {full_phrase}"
    - positive_image: the gold/correct image
    - negative_images: other candidate images (hard negatives from same instance)

    Optimizations:
    - Parallel image loading with ThreadPoolExecutor (6-10 images per sample)
    - Optional in-memory caching for frequently accessed images
    - Pre-computed image paths to avoid repeated Path operations
    """

    def __init__(
        self,
        data_file: str,
        gold_file: str,
        images_dir: str,
        processor,
        num_hard_negatives: int = 9,  # Use all 9 other candidates as hard negatives
        image_size: int = 384,
        cache_images: bool = False,  # Set True if you have enough RAM (~10GB for full dataset)
        num_loading_threads: int = 4,  # Threads for parallel image loading within a sample
    ):
        """
        Args:
            data_file: Path to train.data.v1.txt
            gold_file: Path to train.gold.v1.txt
            images_dir: Path to train_images_v1 directory
            processor: SigLIP2 processor for text/image preprocessing
            num_hard_negatives: Number of hard negatives per positive (max 9)
            image_size: Target image size
            cache_images: If True, cache loaded images in memory (faster but uses more RAM)
            num_loading_threads: Number of threads for parallel image loading
        """
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.num_hard_negatives = min(num_hard_negatives, 9)
        self.image_size = image_size
        self.cache_images = cache_images
        self.num_loading_threads = num_loading_threads

        # Image cache (only used if cache_images=True)
        self._image_cache: Dict[str, Image.Image] = {}

        # Thread pool for parallel image loading
        self._executor = ThreadPoolExecutor(max_workers=num_loading_threads)

        # Load data
        self.instances = []
        with open(data_file, 'r') as f_data, open(gold_file, 'r') as f_gold:
            for data_line, gold_line in zip(f_data, f_gold):
                parts = data_line.strip().split('\t')
                gold_image = gold_line.strip()

                target_word = parts[0]
                full_phrase = parts[1]
                candidate_images = parts[2:12]  # 10 candidates

                # Separate positive and negatives
                if gold_image in candidate_images:
                    negative_images = [img for img in candidate_images if img != gold_image]
                else:
                    # Gold not in candidates (shouldn't happen, but handle it)
                    negative_images = candidate_images[:9]

                self.instances.append({
                    'target_word': target_word,
                    'full_phrase': full_phrase,
                    'positive_image': gold_image,
                    'negative_images': negative_images,
                })

        print(f"Loaded {len(self.instances)} training instances")
        print(f"Image caching: {'enabled' if cache_images else 'disabled'}")
        print(f"Parallel loading threads: {num_loading_threads}")

    def __len__(self):
        return len(self.instances)

    def _load_image(self, image_name: str) -> Image.Image:
        """Load and preprocess a single image with optional caching."""
        # Check cache first
        if self.cache_images and image_name in self._image_cache:
            return self._image_cache[image_name].copy()

        image_path = self.images_dir / image_name
        try:
            img = Image.open(image_path).convert('RGB')
            # Cache if enabled
            if self.cache_images:
                self._image_cache[image_name] = img.copy()
            return img
        except Exception as e:
            print(f"Warning: Error loading {image_path}: {e}, using black placeholder")
            # Return a black placeholder image instead of None
            return Image.new('RGB', (self.image_size, self.image_size), color='black')

    def _load_images_parallel(self, image_names: list) -> list:
        """Load multiple images in parallel using ThreadPoolExecutor."""
        if len(image_names) <= 1:
            return [self._load_image(name) for name in image_names]

        # Submit all image loading tasks
        futures = [self._executor.submit(self._load_image, name) for name in image_names]
        # Collect results in order
        return [f.result() for f in futures]

    def __getitem__(self, idx):
        """
        Returns a dict with:
        - text: the query text
        - positive_image: PIL Image of correct sense
        - negative_images: list of PIL Images (hard negatives)
        """
        instance = self.instances[idx]

        # Format text query (same as inference) - lowercase for SigLIP2
        text = f"this is a photo of a {instance['target_word']}. {instance['full_phrase']}".lower()

        # Collect all images to load
        negatives_to_use = instance['negative_images'][:self.num_hard_negatives]
        all_image_names = [instance['positive_image']] + negatives_to_use

        # Load all images in parallel
        all_images = self._load_images_parallel(all_image_names)

        positive_image = all_images[0]
        negative_images = all_images[1:]

        return {
            'text': text,
            'positive_image': positive_image,
            'negative_images': negative_images,
            'target_word': instance['target_word'],
            'full_phrase': instance['full_phrase'],
        }

    def clear_cache(self):
        """Clear the image cache to free memory."""
        self._image_cache.clear()

    def __del__(self):
        """Clean up thread pool on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


def collate_vwsd_contrastive(batch, processor):
    """
    Custom collate function for contrastive training.

    Returns:
    - input_ids, attention_mask: tokenized texts [batch_size]
    - pixel_values_pos: positive images [batch_size, 3, H, W]
    - pixel_values_neg: negative images [batch_size * num_neg, 3, H, W]
    - num_negatives: list of negative counts per sample
    """
    texts = [item['text'] for item in batch]
    positive_images = [item['positive_image'] for item in batch]

    # Flatten all negatives
    all_negative_images = []
    num_negatives = []
    for item in batch:
        negs = item['negative_images']
        all_negative_images.extend(negs)
        num_negatives.append(len(negs))

    # Process texts
    text_inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    # Process positive images
    pos_inputs = processor(
        images=positive_images,
        return_tensors='pt'
    )

    # Process negative images (if any)
    if all_negative_images:
        neg_inputs = processor(
            images=all_negative_images,
            return_tensors='pt'
        )
        pixel_values_neg = neg_inputs['pixel_values']
    else:
        pixel_values_neg = None

    return {
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'pixel_values_pos': pos_inputs['pixel_values'],
        'pixel_values_neg': pixel_values_neg,
        'num_negatives': num_negatives,
    }


class VWSDPairDataset(Dataset):
    """
    Simpler dataset: each sample is one (text, image, is_positive) pair.
    Used for standard batch contrastive learning.

    For each instance, creates:
    - 1 positive pair: (text, gold_image, 1)
    - N negative pairs: (text, wrong_image, 0)
    """

    def __init__(
        self,
        data_file: str,
        gold_file: str,
        images_dir: str,
        processor,
        include_negatives: bool = True,
        num_negatives_per_positive: int = 1,  # For balanced training
    ):
        self.processor = processor
        self.images_dir = Path(images_dir)
        self.include_negatives = include_negatives
        self.num_negatives = num_negatives_per_positive

        # Load all instances
        self.pairs = []
        with open(data_file, 'r') as f_data, open(gold_file, 'r') as f_gold:
            for data_line, gold_line in zip(f_data, f_gold):
                parts = data_line.strip().split('\t')
                gold_image = gold_line.strip()

                target_word = parts[0]
                full_phrase = parts[1]
                candidate_images = parts[2:12]

                text = f"this is a photo of a {target_word}. {full_phrase}".lower()

                # Add positive pair
                self.pairs.append({
                    'text': text,
                    'image': gold_image,
                    'label': 1,
                })

                # Add negative pairs (sample from wrong candidates)
                if include_negatives:
                    negatives = [img for img in candidate_images if img != gold_image]
                    sampled_negs = random.sample(negatives, min(self.num_negatives, len(negatives)))
                    for neg_img in sampled_negs:
                        self.pairs.append({
                            'text': text,
                            'image': neg_img,
                            'label': 0,
                        })

        print(f"Created {len(self.pairs)} text-image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        image_path = self.images_dir / pair['image']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Return a black image on error
            image = Image.new('RGB', (384, 384), color='black')

        return {
            'text': pair['text'],
            'image': image,
            'label': pair['label'],
        }


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

    train_dir = "data/semeval-2023-task-1-V-WSD-train-v1/train_v1"
    dataset = VWSDContrastiveDataset(
        data_file=f"{train_dir}/train.data.v1.txt",
        gold_file=f"{train_dir}/train.gold.v1.txt",
        images_dir=f"{train_dir}/train_images_v1",
        processor=processor,
    )

    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample text: {sample['text']}")
    print(f"Positive image: {sample['positive_image']}")
    print(f"Num negatives: {len(sample['negative_images'])}")
