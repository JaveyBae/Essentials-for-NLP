"""
VWSD Augmented Dataset for SigLIP2 Fine-tuning with Text Augmentation Support.

Extends VWSDContrastiveDataset to support VLM-generated text augmentations:
- Caption: Use VLM-generated image descriptions as text queries
- Definition: Use visual definitions as context
- Paraphrase: Use paraphrased context phrases

During training, randomly samples between original text and augmented variants
to increase text diversity and prevent overfitting.
"""

import json
import hashlib
import random
from pathlib import Path
from typing import Optional, List, Dict
import logging

from finetune.vwsd_dataset import VWSDContrastiveDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VWSDAugmentedDataset(VWSDContrastiveDataset):
    """
    Dataset for contrastive fine-tuning with text augmentation.

    Each sample returns:
    - text: either original query or augmented version (based on aug_prob)
    - positive_image: the gold/correct image
    - negative_images: other candidate images (hard negatives)

    Augmentation modes:
    - "caption": Use VLM-generated image caption as context
    - "definition": Use visual definition as context
    - "paraphrase": Use paraphrased context phrase
    """

    def __init__(
        self,
        data_file: str,
        gold_file: str,
        images_dir: str,
        processor,
        augmentation_file: Optional[str] = None,
        aug_prob: float = 0.5,
        aug_types: Optional[List[str]] = None,
        num_hard_negatives: int = 9,
        image_size: int = 384,
        cache_images: bool = False,
        num_loading_threads: int = 4,
    ):
        """
        Args:
            data_file: Path to train.data.v1.txt
            gold_file: Path to train.gold.v1.txt
            images_dir: Path to train_images_v1 directory
            processor: SigLIP2 processor for text/image preprocessing
            augmentation_file: Path to augmentation cache JSON (merged index)
            aug_prob: Probability of using augmented text (default: 0.5)
            aug_types: List of augmentation types to use (default: ["caption", "definition"])
            num_hard_negatives: Number of hard negatives per positive (max 9)
            image_size: Target image size
            cache_images: If True, cache loaded images in memory
            num_loading_threads: Number of threads for parallel image loading
        """
        super().__init__(
            data_file=data_file,
            gold_file=gold_file,
            images_dir=images_dir,
            processor=processor,
            num_hard_negatives=num_hard_negatives,
            image_size=image_size,
            cache_images=cache_images,
            num_loading_threads=num_loading_threads,
        )

        self.aug_prob = aug_prob
        self.aug_types = aug_types or ["caption", "definition"]
        self.augmentations = {}

        # Load augmentation cache
        if augmentation_file and Path(augmentation_file).exists():
            self.augmentations = self._load_augmentations(augmentation_file)
            logger.info(f"Loaded augmentations for {len(self.augmentations)} instances")
            logger.info(f"Augmentation probability: {aug_prob:.0%}")
            logger.info(f"Augmentation types: {self.aug_types}")

            # Report coverage
            self._report_augmentation_coverage()
        else:
            logger.warning(f"No augmentation file found at {augmentation_file}")
            logger.warning("Running without text augmentation")

    def _load_augmentations(self, path: str) -> Dict:
        """Load augmentation cache from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both formats: direct dict or with 'instances' key
        if 'instances' in data:
            return data['instances']
        return data

    def _get_cache_key(self, target_word: str, full_phrase: str) -> str:
        """Generate cache key matching augmentation generator."""
        phrase_hash = hashlib.md5(full_phrase.encode('utf-8')).hexdigest()[:8]
        return f"{target_word}_{phrase_hash}"

    def _report_augmentation_coverage(self):
        """Report augmentation coverage statistics."""
        total = len(self.instances)
        covered = 0
        aug_counts = {aug_type: 0 for aug_type in self.aug_types}

        for instance in self.instances:
            key = self._get_cache_key(instance['target_word'], instance['full_phrase'])
            if key in self.augmentations:
                covered += 1
                aug_data = self.augmentations[key].get('augmentations', {})
                for aug_type in self.aug_types:
                    if aug_type in aug_data:
                        aug_counts[aug_type] += 1

        logger.info(f"Augmentation coverage: {covered}/{total} instances ({covered/total:.1%})")
        for aug_type, count in aug_counts.items():
            logger.info(f"  - {aug_type}: {count}/{total} ({count/total:.1%})")

    def _sample_augmented_text(self, target_word: str, full_phrase: str) -> str:
        """Sample an augmented text query for this instance."""
        key = self._get_cache_key(target_word, full_phrase)

        if key not in self.augmentations:
            # No augmentation available, return original
            return f"this is a photo of a {target_word}. {full_phrase}".lower()

        aug_data = self.augmentations[key].get('augmentations', {})

        # Collect available augmented texts
        candidates = []

        if 'caption' in self.aug_types and 'caption' in aug_data:
            caption = aug_data['caption']
            if caption:
                # Caption-based: use caption as context description
                candidates.append(
                    f"this is a photo of a {target_word}. {caption}".lower()
                )

        if 'definition' in self.aug_types and 'definition' in aug_data:
            definition = aug_data['definition']
            if definition:
                # Definition-based: use visual definition as context
                candidates.append(
                    f"this is a photo of a {target_word}. {definition}".lower()
                )

        if 'paraphrase' in self.aug_types and 'paraphrase' in aug_data:
            paraphrases = aug_data['paraphrase']
            if isinstance(paraphrases, list):
                for para in paraphrases:
                    if para:
                        candidates.append(
                            f"this is a photo of a {target_word}. {para}".lower()
                        )
            elif paraphrases:
                candidates.append(
                    f"this is a photo of a {target_word}. {paraphrases}".lower()
                )

        if not candidates:
            # No valid augmentations, return original
            return f"this is a photo of a {target_word}. {full_phrase}".lower()

        return random.choice(candidates)

    def __getitem__(self, idx):
        """
        Get item with optional text augmentation.

        Returns a dict with:
        - text: either original or augmented query text
        - positive_image: PIL Image of correct sense
        - negative_images: list of PIL Images (hard negatives)
        - target_word: the ambiguous target word
        - full_phrase: original context phrase
        """
        instance = self.instances[idx]

        # Decide whether to use augmented text
        if random.random() < self.aug_prob and self.augmentations:
            text = self._sample_augmented_text(
                instance['target_word'],
                instance['full_phrase']
            )
        else:
            # Original text
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


if __name__ == "__main__":
    # Test the augmented dataset
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

    train_dir = "data/semeval-2023-task-1-V-WSD-train-v1/train_v1"
    aug_file = "results/text_augmentations/train_augmentations_index.json"

    # Test without augmentation
    print("Testing without augmentation:")
    dataset_no_aug = VWSDAugmentedDataset(
        data_file=f"{train_dir}/train.data.v1.txt",
        gold_file=f"{train_dir}/train.gold.v1.txt",
        images_dir=f"{train_dir}/train_images_v1",
        processor=processor,
        aug_prob=0.0,
    )
    print(f"Dataset size: {len(dataset_no_aug)}")
    sample = dataset_no_aug[0]
    print(f"Sample text (no aug): {sample['text']}")

    # Test with augmentation (if cache exists)
    if Path(aug_file).exists():
        print("\nTesting with augmentation:")
        dataset_aug = VWSDAugmentedDataset(
            data_file=f"{train_dir}/train.data.v1.txt",
            gold_file=f"{train_dir}/train.gold.v1.txt",
            images_dir=f"{train_dir}/train_images_v1",
            processor=processor,
            augmentation_file=aug_file,
            aug_prob=1.0,  # Always use augmented for testing
            aug_types=["caption", "definition", "paraphrase"],
        )
        sample = dataset_aug[0]
        print(f"Sample text (augmented): {sample['text']}")

        # Show multiple samples to see variety
        print("\nSampling 5 augmented texts for same instance:")
        for i in range(5):
            sample = dataset_aug[0]
            print(f"  {i+1}. {sample['text'][:80]}...")
    else:
        print(f"\nAugmentation file not found at {aug_file}")
        print("Run: python finetune/generate_augmentations.py --type all")
