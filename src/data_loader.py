"""
Data loader for Visual Word Sense Disambiguation (VWSD) task.
Loads test data and candidate images for evaluation.

Optimizations:
- Parallel image loading with ThreadPoolExecutor
- LRU cache for frequently accessed images
- Batch loading support
- Progress tracking with tqdm
"""

import os
from typing import List, Dict, Tuple, Optional
from PIL import Image
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class VWSDInstance:
    """Represents a single VWSD test instance."""
    instance_id: int
    target_word: str
    full_phrase: str  # Limited textual context containing the target_word
    candidate_images: List[str]  # List of 10 image filenames
    gold_image: str = None  # Gold label (if available)


class VWSDDataLoader:
    """Loads and manages VWSD test data with optimized parallel loading."""

    def __init__(self, data_dir: str = "data", num_workers: int = 8, use_cache: bool = True):
        """
        Initialize data loader.

        Args:
            data_dir: Root directory containing test_data/ and test_images/ folders
            num_workers: Number of parallel workers for image loading (default: 8)
            use_cache: Whether to cache loaded images (default: True)
        """
        self.data_dir = data_dir
        self.test_data_dir = os.path.join(data_dir, "test_data")
        self.test_images_dir = os.path.join(data_dir, "test_images", "test_images_resized")
        self.num_workers = num_workers
        self.use_cache = use_cache

        # Verify directories exist
        if not os.path.exists(self.test_data_dir):
            raise FileNotFoundError(f"Test data directory not found: {self.test_data_dir}")
        if not os.path.exists(self.test_images_dir):
            raise FileNotFoundError(f"Test images directory not found: {self.test_images_dir}")

    def load_test_data(self, language: str = "en") -> List[VWSDInstance]:
        """
        Load test data for a specific language.

        Args:
            language: Language code ('en', 'fa', 'it')

        Returns:
            List of VWSDInstance objects
        """
        # Load test data file
        data_file = os.path.join(self.test_data_dir, f"{language}.test.data.v1.1.txt")
        gold_file = os.path.join(self.test_data_dir, f"{language}.test.gold.v1.1.txt")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Test data file not found: {data_file}")

        instances = []

        # Read data file
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Read gold labels if available
        gold_labels = []
        if os.path.exists(gold_file):
            with open(gold_file, 'r', encoding='utf-8') as f:
                gold_labels = [line.strip() for line in f if line.strip()]

        # Parse each line
        for idx, line in enumerate(lines):
            parts = line.split('\t')
            if len(parts) < 12:  # Should have target_word + context + 10 images
                print(f"Warning: Line {idx+1} has {len(parts)} parts, expected 12+")
                continue

            target_word = parts[0]
            full_phrase = parts[1]
            candidate_images = parts[2:12]  # Next 10 are image filenames

            gold_image = gold_labels[idx] if idx < len(gold_labels) else None

            instance = VWSDInstance(
                instance_id=idx,
                target_word=target_word,
                full_phrase=full_phrase,
                candidate_images=candidate_images,
                gold_image=gold_image
            )
            instances.append(instance)

        print(f"Loaded {len(instances)} {language} test instances")
        return instances

    def _load_single_image(self, image_filename: str) -> Tuple[str, Optional[Image.Image], Optional[str]]:
        """
        Internal method to load a single image (used by parallel loader).

        Args:
            image_filename: Name of the image file

        Returns:
            Tuple of (filename, image, error_message)
        """
        try:
            image_path = os.path.join(self.test_images_dir, image_filename)
            if not os.path.exists(image_path):
                return image_filename, None, f"Image not found: {image_path}"

            img = Image.open(image_path).convert('RGB')
            return image_filename, img, None
        except Exception as e:
            return image_filename, None, str(e)

    def load_image(self, image_filename: str) -> Image.Image:
        """
        Load a single image by filename.

        Args:
            image_filename: Name of the image file

        Returns:
            PIL Image object
        """
        if self.use_cache:
            return self._load_image_cached(image_filename)

        image_path = os.path.join(self.test_images_dir, image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        return Image.open(image_path).convert('RGB')

    @lru_cache(maxsize=512)
    def _load_image_cached(self, image_filename: str) -> Image.Image:
        """
        Load and cache a single image (LRU cache for frequently accessed images).

        Args:
            image_filename: Name of the image file

        Returns:
            PIL Image object
        """
        image_path = os.path.join(self.test_images_dir, image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        return Image.open(image_path).convert('RGB')

    def load_instance_images(self, instance: VWSDInstance, parallel: bool = True) -> List[Image.Image]:
        """
        Load all 10 candidate images for an instance (with optional parallel loading).

        Args:
            instance: VWSDInstance object
            parallel: Whether to use parallel loading (default: True)

        Returns:
            List of 10 PIL Image objects
        """
        if not parallel or self.num_workers <= 1:
            # Sequential loading
            images = []
            for img_filename in instance.candidate_images:
                try:
                    img = self.load_image(img_filename)
                    images.append(img)
                except FileNotFoundError as e:
                    logger.warning(f"Warning: {e}")
                    # Create a blank placeholder image
                    images.append(Image.new('RGB', (224, 224), color='gray'))
            return images

        # Parallel loading
        return self.load_images_parallel(instance.candidate_images)

    def load_images_parallel(self, image_filenames: List[str], show_progress: bool = False) -> List[Image.Image]:
        """
        Load multiple images in parallel.

        Args:
            image_filenames: List of image filenames to load
            show_progress: Whether to show progress bar (default: False)

        Returns:
            List of PIL Image objects (in same order as input)
        """
        # Create a mapping to preserve order
        filename_to_index = {filename: idx for idx, filename in enumerate(image_filenames)}
        results = [None] * len(image_filenames)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self._load_single_image, filename): filename
                      for filename in image_filenames}

            # Collect results with optional progress bar
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Loading images")

            for future in iterator:
                filename, img, error = future.result()
                idx = filename_to_index[filename]

                if error:
                    logger.warning(f"Warning: {error}")
                    # Create a blank placeholder image
                    results[idx] = Image.new('RGB', (224, 224), color='gray')
                else:
                    results[idx] = img

        return results

    def load_batch_images(self, instances: List[VWSDInstance], show_progress: bool = True) -> List[List[Image.Image]]:
        """
        Load images for multiple instances in parallel (batch loading).

        Args:
            instances: List of VWSDInstance objects
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of image lists (one per instance)
        """
        # Collect all unique image filenames
        all_filenames = []
        instance_image_map = []  # Map instances to their image indices

        for instance in instances:
            instance_indices = []
            for filename in instance.candidate_images:
                if filename not in all_filenames:
                    all_filenames.append(filename)
                instance_indices.append(all_filenames.index(filename))
            instance_image_map.append(instance_indices)

        # Load all unique images in parallel
        logger.info(f"Loading {len(all_filenames)} unique images for {len(instances)} instances...")
        all_images = self.load_images_parallel(all_filenames, show_progress=show_progress)

        # Map back to instances
        batch_results = []
        for instance_indices in instance_image_map:
            instance_images = [all_images[idx] for idx in instance_indices]
            batch_results.append(instance_images)

        return batch_results

    def get_image_path(self, image_filename: str) -> str:
        """Get full path to an image file."""
        return os.path.join(self.test_images_dir, image_filename)

    def clear_cache(self):
        """Clear the image cache."""
        if self.use_cache:
            self._load_image_cached.cache_clear()
            logger.info("Image cache cleared")


def main():
    """Test the data loader with parallel loading."""
    import time

    # Test with different configurations
    print("=" * 60)
    print("Testing VWSDDataLoader with optimizations")
    print("=" * 60)

    # Sequential loading
    print("\n1. Sequential loading (num_workers=1):")
    loader_seq = VWSDDataLoader(num_workers=1)
    instances = loader_seq.load_test_data("en")
    print(f"  Total instances: {len(instances)}")

    first = instances[0]
    print(f"\n  First instance:")
    print(f"    ID: {first.instance_id}")
    print(f"    Target word: {first.target_word}")
    print(f"    Full phrase: {first.full_phrase}")
    print(f"    Candidate images: {first.candidate_images[:3]}...")

    start = time.time()
    images_seq = loader_seq.load_instance_images(first, parallel=False)
    time_seq = time.time() - start
    print(f"\n  Sequential loading time: {time_seq:.3f}s")
    print(f"  Loaded {len(images_seq)} images")
    print(f"  First image size: {images_seq[0].size}")

    # Parallel loading
    print("\n2. Parallel loading (num_workers=4):")
    loader_par = VWSDDataLoader(num_workers=4)
    loader_par.load_test_data("en")

    start = time.time()
    images_par = loader_par.load_instance_images(first, parallel=True)
    time_par = time.time() - start
    print(f"  Parallel loading time: {time_par:.3f}s")
    print(f"  Loaded {len(images_par)} images")
    print(f"  Speedup: {time_seq/time_par:.2f}x")

    # Batch loading
    print("\n3. Batch loading (first 5 instances):")
    start = time.time()
    batch_images = loader_par.load_batch_images(instances[:5], show_progress=True)
    time_batch = time.time() - start
    print(f"  Batch loading time: {time_batch:.3f}s")
    print(f"  Loaded images for {len(batch_images)} instances")
    print(f"  Total images loaded: {sum(len(imgs) for imgs in batch_images)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
