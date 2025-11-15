"""
VWSD Inference Engine using SigLIP2.
Embedding-based ranking for visual word sense disambiguation.
"""

import torch
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of VWSD inference."""
    instance_id: int
    ranked_images: List[str]  # Ranked list of image filenames
    scores: List[float]  # Corresponding similarity scores
    target_word: str = None  # The ambiguous target word
    full_phrase: str = None  # Limited textual context containing the target_word
    gold_image: str = None  # Gold label (if available)


class SigLIP2Inference:
    """VWSD inference using SigLIP2 embedding similarity."""

    def __init__(self, model, processor, device="cuda"):
        """
        Initialize inference engine.

        Args:
            model: Loaded SigLIP2 model
            processor: Loaded SigLIP2 processor
            device: Device to run inference on
        """
        self.model = model
        self.processor = processor
        self.device = device

    def rank_images(
        self,
        instances_data: List[Dict]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Rank images for one or more VWSD instances using embedding similarity.

        Strategy:
        1. Format text queries: "this is a photo of a {target_word}. {full_phrase}" for each instance
        2. Process each instance: 1 text query × 10 images → compute similarities
        3. Rank by cosine similarity (via model's logits_per_image)

        Args:
            instances_data: List of dicts, each containing:
                - 'images': List[Image.Image] (candidate images, typically 10)
                - 'image_filenames': List[str]
                - 'target_word': str
                - 'full_phrase': str

        Returns:
            List of (ranked_filenames, scores) tuples, one per instance
        """
        num_instances = len(instances_data)
        logger.debug(f"Processing {num_instances} instances with SigLIP2")

        results = []

        # Process instances sequentially (true batching is complex due to different instance sizes)
        for instance_data in instances_data:
            target_word = instance_data['target_word']
            full_phrase = instance_data['full_phrase']
            images = instance_data['images']
            image_filenames = instance_data['image_filenames']

            # Format text query (lowercase as per SigLIP2 training)
            text_query = f"this is a photo of a {target_word}. {full_phrase}".lower()

            # Prepare inputs: single text, multiple images
            inputs = self.processor(
                text=[text_query],
                images=images,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            with torch.inference_mode():
                outputs = self.model(**inputs)

            # Get similarity scores
            logits_per_image = outputs.logits_per_image.squeeze(-1)  # Shape: (num_images,)
            probs = torch.sigmoid(logits_per_image)
            scores = probs.cpu().tolist()

            # Rank by scores (highest first)
            ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            ranked_filenames = [image_filenames[idx] for idx in ranked_indices]
            ranked_scores = [scores[idx] for idx in ranked_indices]

            results.append((ranked_filenames, ranked_scores))

            # Cleanup
            del inputs, outputs, logits_per_image, probs

        # Explicit memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


def main():
    """Test SigLIP2 inference engine."""
    print("=" * 60)
    print("SigLIP2 Inference Engine")
    print("=" * 60)
    print("\nStrategy:")
    print("  1. Format text: 'this is a photo of a {word}. {phrase}'")
    print("  2. Compute image embeddings (batch all candidates)")
    print("  3. Compute text embedding")
    print("  4. Rank by sigmoid(similarity)")
    print("\nAdvantages over VLM:")
    print("  - Much faster (no generation, just embeddings)")
    print("  - Lower memory usage")
    print("  - More stable (no prompt engineering needed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
