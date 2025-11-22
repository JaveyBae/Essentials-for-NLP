"""
VWSD Inference Engine using SigLIP2.
Embedding-based ranking for visual word sense disambiguation.

Supports two prompt strategies:
1. Default: "this is a photo of a {target_word}. {full_phrase}"
2. With definition: "this is a photo of a {target_word}. {full_phrase}. {definition}"
"""

import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prompt templates for SigLIP2
# Note: SigLIP2 was trained on simple captions, so complex prompts may hurt performance.
# The "with_definition" template appends the definition, which can introduce noise
# if definitions are inaccurate or too generic.
PROMPT_TEMPLATES = {
    "default": "this is a photo of a {target_word}. {full_phrase}",
    # Option 1: Append definition as visual clue (may add noise)
    "with_definition": "this is a photo of a {target_word}. {full_phrase}. visual clue: {definition}",
    # Option 2: Use definition directly as the main descriptor (replaces full_phrase)
    "definition_only": "this is a photo of {definition}",
    # Option 3: Combine target word with definition (no full_phrase)
    "target_definition": "this is a photo of a {target_word}. {definition}",
}


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
        instances_data: List[Dict],
        definitions: Optional[Dict[Tuple[str, str], str]] = None,
        use_definitions: bool = False,
        prompt_template: str = "with_definition"
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Rank images for one or more VWSD instances using embedding similarity.

        Strategy:
        1. Format text queries based on use_definitions flag and prompt_template:
           - default: "this is a photo of a {target_word}. {full_phrase}"
           - with_definition: "this is a photo of a {target_word}. {full_phrase}. visual clue: {definition}"
           - definition_only: "this is a photo of {definition}"
           - target_definition: "this is a photo of a {target_word}. {definition}"
        2. Process each instance: 1 text query × 10 images → compute similarities
        3. Rank by cosine similarity (via model's logits_per_image)

        Args:
            instances_data: List of dicts, each containing:
                - 'images': List[Image.Image] (candidate images, typically 10)
                - 'image_filenames': List[str]
                - 'target_word': str
                - 'full_phrase': str
            definitions: Optional dict mapping (target_word, full_phrase) -> definition string
                         Required if use_definitions=True
            use_definitions: Whether to include definitions in the text query
            prompt_template: Which template to use when use_definitions=True.
                            Options: 'with_definition', 'definition_only', 'target_definition'

        Returns:
            List of (ranked_filenames, scores) tuples, one per instance
        """
        num_instances = len(instances_data)
        prompt_type = prompt_template if use_definitions else "default"
        logger.debug(f"Processing {num_instances} instances with SigLIP2 (prompt: {prompt_type})")

        if use_definitions and definitions is None:
            raise ValueError("definitions dict is required when use_definitions=True")

        results = []

        # Process instances sequentially (true batching is complex due to different instance sizes)
        for instance_data in instances_data:
            target_word = instance_data['target_word']
            full_phrase = instance_data['full_phrase']
            images = instance_data['images']
            image_filenames = instance_data['image_filenames']

            # Format text query based on whether definitions are used
            if use_definitions:
                definition = definitions.get((target_word, full_phrase), "")
                if not definition:
                    logger.warning(f"No definition found for '{target_word}' in '{full_phrase}', using default prompt")
                    text_query = PROMPT_TEMPLATES["default"].format(
                        target_word=target_word,
                        full_phrase=full_phrase
                    )
                else:
                    # Use the selected prompt template
                    text_query = PROMPT_TEMPLATES[prompt_template].format(
                        target_word=target_word,
                        full_phrase=full_phrase,
                        definition=definition
                    )
            else:
                text_query = PROMPT_TEMPLATES["default"].format(
                    target_word=target_word,
                    full_phrase=full_phrase
                )

            # Lowercase as per SigLIP2 training
            text_query = text_query.lower()

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
    print("\nPrompt Strategies:")
    print("  1. Default: 'this is a photo of a {word}. {phrase}'")
    print("  2. With definition: 'this is a photo of a {word}. {phrase}. visual clue: {definition}'")
    print("\nInference Strategy:")
    print("  1. Format text query based on selected prompt strategy")
    print("  2. Compute image embeddings (batch all candidates)")
    print("  3. Compute text embedding")
    print("  4. Rank by sigmoid(similarity)")
    print("\nAdvantages over VLM:")
    print("  - Much faster (no generation, just embeddings)")
    print("  - Lower memory usage")
    print("  - More stable (no prompt engineering needed)")
    print("\nUsage with definitions:")
    print("  # Load definitions from cache")
    print("  from src.qwen3_inference import DefinitionCache")
    print("  cache = DefinitionCache()")
    print("  cache.load('en', 'qwen3-8b')")
    print("  definitions = {(e['target_word'], e['full_phrase']): e['definition']")
    print("                 for e in cache.cache.values()}")
    print("  ")
    print("  # Use with inference engine")
    print("  results = engine.rank_images(instances_data,")
    print("                               definitions=definitions,")
    print("                               use_definitions=True)")
    print("=" * 60)


if __name__ == "__main__":
    main()
