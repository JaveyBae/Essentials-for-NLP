"""
VWSD Inference Engine using Qwen-VL models.
Implements single-image evaluation strategy for visual word sense disambiguation.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Single-image Prompt Templates ==========
# Process one image at a time to avoid attention dilution

PROMPT_TEMPLATES = {
    "matching": '''Task: Given a word and textual context, rate how well this image matches the intended meaning.

Target word: "{target_word}"
Full phrase: "{full_phrase}"

The word "{target_word}" can have multiple meanings.
Rate how well this image matches the SPECIFIC meaning of "{target_word}" as used in the phrase "{full_phrase}".

Rating scale:
0 = completely unrelated or shows a different meaning
5 = somewhat related but not the exact meaning
10 = perfect match for the intended meaning

Provide only a single number (0-10) as your rating.''',

    "matching_cot": '''Task: Rate how well this image matches the SPECIFIC meaning of a word in context.

Target word: "{target_word}"
Full phrase: "{full_phrase}"

Think step-by-step:
1. What does "{target_word}" mean in "{full_phrase}"?
2. What does the image show?
3. Do they match?

Rating scale:
0-3 = Wrong meaning or unrelated
4-6 = Partially related
7-10 = Clear match

Format your answer as:
[1-2 sentence reasoning]
Rating: X

Example:
The phrase means "financial institution." The image shows a riverbank. Different meanings.
Rating: 0''',

    "description": '''Task: Describe what you see in this image, focusing on aspects relevant to the word "{target_word}".

Target word: "{target_word}"
Full phrase: "{full_phrase}"

Describe the main subject or concept shown in this image (8-15 words).
Focus on elements that could relate to different meanings of "{target_word}".'''
}

MAX_TOKENS = {
    "matching": 50,          # Only need one number
    "matching_cot": 300,     # Reasoning + final rating (increased to prevent truncation)
    "description": 100       # Short description
}


@dataclass
class InferenceResult:
    """Result of VWSD inference."""
    instance_id: int
    ranked_images: List[str]  # Ranked list of image filenames
    scores: List[float]  # Corresponding scores
    target_word: str = None  # The ambiguous target word
    full_phrase: str = None  # Limited textual context containing the target_word
    gold_image: str = None  # Gold label (if available)


class QwenVLInference:
    """VWSD inference using Qwen-VL models."""

    def __init__(self, model, processor, device="cuda"):
        """
        Initialize inference engine.

        Args:
            model: Loaded Qwen-VL model
            processor: Loaded Qwen-VL processor
            device: Device to run inference on
        """
        self.model = model
        self.processor = processor
        self.device = device

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into contextualized embeddings using Qwen's text encoder.

        Args:
            text: Input text string

        Returns:
            Mean-pooled embedding vector of shape (hidden_size,) e.g., (2048,)
        """
        # Tokenize the text
        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        with torch.inference_mode():
            # Get input embeddings from the language model
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

            # Forward through language model to get contextualized representations
            outputs = self.model.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Get last hidden state: (1, seq_len, hidden_size)
            hidden_states = outputs.last_hidden_state

            # Mean pooling over sequence dimension
            if attention_mask is not None:
                # Weighted average based on attention mask (ignore padding)
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask  # Shape: (1, hidden_size)
            else:
                # Simple average
                pooled = hidden_states.mean(dim=1)  # Shape: (1, hidden_size)

        return pooled.squeeze(0)  # Shape: (hidden_size,)

    def rank_images(self,
                    instances_data: List[Dict],
                    method: str = "matching",
                    batch_size: int = 1) -> List[Tuple[List[str], List[float]]]:
        """
        Rank images for one or more VWSD instances.

        Strategy: For each instance, all ~10 candidate images are batched together
        and evaluated in a single forward pass for efficiency.

        Args:
            instances_data: List of dicts, each containing:
                - 'images': List[Image.Image] (candidate images, typically 10)
                - 'image_filenames': List[str]
                - 'target_word': str
                - 'full_phrase': str
            method: Scoring method ('matching' or 'description')
            batch_size: Number of instances to process in parallel (default: 1)
                       Higher values use more GPU memory but may be faster

        Returns:
            List of (ranked_filenames, scores) tuples, one per instance
        """
        if method not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown method: {method}. Must be one of {list(PROMPT_TEMPLATES.keys())}")

        num_instances = len(instances_data)
        logger.debug(f"Processing {num_instances} instances (batch_size={batch_size}, method={method})")

        results = []

        # Cache for phrase embeddings when using description method
        phrase_embedding_cache = {}

        # Process instances in batches
        for batch_start in range(0, num_instances, batch_size):
            batch_end = min(batch_start + batch_size, num_instances)
            batch_instances = instances_data[batch_start:batch_end]

            # Process each instance in the batch
            for instance_idx, instance_data in enumerate(batch_instances):
                global_idx = batch_start + instance_idx

                images = instance_data['images']
                image_filenames = instance_data['image_filenames']
                target_word = instance_data['target_word']
                full_phrase = instance_data['full_phrase']
                num_images = len(images)

                logger.debug(f"Instance {global_idx+1}/{num_instances}: Evaluating {num_images} images")

                # Format prompt (same for all images in this instance)
                prompt = PROMPT_TEMPLATES[method].format(
                    target_word=target_word,
                    full_phrase=full_phrase
                )

                # Cache phrase embedding for description method
                if method == "description" and full_phrase not in phrase_embedding_cache:
                    with torch.inference_mode():
                        phrase_embedding_cache[full_phrase] = self.encode_text(full_phrase)

                # Stream processing: build texts and process images in batches
                scores = []

                # Process all images for this instance in one batch
                texts = []
                for img in images:
                    content = [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                    messages = [{"role": "user", "content": content}]
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    texts.append(text)

                # Prepare batch inputs
                inputs = self.processor(
                    text=texts,
                    images=[[img] for img in images],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                # Generate scores for all candidate images at once
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_TOKENS[method],
                        do_sample=False,
                        temperature=1.0,
                        top_p=None,
                        top_k=None,
                    )

                # Decode responses
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                # Parse each response to get score
                for img_idx, output_text in enumerate(output_texts):
                    output_text = output_text.strip()

                    if method in ["matching", "matching_cot"]:
                        score = self._parse_single_image_score(output_text)
                    elif method == "description":
                        score = self._score_description(
                            output_text,
                            full_phrase,
                            target_word,
                            phrase_embedding_cache.get(full_phrase)
                        )

                    scores.append(score)
                    logger.debug(f"  Image {img_idx+1}/{num_images}: score={score:.2f}, response='{output_text[:50]}...')")

                # Rank by scores (highest first)
                ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
                ranked_filenames = [image_filenames[idx] for idx in ranked_indices]
                ranked_scores = [scores[idx] for idx in ranked_indices]

                results.append((ranked_filenames, ranked_scores))
                logger.debug(f"Instance {global_idx+1} top-3: {ranked_filenames[:3]} with scores {ranked_scores[:3]}")

                # Explicit memory cleanup
                del inputs, generated_ids, generated_ids_trimmed, output_texts, texts, scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Clear phrase embedding cache
        phrase_embedding_cache.clear()

        return results

    def _parse_single_image_score(self, response: str) -> float:
        """
        Parse model response and extract a single numerical score (0-10).

        Handles both simple format (just a number) and CoT format (reasoning + "Rating: X").

        Args:
            response: Model's response text (should contain a number 0-10)

        Returns:
            Score normalized to [0.0, 1.0] range
        """
        import re

        # First, check if response contains "Rating:" pattern (CoT format)
        rating_match = re.search(r'[Rr]ating\s*:\s*(\d+)', response)

        if rating_match:
            # Extract number after "Rating:"
            rating = int(rating_match.group(1))
        else:
            # Fall back: extract all numbers and take the LAST one (likely the final rating)
            # This handles cases where the model puts the rating at the end without "Rating:" prefix
            numbers = re.findall(r'\b(\d+)\b', response)

            # Filter to only valid ratings (0-10)
            valid_numbers = [int(n) for n in numbers if 0 <= int(n) <= 10]

            if valid_numbers:
                # Take the last valid number (most likely the final rating)
                rating = valid_numbers[-1]
                logger.debug(f"Extracted rating {rating} from response (no 'Rating:' prefix found)")
            else:
                # No valid rating found - check if model refused/explained
                if len(response) > 100:
                    # Long response without rating - model likely explained instead of rating
                    logger.warning(f"Model provided explanation without rating. Response: '{response[:100]}...'. Using default 0.0 (unrelated)")
                    return 0.0  # Default to unrelated rather than middle score
                else:
                    logger.warning(f"Could not parse score from response: '{response}'. Using default 0.5")
                    return 0.5

        # Clamp to [0, 10] and normalize to [0.0, 1.0]
        rating = max(0, min(rating, 10))
        score = rating / 10.0

        return score

    def _score_description(self, description: str, full_phrase: str, target_word: str,
                          phrase_embedding: torch.Tensor = None) -> float:
        """
        Score an image description based on semantic similarity with context.

        Uses Qwen's text encoder to compute embeddings and cosine similarity
        between the description and the full phrase.

        Args:
            description: Model's description of the image
            full_phrase: Full phrase containing target word
            target_word: The ambiguous target word
            phrase_embedding: Cached embedding for full_phrase (optional, computed if not provided)

        Returns:
            Score based on semantic similarity (0.0 to 1.0)
        """
        # Use cached phrase embedding if provided, otherwise compute it
        if phrase_embedding is None:
            with torch.inference_mode():
                phrase_embedding = self.encode_text(full_phrase)

        # Encode description
        with torch.inference_mode():
            desc_embedding = self.encode_text(description)

        # Compute cosine similarity
        # F.cosine_similarity expects inputs of shape (1, D) or (N, D)
        similarity = F.cosine_similarity(
            phrase_embedding.unsqueeze(0),
            desc_embedding.unsqueeze(0)
        ).item()

        # Cosine similarity ranges from -1 to 1, normalize to [0, 1]
        score = (similarity + 1.0) / 2.0

        # Optional: Small bonus if target word appears in description
        # (indicates the description is focused on the target concept)
        target_lower = target_word.lower()
        description_lower = description.lower()
        if target_lower in description_lower:
            score = min(score + 0.1, 1.0)

        # Ensure score is in valid range
        return max(0.0, min(score, 1.0))


def main():
    """Test inference engine."""
    print("Inference engine ready!")
    print("\nMethods:")
    print("  - matching: Rate each image individually (0-10) for relevance to word sense")
    print("  - description: Generate description for each image, score by word overlap")
    print("\nStrategy:")
    print("  - Process one image at a time to avoid attention dilution")
    print("  - Batch processing for multiple test instances")


if __name__ == "__main__":
    main()
