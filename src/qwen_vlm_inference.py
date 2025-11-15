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


# ========== Prompt Templates ==========
# Four distinct methods for Visual WSD:
# 1. matching: Pure baseline (target_word + context only, no definitions)
# 2. matching_cot: Pure baseline + CoT (target_word + context + reasoning, no definitions)
# 3. description: Definition-based matching (uses sense definition + rating)
# 4. embedding: Direct cosine similarity (no prompts, no definitions)

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

    "description": '''Task: Rate how well this image matches the intended meaning of a word, using the provided visual clue.

Target word: "{target_word}"
Context: "{full_phrase}"
Visual clue: {sense_definition}

Rate how well this image matches the meaning of "{target_word}" in "{full_phrase}".
Use the visual clue as a reference for what to look for.

Rating scale:
0 = completely unrelated or shows a different meaning
5 = somewhat related
10 = perfect match for the intended meaning

Provide only a single number (0-10) as your rating.''',

    "embedding": None  # Embedding method doesn't use prompts
}

MAX_TOKENS = {
    "matching": 50,          # Only need one number
    "matching_cot": 300,     # Reasoning + final rating
    "description": 50,       # Only need one number (definition-based matching)
    "embedding": None        # No generation
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

    def __init__(self, model, processor, device="cuda", definitions=None):
        """
        Initialize inference engine.

        Args:
            model: Loaded Qwen-VL model
            processor: Loaded Qwen-VL processor
            device: Device to run inference on
            definitions: Optional dict mapping (target_word, context) -> definition
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.definitions = definitions or {}

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

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image into embedding using Qwen3-VL's full model forward pass.

        Args:
            image: PIL Image object

        Returns:
            Mean-pooled image embedding vector of shape (hidden_size,)
        """
        # Create a simple text prompt for the image
        # Using a neutral prompt that works for any image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.inference_mode():
            # Forward pass with output_hidden_states to get intermediate features
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # Get the last hidden state from the final layer
            # outputs.hidden_states is a tuple of hidden states from each layer
            # We want the last layer: shape (batch_size, sequence_length, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]

            # Mean pool over the sequence dimension to get a single embedding
            # This pools all visual and text tokens together
            image_embedding = last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)

        return image_embedding.squeeze(0)  # Shape: (hidden_size,)

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

        # Cache for phrase embeddings when using embedding method
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

                # ============ EMBEDDING METHOD: Direct cosine similarity ============
                if method == "embedding":
                    # Format text query same as SigLIP2 for consistency
                    text_query = f"this is a photo of a {target_word}. {full_phrase}".lower()

                    # Encode text query once (cache by query to avoid recomputation)
                    cache_key = (target_word, full_phrase)
                    if cache_key not in phrase_embedding_cache:
                        with torch.inference_mode():
                            phrase_embedding_cache[cache_key] = self.encode_text(text_query)

                    text_embedding = phrase_embedding_cache[cache_key]
                    scores = []

                    # Encode each image and compute cosine similarity
                    for img_idx, img in enumerate(images):
                        image_embedding = self.encode_image(img)

                        # Compute cosine similarity (same as SigLIP2 approach)
                        similarity = F.cosine_similarity(
                            text_embedding.unsqueeze(0),
                            image_embedding.unsqueeze(0)
                        ).item()

                        scores.append(similarity)
                        logger.debug(f"  Image {img_idx+1}/{num_images}: similarity={similarity:.4f}")

                # ============ GENERATIVE METHODS: Prompting + parsing ============
                else:
                    # Route to appropriate prompt based on method
                    definition_key = (target_word, full_phrase)
                    has_definition = definition_key in self.definitions

                    if method == "description":
                        # Description method REQUIRES definitions
                        if not has_definition:
                            raise ValueError(
                                f"Description method requires definitions, but no definition found for "
                                f"instance {global_idx+1} (target_word='{target_word}', context='{full_phrase}'). "
                                f"Please run definition generation first or use a different method."
                            )
                        # Format with definition
                        prompt = PROMPT_TEMPLATES[method].format(
                            target_word=target_word,
                            full_phrase=full_phrase,
                            sense_definition=self.definitions[definition_key]
                        )
                        if global_idx == 0:
                            logger.info(f"Using definition-based prompts (example: '{self.definitions[definition_key][:60]}...')")

                    elif method in ["matching", "matching_cot"]:
                        # Matching methods do NOT use definitions (pure baseline)
                        prompt = PROMPT_TEMPLATES[method].format(
                            target_word=target_word,
                            full_phrase=full_phrase
                        )
                        if global_idx == 0:
                            logger.info(f"Using pure baseline prompts (no definitions)")

                    else:
                        raise ValueError(f"Unknown generative method: {method}")

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

                    # Get max tokens for this method
                    max_tokens = MAX_TOKENS[method]

                    # Generate scores for all candidate images at once
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
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

                        # All generative methods now use VLM rating (0-10)
                        if method in ["matching", "matching_cot", "description"]:
                            score = self._parse_single_image_score(output_text)

                        scores.append(score)
                    logger.debug(f"  Image {img_idx+1}/{num_images}: score={score:.2f}, response='{output_text[:50]}...')")

                    # Explicit memory cleanup for generative methods
                    del inputs, generated_ids, generated_ids_trimmed, output_texts, texts
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Rank by scores (highest first)
                ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
                ranked_filenames = [image_filenames[idx] for idx in ranked_indices]
                ranked_scores = [scores[idx] for idx in ranked_indices]

                results.append((ranked_filenames, ranked_scores))
                logger.debug(f"Instance {global_idx+1} top-3: {ranked_filenames[:3]} with scores {ranked_scores[:3]}")

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
