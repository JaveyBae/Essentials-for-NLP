#!/usr/bin/env python3
"""
Qwen3 Text Model Inference for Visual WSD Sense Definition Generation.

This module provides:
1. DefinitionCache: Persistent JSON cache for word sense definitions (sorted, human-readable)
2. Qwen3DefinitionGenerator: Generate sense definitions using Qwen3 text models
3. generate_or_load_definitions: Main API for definition generation with caching
4. export_definitions_to_readable_format: Optional utility to export cache to CSV/JSON array format
5. CLI interface: Standalone script for pre-generating definitions

Architecture:
- Uses Qwen3 text models (4B/8B/14B) for generating contextual sense definitions
- Single JSON cache file (sorted by target word for easy inspection)
- Batch generation support for efficiency
- Sequential pipeline: Load model → Generate → Cache → Unload

Usage as library:
    from src.qwen3_inference import generate_or_load_definitions

    # Default: bfloat16 full precision
    definitions = generate_or_load_definitions(
        test_instances=instances,
        language="en",
        model_name="qwen3-8b",
        quantization="bfloat16"
    )

Usage as CLI:
    # Default: bfloat16
    python src/qwen3_inference.py --language en --model qwen3-8b
    # Or with 4-bit quantization
    python src/qwen3_inference.py --language en --model qwen3-8b --quantization 4bit
"""

import argparse
import json
import hashlib
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefinitionCache:
    """Persistent cache for word sense definitions.

    Cache format is a simple JSON hashmap:
    {
        "word_hash": {
            "target_word": "bank",
            "full_phrase": "river bank",
            "definition": "Sloped land beside water",
            "model": "qwen3-8b",
            "timestamp": "2025-11-15T..."
        },
        ...
    }
    """

    def __init__(self, cache_dir: str = "results/sense_definitions"):
        """
        Initialize definition cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, target_word: str, full_phrase: str) -> str:
        """
        Generate unique cache key for word-full_phrase pair.

        Uses hash of full_phrase to handle identical phrases for different words.

        Args:
            target_word: Target ambiguous word
            full_phrase: Full phrase containing the target word

        Returns:
            Unique cache key string
        """
        phrase_hash = hashlib.md5(full_phrase.encode('utf-8')).hexdigest()[:8]
        return f"{target_word}_{phrase_hash}"

    def get_cache_path(self, language: str, model_name: str) -> Path:
        """
        Get cache file path for specific language and model.

        Args:
            language: Language code (e.g., 'en', 'fa', 'it')
            model_name: Model name (e.g., 'qwen3-8b', 'qwen3-14b')

        Returns:
            Path to cache file
        """
        filename = f"{language}_{model_name}.json"
        return self.cache_dir / filename

    def load(self, language: str, model_name: str) -> int:
        """
        Load cache from disk for specific language and model.

        Args:
            language: Language code
            model_name: Model name

        Returns:
            Number of entries loaded
        """
        cache_path = self.get_cache_path(language, model_name)

        if not cache_path.exists():
            logger.info(f"No cache found at {cache_path}")
            return 0

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached definitions from {cache_path}")
            return len(self.cache)
        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {e}")
            self.cache = {}
            return 0

    def save(self, language: str, model_name: str):
        """
        Save cache to disk for specific language and model.

        Saves in a readable JSON format with sorted entries for easy inspection.

        Args:
            language: Language code
            model_name: Model name
        """
        cache_path = self.get_cache_path(language, model_name)

        try:
            # Sort cache by target word for readability
            sorted_cache = dict(sorted(
                self.cache.items(),
                key=lambda x: (x[1].get('target_word', ''), x[1].get('full_phrase', ''))
            ))

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.cache)} definitions to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache to {cache_path}: {e}")

    def get(self, target_word: str, full_phrase: str) -> Optional[str]:
        """
        Retrieve definition from cache.

        Args:
            target_word: Target word
            full_phrase: Full phrase containing the target word

        Returns:
            Cached definition or None if not found
        """
        key = self._get_cache_key(target_word, full_phrase)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]['definition']
        else:
            self.misses += 1
            return None

    def set(
        self,
        target_word: str,
        full_phrase: str,
        definition: str,
        model_name: str
    ):
        """
        Store definition in cache.

        Args:
            target_word: Target word
            full_phrase: Full phrase containing the target word
            definition: Generated definition
            model_name: Model used for generation
        """
        key = self._get_cache_key(target_word, full_phrase)
        self.cache[key] = {
            "target_word": target_word,
            "full_phrase": full_phrase,
            "definition": definition,
            "model": model_name,
            "timestamp": datetime.now().isoformat()
        }

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": self.get_hit_rate()
        }


class Qwen3DefinitionGenerator:
    """Generate word sense definitions using Qwen3 text models (inference only)."""

    DEFINITION_PROMPT_TEMPLATE = """Target word: {target_word}
Context phrase: {full_phrase}
Visual definition (describe what you would SEE in an image):"""

    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize definition generator with pre-loaded model.

        Args:
            model: Loaded Qwen3 model
            tokenizer: Loaded Qwen3 tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_definition(
        self,
        target_word: str,
        full_phrase: str,
        max_tokens: int = 60,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a single definition for a target word in full phrase.

        Args:
            target_word: The ambiguous word
            full_phrase: Sentence/phrase containing the target word
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0 = deterministic, default)

        Returns:
            Generated definition string (cleaned)
        """
        prompt = self.DEFINITION_PROMPT_TEMPLATE.format(
            target_word=target_word,
            full_phrase=full_phrase
        )

        # Format as chat message with few-shot examples
        system_prompt = """You are a visual lexical semantic expert. For Visual Word Sense Disambiguation, provide SHORT, VISUAL, CONCRETE definitions that describe what you would SEE in an image.

CRITICAL RULES:
1. Focus on VISUAL APPEARANCE (what can be seen in an image)
2. Use ALL words from the context phrase, not just the target word
3. Keep it SHORT (5-10 words maximum)
4. Be CONCRETE and SPECIFIC (avoid abstract concepts)
5. Output ONLY the definition, no explanations

Examples:
Target word: bank
Context phrase: river bank
Visual definition: Sloped land beside a river or stream

Target word: bass
Context phrase: bass guitar
Visual definition: Large, low-pitched electric guitar with four strings

Target word: goal
Context phrase: football goal
Visual definition: Netted frame where players score in soccer

Target word: neptune
Context phrase: neptune statue
Visual definition: Stone or bronze statue of the Roman sea god

Target word: bank
Context phrase: money bank
Visual definition: Building where people deposit and withdraw money"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Tokenize
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        return generated_text

    def generate_batch(
        self,
        instances: List[Dict[str, str]],
        batch_size: int = 32,
        max_tokens: int = 60,
        temperature: float = 0.0,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate definitions for multiple instances in batches.

        Args:
            instances: List of dicts with 'target_word' and 'full_phrase' keys
            batch_size: Number of instances to process at once
            max_tokens: Maximum tokens per definition
            temperature: Generation temperature (0 = deterministic, default)
            show_progress: Show progress bar

        Returns:
            List of generated definitions (same order as input)
        """
        definitions = []
        total_batches = (len(instances) + batch_size - 1) // batch_size

        iterator = range(0, len(instances), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="Generating definitions"
            )

        for i in iterator:
            batch = instances[i:i + batch_size]

            # Create prompts for batch
            prompts = [
                self.DEFINITION_PROMPT_TEMPLATE.format(
                    target_word=inst['target_word'],
                    full_phrase=inst['full_phrase']
                )
                for inst in batch
            ]

            # Format as chat messages
            system_prompt = """You are a visual lexical semantic expert. For Visual Word Sense Disambiguation, provide SHORT, VISUAL, CONCRETE definitions that describe what you would SEE in an image.

CRITICAL RULES:
1. Focus on VISUAL APPEARANCE (what can be seen in an image)
2. Use ALL words from the context phrase, not just the target word
3. Keep it SHORT (5-10 words maximum)
4. Be CONCRETE and SPECIFIC (avoid abstract concepts)
5. Output ONLY the definition, no explanations

Examples:
Target word: bank
Context phrase: river bank
Visual definition: Sloped land beside a river or stream

Target word: bass
Context phrase: bass guitar
Visual definition: Large, low-pitched electric guitar with four strings

Target word: goal
Context phrase: football goal
Visual definition: Netted frame where players score in soccer

Target word: neptune
Context phrase: neptune statue
Visual definition: Stone or bronze statue of the Roman sea god

Target word: bank
Context phrase: money bank
Visual definition: Building where people deposit and withdraw money"""

            messages_list = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ]
                for p in prompts
            ]
            texts = [
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for msgs in messages_list
            ]

            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate batch
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                if temperature == 0.0:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9

                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode batch results
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_text = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()

                definitions.append(generated_text)

        return definitions


def generate_or_load_definitions(
    test_instances: List,
    language: str,
    model_name: str = "qwen3-8b",
    quantization: str = "bfloat16",
    cache_dir: str = "results/sense_definitions",
    force_regenerate: bool = False,
    batch_size: int = 16
) -> Dict[str, str]:
    """
    Generate or load sense definitions for test instances.

    This function manages the entire definition generation pipeline:
    1. Load cache if available (unless force_regenerate=True)
    2. Generate missing definitions using Qwen3
    3. Save updated cache
    4. Cleanup model from GPU

    Args:
        test_instances: List of test instances with target_word and full_phrase
        language: Language code ('en', 'fa', 'it')
        model_name: Qwen3 model to use
        quantization: Quantization method ('4bit' for AWQ-INT4, 'bfloat16' for full precision, default: 'bfloat16')
        cache_dir: Directory for cache files
        force_regenerate: If True, ignore cache and regenerate all
        batch_size: Batch size for generation

    Returns:
        Dictionary mapping (target_word, full_phrase) -> definition
    """
    from src.qwen3_loader import load_qwen3_model

    # Initialize cache
    cache = DefinitionCache(cache_dir)

    # Load existing cache unless force regenerate
    if not force_regenerate:
        cache.load(language, model_name)

    # Collect instances that need definitions
    instances_to_generate = []
    definitions_map = {}

    for instance in test_instances:
        target_word = instance.target_word
        full_phrase = instance.full_phrase

        if not force_regenerate:
            # Try to get from cache
            cached_def = cache.get(target_word, full_phrase)
            if cached_def:
                definitions_map[(target_word, full_phrase)] = cached_def
                continue

        # Need to generate
        instances_to_generate.append({
            "target_word": target_word,
            "full_phrase": full_phrase
        })

    # Report cache statistics
    if not force_regenerate:
        stats = cache.get_stats()
        logger.info(
            f"Cache stats: {stats['cache_hits']} hits, {stats['cache_misses']} misses, "
            f"hit rate: {stats['hit_rate']:.1%}"
        )

    # Generate missing definitions if any
    if instances_to_generate:
        logger.info(
            f"Generating {len(instances_to_generate)} definitions using {model_name}..."
        )

        # Load Qwen3 model
        model, tokenizer = load_qwen3_model(
            model_name=model_name,
            quantization=quantization
        )

        # Create generator and generate definitions
        generator = Qwen3DefinitionGenerator(model, tokenizer)
        generated_defs = generator.generate_batch(
            instances_to_generate,
            batch_size=batch_size,
            show_progress=True
        )

        # Aggressive cleanup to free GPU memory before VLM loading
        logger.info("Cleaning up definition model from GPU...")

        # Delete references
        del generator
        del model
        del tokenizer

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache multiple times (helps with fragmentation)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.empty_cache()  # Clear again after sync

        # Log GPU memory after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"  GPU Memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        logger.info("✓ Definition model cleaned up from GPU")

        # Update cache and definitions map
        for inst, definition in zip(instances_to_generate, generated_defs):
            target_word = inst['target_word']
            full_phrase = inst['full_phrase']

            cache.set(target_word, full_phrase, definition, model_name)
            definitions_map[(target_word, full_phrase)] = definition

        # Save updated cache
        cache.save(language, model_name)

        logger.info(f"✓ Generated and cached {len(generated_defs)} new definitions")
    else:
        logger.info("✓ All definitions loaded from cache (no generation needed)")

    return definitions_map


def export_definitions_to_readable_format(
    cache_dir: str,
    language: str,
    model_name: str,
    output_format: str = "json"
) -> str:
    """
    OPTIONAL: Export cached definitions to alternative formats (JSON array or CSV).

    NOTE: The main cache file is already human-readable (sorted JSON hashmap).
    This function is only needed if you want array format or CSV export.

    Args:
        cache_dir: Cache directory
        language: Language code
        model_name: Model name
        output_format: 'json' or 'csv'

    Returns:
        Path to exported file
    """
    # Load cache
    cache = DefinitionCache(cache_dir)
    num_loaded = cache.load(language, model_name)

    if num_loaded == 0:
        logger.warning(f"No definitions found for {language}/{model_name}")
        return None

    # Prepare export directory
    export_dir = Path(cache_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        # Export as readable JSON with target words as keys
        output_file = export_dir / f"{language}_{model_name}_readable.json"

        # Convert cache to readable format
        readable_data = []
        for key, entry in cache.cache.items():
            readable_data.append({
                "target_word": entry["target_word"],
                "full_phrase": entry.get("full_phrase", entry.get("context", "")),  # Handle both old and new format
                "definition": entry["definition"],
                "model": entry["model"],
                "timestamp": entry["timestamp"]
            })

        # Sort by target word for easier browsing
        readable_data.sort(key=lambda x: (x["target_word"], x["full_phrase"]))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Exported {len(readable_data)} definitions to {output_file}")

    elif output_format == "csv":
        # Export as CSV
        output_file = export_dir / f"{language}_{model_name}_readable.csv"

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["target_word", "full_phrase", "definition", "model", "timestamp"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            # Sort by target word
            sorted_entries = sorted(
                cache.cache.values(),
                key=lambda x: (x["target_word"], x.get("full_phrase", x.get("context", "")))
            )

            for entry in sorted_entries:
                writer.writerow({
                    "target_word": entry["target_word"],
                    "full_phrase": entry.get("full_phrase", entry.get("context", "")),  # Handle both old and new format
                    "definition": entry["definition"],
                    "model": entry["model"],
                    "timestamp": entry["timestamp"]
                })

        logger.info(f"✓ Exported {len(cache.cache)} definitions to {output_file}")

    else:
        raise ValueError(f"Unsupported format: {output_format}. Choose 'json' or 'csv'")

    return str(output_file)


def main():
    """CLI interface for standalone definition generation."""
    parser = argparse.ArgumentParser(
        description="Generate sense definitions for VWSD test data (always regenerates)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for English (default: qwen3-14b, bf16, batch 32)
  python src/qwen3_inference.py --language en

  # Use smaller model with 4-bit quantization
  python src/qwen3_inference.py --language en --model qwen3-8b --quantization 4bit

  # Test mode (first 5 instances only)
  python src/qwen3_inference.py --language en --model qwen3-8b --test-mode

Note: This script ALWAYS regenerates definitions, never loads from cache.
        """
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "fa", "it"],
        help="Language to process"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-14b",
        choices=["qwen3-4b", "qwen3-8b", "qwen3-14b"],
        help="Qwen3 model to use (default: qwen3-14b)"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="bfloat16",
        choices=["4bit", "bfloat16"],
        help="Quantization: 4bit (AWQ-INT4, VRAM-efficient) or bfloat16 (full precision, default)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sense_definitions",
        help="Output directory for cache and exports (default: results/sense_definitions)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)"
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: only process first 5 instances for quick testing"
    )

    args = parser.parse_args()

    logger.info(f"Sense Definition Generator - Language: {args.language} | Model: {args.model} | Quantization: {args.quantization} | Batch: {args.batch_size}")
    logger.info(f"NOTE: Cache will be ignored, definitions will be regenerated")

    # Load data and generate definitions
    logger.info("Loading test data...")

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import VWSDDataLoader

    data_loader = VWSDDataLoader(data_dir=args.data_dir)
    instances = data_loader.load_test_data(language=args.language)

    # Test mode: only process first 5 instances
    if args.test_mode:
        instances = instances[:5]
        logger.info(f"TEST MODE: Processing only first {len(instances)} instances")
    else:
        logger.info(f"Loaded {len(instances)} test instances")

    logger.info("Generating definitions (ignoring cache)...")
    definitions = generate_or_load_definitions(
        test_instances=instances,
        language=args.language,
        model_name=args.model,
        quantization=args.quantization,
        cache_dir=args.output_dir,
        force_regenerate=True,  # Always force regenerate
        batch_size=args.batch_size
    )

    logger.info(f"✓ Total definitions: {len(definitions)}")

    cache_file = f"{args.output_dir}/{args.language}_{args.model}.json"
    logger.info(f"✓ Definitions saved to: {cache_file}")
    logger.info(f"   (File is human-readable JSON sorted by target word)")

    # Print sample definitions
    logger.info("\nSample definitions (first 5):")
    for i, ((word, full_phrase), definition) in enumerate(list(definitions.items())[:5]):
        logger.info(f"  {i+1}. '{word}' in '{full_phrase}' → {definition}")


if __name__ == "__main__":
    main()
