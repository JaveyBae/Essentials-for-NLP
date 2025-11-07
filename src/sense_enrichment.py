"""
Sense Enrichment Cache Manager.
Manages persistent caching of word sense definitions for Visual WSD.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefinitionCache:
    """Persistent cache for word sense definitions."""

    def __init__(self, cache_dir: str = "data/sense_definitions"):
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

        Args:
            language: Language code
            model_name: Model name
        """
        cache_path = self.get_cache_path(language, model_name)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
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


def generate_or_load_definitions(
    test_instances: List,
    language: str,
    model_name: str = "qwen3-8b",
    quantization: str = "4bit",
    cache_dir: str = "data/sense_definitions",
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
        quantization: Quantization method ('4bit', '8bit', or None)
        cache_dir: Directory for cache files
        force_regenerate: If True, ignore cache and regenerate all
        batch_size: Batch size for generation

    Returns:
        Dictionary mapping (target_word, full_phrase) -> definition
    """
    from models.definition_generator import Qwen3DefinitionGenerator

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

        with Qwen3DefinitionGenerator(
            model_name=model_name,
            quantization=quantization
        ) as generator:
            generated_defs = generator.generate_batch(
                instances_to_generate,
                batch_size=batch_size,
                show_progress=True
            )

        # Update cache and definitions map
        for inst, definition in zip(instances_to_generate, generated_defs):
            target_word = inst['target_word']
            full_phrase = inst['full_phrase']

            cache.set(target_word, full_phrase, definition, model_name)
            definitions_map[(target_word, full_phrase)] = definition

        # Save updated cache
        cache.save(language, model_name)

        logger.info(f"✓ Generated and cached {len(generated_defs)} new definitions")

        # Auto-export to readable format for inspection
        try:
            export_file = export_definitions_to_readable_format(
                cache_dir, language, model_name, output_format="json"
            )
            if export_file:
                logger.info(f"✓ Definitions also exported to: {export_file}")
        except Exception as e:
            logger.warning(f"Could not export definitions: {e}")
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
    Export cached definitions to human-readable JSON or CSV format.

    Args:
        cache_dir: Cache directory
        language: Language code
        model_name: Model name
        output_format: 'json' or 'csv'

    Returns:
        Path to exported file
    """
    import json
    import csv

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
    """Test cache functionality."""
    from dataclasses import dataclass

    @dataclass
    class MockInstance:
        target_word: str
        full_phrase: str

    print("=" * 80)
    print("Sense Enrichment Cache Manager Test")
    print("=" * 80)

    # Create test instances
    test_instances = [
        MockInstance("bank", "I went to the bank to deposit money"),
        MockInstance("bank", "We sat on the river bank"),
        MockInstance("bass", "He caught a large bass"),
    ]

    # Test cache functionality
    print("\n[Test 1: Generate and Cache Definitions]")
    definitions = generate_or_load_definitions(
        test_instances,
        language="en",
        model_name="qwen3-1.7b",  # Small model for testing
        quantization="4bit",
        cache_dir="data/sense_definitions_test",
        force_regenerate=False,
        batch_size=3
    )

    print(f"\nGenerated {len(definitions)} definitions:")
    for (word, full_phrase), defn in definitions.items():
        print(f"\n  Word: '{word}'")
        print(f"  Full Phrase: {full_phrase}")
        print(f"  Definition: {defn}")

    # Test cache loading
    print("\n[Test 2: Load from Cache]")
    definitions_cached = generate_or_load_definitions(
        test_instances,
        language="en",
        model_name="qwen3-1.7b",
        cache_dir="data/sense_definitions_test",
        force_regenerate=False
    )

    print(f"\nLoaded {len(definitions_cached)} definitions from cache (should be instant)")

    # Test force regenerate
    print("\n[Test 3: Force Regenerate]")
    definitions_regen = generate_or_load_definitions(
        test_instances,
        language="en",
        model_name="qwen3-1.7b",
        cache_dir="data/sense_definitions_test",
        force_regenerate=True,
        batch_size=3
    )

    print(f"\nRegenerated {len(definitions_regen)} definitions")

    print("\n" + "=" * 80)
    print("Cache test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
