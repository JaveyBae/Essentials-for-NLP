#!/usr/bin/env python3
"""
Standalone script to generate sense definitions for VWSD test data.

This script:
1. Loads test instances from VWSD data
2. Generates contextual sense definitions using Qwen3 text models
3. ALWAYS regenerates (never loads from cache)
4. Exports to human-readable JSON/CSV format

Usage:
    # Generate definitions for English test set (uses qwen3-14b, bf16 by default)
    python scripts/generate_definitions.py --language en

    # Use smaller/faster model with 4-bit quantization
    python scripts/generate_definitions.py --language en --model qwen3-8b --quantization 4bit

    # Process all languages
    python scripts/generate_definitions.py --language en
    python scripts/generate_definitions.py --language fa
    python scripts/generate_definitions.py --language it
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import VWSDDataLoader
from src.sense_enrichment import generate_or_load_definitions, export_definitions_to_readable_format
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sense definitions for VWSD test data (always regenerates)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for English (default: qwen3-14b, bf16, batch 32)
  python scripts/generate_definitions.py --language en

  # Use smaller model with 4-bit quantization
  python scripts/generate_definitions.py --language en --model qwen3-8b --quantization 4bit

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
        default="bf16",
        choices=["4bit", "bf16"],
        help="Quantization: 4bit (VRAM-efficient) or bf16 (higher quality, default)"
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

    # Export to readable JSON format
    logger.info("Exporting to JSON...")
    output_file = export_definitions_to_readable_format(
        cache_dir=args.output_dir,
        language=args.language,
        model_name=args.model,
        output_format="json"
    )

    logger.info(f"✓ Cache saved to: {args.output_dir}/{args.language}_{args.model}.json")
    logger.info(f"✓ Human-readable export: {output_file}")

    # Print sample definitions
    logger.info("\nSample definitions (first 3):")
    for i, ((word, full_phrase), definition) in enumerate(list(definitions.items())[:3]):
        logger.info(f"{i+1}. '{word}' in '{full_phrase}' → {definition}")


if __name__ == "__main__":
    main()
