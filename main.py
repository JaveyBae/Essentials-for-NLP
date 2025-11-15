#!/usr/bin/env python3
"""
Main execution script for Visual Word Sense Disambiguation.
Supports both SigLIP2 (embedding-based) and VLM (Qwen3-VL) approaches.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import VWSDDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_predictions(results: list, output_file: str):
    """
    Save predictions to file in the required format.

    Format: Each line contains tab-separated ranked image filenames
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            line = '\t'.join(result.ranked_images)
            f.write(line + '\n')

    logger.info(f"Predictions saved to: {output_file}")


def save_detailed_predictions(results: list, output_file: str):
    """
    Save detailed predictions including full_phrase and top-1 image.

    Format: Each line contains: target_word <tab> full_phrase <tab> top1_image_id
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            target_word = result.target_word if result.target_word else "N/A"
            full_phrase = result.full_phrase if result.full_phrase else "N/A"
            top1_image = result.ranked_images[0] if result.ranked_images else "N/A"

            line = f"{target_word}\t{full_phrase}\t{top1_image}"
            f.write(line + '\n')

    logger.info(f"Detailed predictions saved to: {output_file}")


def run_inference_siglip2(args, data_loader, instances):
    """Run SigLIP2 inference (embedding-based ranking)."""
    from src.siglip2_loader import load_siglip2_model
    from src.siglip2_inference import SigLIP2Inference, InferenceResult

    logger.info(f"Loading SigLIP2 model...")

    # SigLIP2 always uses F32 or float16 (no quantization support)
    model, processor = load_siglip2_model(
        model_name=args.siglip2_model,
        quantization=None,  # SigLIP2 doesn't support quantization
        device="cuda"
    )

    # Initialize inference engine
    inference_engine = SigLIP2Inference(model, processor, device="cuda")

    logger.info(f"Running SigLIP2 inference...")

    results = []

    # Process instances one by one
    for instance in tqdm(instances, desc="Processing instances"):
        images = data_loader.load_instance_images(instance)

        instance_data = [{
            'images': images,
            'image_filenames': instance.candidate_images,
            'target_word': instance.target_word,
            'full_phrase': instance.full_phrase
        }]

        batch_results = inference_engine.rank_images(instances_data=instance_data)
        ranked_filenames, scores = batch_results[0]

        result = InferenceResult(
            instance_id=instance.instance_id,
            ranked_images=ranked_filenames,
            scores=scores,
            target_word=instance.target_word,
            full_phrase=instance.full_phrase,
            gold_image=instance.gold_image
        )
        results.append(result)

    return results


def run_inference_vlm(args, data_loader, instances):
    """Run Qwen3-VL inference (generative VLM with prompting)."""
    from src.qwen3_inference import generate_or_load_definitions
    from src.qwen_vlm_inference import InferenceResult

    pipeline_start = time.time()

    # Stage 1: Definition Generation (ONLY for description method)
    definitions = {}
    if args.method == "description":
        logger.info(f"[Stage 1/3] Generating sense definitions with {args.definition_model}...")
        def_start = time.time()

        definitions = generate_or_load_definitions(
            test_instances=instances,
            language=args.language,
            model_name=args.definition_model,
            quantization=args.quantization,  # Use unified quantization parameter
            cache_dir=args.definition_cache_dir,
            force_regenerate=args.regenerate_definitions,
            batch_size=args.definition_batch_size
        )

        def_time = time.time() - def_start
        logger.info(f"Definitions ready: {len(definitions)} definitions in {def_time:.2f}s")
    else:
        logger.info(f"[Stage 1/3] Skipping definition generation (not needed for method '{args.method}')")

    # Stage 2: Visual WSD with Qwen3-VL
    logger.info(f"[Stage 2/3] Loading Qwen3-VL model...")

    from src.qwen_vlm_loader import load_model
    from src.qwen_vlm_inference import QwenVLInference

    model, processor = load_model(
        model_name=args.vlm_model,
        quantization=args.quantization,
        device="cuda"
    )

    # Initialize inference engine with definitions (always enabled)
    inference_engine = QwenVLInference(
        model,
        processor,
        device="cuda",
        definitions=definitions
    )

    # Run VLM inference
    logger.info(f"[Stage 3/3] Running inference (method: {args.method}, batch_size: {args.vlm_batch_size})")

    results = []

    # Batch processing: process multiple instances at once
    if args.vlm_batch_size > 1:
        num_batches = (len(instances) + args.vlm_batch_size - 1) // args.vlm_batch_size

        with tqdm(total=len(instances), desc="Processing instances") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.vlm_batch_size
                end_idx = min(start_idx + args.vlm_batch_size, len(instances))
                batch_instances = instances[start_idx:end_idx]

                # Prepare batch data
                batch_data = []
                for instance in batch_instances:
                    images = data_loader.load_instance_images(instance)
                    batch_data.append({
                        'images': images,
                        'image_filenames': instance.candidate_images,
                        'target_word': instance.target_word,
                        'full_phrase': instance.full_phrase
                    })

                # Process entire batch at once
                batch_results = inference_engine.rank_images(
                    instances_data=batch_data,
                    method=args.method
                )

                # Create results
                for instance, (ranked_filenames, scores) in zip(batch_instances, batch_results):
                    result = InferenceResult(
                        instance_id=instance.instance_id,
                        ranked_images=ranked_filenames,
                        scores=scores,
                        target_word=instance.target_word,
                        full_phrase=instance.full_phrase,
                        gold_image=instance.gold_image
                    )
                    results.append(result)

                pbar.update(len(batch_instances))

    # Single-instance processing (using batch mode with batch_size=1)
    else:
        for instance in tqdm(instances, desc="Processing instances"):
            # Load images
            images = data_loader.load_instance_images(instance)

            # Rank images using batch mode with single instance
            instance_data = [{
                'images': images,
                'image_filenames': instance.candidate_images,
                'target_word': instance.target_word,
                'full_phrase': instance.full_phrase
            }]

            batch_results = inference_engine.rank_images(
                instances_data=instance_data,
                method=args.method
            )

            # Extract single result from batch
            ranked_filenames, scores = batch_results[0]

            # Create result
            result = InferenceResult(
                instance_id=instance.instance_id,
                ranked_images=ranked_filenames,
                scores=scores,
                target_word=instance.target_word,
                full_phrase=instance.full_phrase,
                gold_image=instance.gold_image
            )
            results.append(result)

    # Pipeline summary
    total_time = time.time() - pipeline_start
    logger.info(f"=" * 60)
    logger.info(f"Pipeline complete: {total_time:.2f}s total")
    logger.info(f"=" * 60)

    return results


def run_inference(args):
    """Run inference on VWSD test set (delegates to SigLIP2 or VLM)."""

    # Load data first
    logger.info("Loading test data...")
    data_loader = VWSDDataLoader(data_dir=args.data_dir)
    instances = data_loader.load_test_data(language=args.language)
    logger.info(f"Loaded {len(instances)} test instances")

    # Route to appropriate inference method
    if args.model_type == "siglip2":
        results = run_inference_siglip2(args, data_loader, instances)
    elif args.model_type == "vlm":
        results = run_inference_vlm(args, data_loader, instances)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Save predictions
    # Create output directory based on model type
    if args.model_type == "siglip2":
        # SigLIP2: just model_name (always F32 or float16, no quantization)
        output_dir = os.path.join(
            args.output_dir,
            args.siglip2_model
        )
    else:  # vlm
        # VLM output directory naming:
        # - matching/matching_cot/embedding: baseline (no definitions)
        # - description: definition-based (includes definition model name)
        quant_str = args.quantization

        if args.method == "description":
            # Description method includes definition model
            output_dir = os.path.join(
                args.output_dir,
                f"{args.vlm_model}_{args.method}_{quant_str}_def-{args.definition_model}_batch{args.vlm_batch_size}"
            )
        else:
            # Baseline methods: no definition info
            output_dir = os.path.join(
                args.output_dir,
                f"{args.vlm_model}_{args.method}_{quant_str}_batch{args.vlm_batch_size}"
            )
    os.makedirs(output_dir, exist_ok=True)

    # Save prediction file
    prediction_file = os.path.join(output_dir, f"prediction.{args.language}.txt")
    save_predictions(results, prediction_file)

    # Save detailed predictions (target_word, full_phrase, top1_image)
    detailed_file = os.path.join(output_dir, f"detailed_prediction.{args.language}.txt")
    save_detailed_predictions(results, detailed_file)

    # Calculate quick accuracy
    if results[0].gold_image is not None:
        correct = sum(1 for r in results if r.ranked_images[0] == r.gold_image)
        accuracy = correct / len(results)
        logger.info(f"Accuracy: {correct}/{len(results)} = {accuracy:.4f}")

    return output_dir


def run_evaluation(prediction_dir: str, args):
    """Run evaluation metrics."""

    logger.info("Running evaluation...")

    # Import evaluation script
    import subprocess

    # Model name for results (extract from output_dir name)
    model_name = os.path.basename(prediction_dir)

    # Run evaluation
    eval_cmd = [
        "python", "eval/vwsd_ranking_metric.py",
        "-p", prediction_dir,
        "-d", os.path.join(args.data_dir, "test_data"),
        "-l", args.language,
        "-o", "results/rank_metrics.jsonl",
        "--model-name", model_name
    ]

    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=False, text=True)
        logger.info("Evaluation complete!")

    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Visual Word Sense Disambiguation - Supports SigLIP2 and Qwen3-VL"
    )

    # Model type selection
    parser.add_argument(
        "--model-type",
        type=str,
        default="siglip2",
        choices=["siglip2", "vlm"],
        help="Model type: 'siglip2' (embedding-based, fast) or 'vlm' (Qwen3-VL, generative)"
    )

    # SigLIP2 arguments
    parser.add_argument(
        "--siglip2-model",
        type=str,
        default="siglip2-so400m-patch14-384",
        choices=[
            # Top picks (recommended)
            "siglip2-so400m-patch14-384",        # RECOMMENDED: Most popular, 1B params (476k downloads, 62 likes)
            "siglip2-giant-opt-patch16-384",     # Best quality, 2B params (158k downloads)
            "siglip2-base-patch16-224",          # Fastest, 0.4B params (111k downloads, 77 likes)
            "siglip2-so400m-patch16-512",        # High-res, 1B params (144k downloads)
            # Other variants
            "siglip2-base-patch16-512",
            "siglip2-large-patch16-512",
            "siglip2-so400m-patch14-224",
            "siglip2-so400m-patch16-256",
            "siglip2-so400m-patch16-384",
            "siglip2-giant-opt-patch16-256",
        ],
        help="SigLIP2 model variant (default: siglip2-so400m-patch14-384, most popular)"
    )

    # VLM arguments
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen3-vl-8b",
        choices=["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b"],
        help="Qwen3-VL model to use for visual inference (default: qwen3-vl-8b, only for --model-type vlm)"
    )

    # Quantization parameter (applies to VLM models only: Qwen3-VL and definition generator)
    # Note: SigLIP2 always uses F32 (base/large) or float16 (so400m/giant-opt) and does not support quantization
    parser.add_argument(
        "--quantization",
        type=str,
        default="bfloat16",
        choices=["4bit", "bfloat16"],
        help="VLM model quantization (Qwen3-VL and definition models only): 'bfloat16' (full precision, default) or '4bit' (AWQ-INT4 quantization for lower VRAM). Does not apply to SigLIP2."
    )

    # Inference arguments (VLM only)
    parser.add_argument(
        "--method",
        type=str,
        default="matching",
        choices=["matching", "matching_cot", "description", "embedding"],
        help="VLM inference method: 'matching' (baseline: 0-10 rating, no definitions), 'matching_cot' (baseline + CoT, no definitions), 'description' (definition-based: uses sense definitions + rating, REQUIRES definitions), 'embedding' (direct cosine similarity, no prompts/definitions). Only used with --model-type vlm"
    )

    # Batch size parameter (VLM only)
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=1,
        help="Number of test instances to process in parallel (default: 1)."
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "fa", "it"],
        help="Language to evaluate"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory containing test data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/predictions",
        help="Directory to save predictions"
    )

    # Definition enrichment arguments (VLM only - always enabled)
    parser.add_argument(
        "--definition-model",
        type=str,
        default="qwen3-8b",
        choices=["qwen3-4b", "qwen3-8b", "qwen3-14b"],
        help="Qwen3 text model to use for generating sense definitions (default: qwen3-8b for good quality/VRAM balance, only for --model-type vlm)"
    )

    parser.add_argument(
        "--definition-cache-dir",
        type=str,
        default="results/sense_definitions",
        help="Directory to cache generated definitions (default: results/sense_definitions)"
    )

    parser.add_argument(
        "--regenerate-definitions",
        action="store_true",
        help="Force regenerate definitions (ignore cache)"
    )

    parser.add_argument(
        "--definition-batch-size",
        type=int,
        default=32,
        help="Batch size for definition generation (default: 32)"
    )

    args = parser.parse_args()

    # Print configuration
    if args.model_type == "siglip2":
        # Determine dtype based on model variant
        dtype_str = "F32" if "base" in args.siglip2_model or "large" in args.siglip2_model else "float16"
        config_parts = [
            f"Model: SigLIP2 ({args.siglip2_model})",
            f"Dtype: {dtype_str}",  # SigLIP2 uses F32 or float16, no quantization
            f"Language: {args.language}"
        ]
    else:  # vlm
        config_parts = [
            f"Model: VLM ({args.vlm_model})",
            f"Method: {args.method}",
            f"Quantization: {args.quantization}",
            f"Batch: {args.vlm_batch_size}",
            f"Language: {args.language}"
        ]

        # Only show definition model for description method
        if args.method == "description":
            config_parts.append(f"Definitions: {args.definition_model}")

    logger.info(" | ".join(config_parts))

    # Run inference
    prediction_dir = run_inference(args)

    # Run evaluation
    run_evaluation(prediction_dir, args)


if __name__ == "__main__":
    main()
