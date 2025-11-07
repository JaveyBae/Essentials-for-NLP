#!/usr/bin/env python3
"""
Main execution script for Visual Word Sense Disambiguation.
Runs inference and evaluation on VWSD test set.
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import VWSDDataLoader
from src.inference import QwenVLInference, InferenceResult
from models.model_loader import load_model

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


def run_inference(args):
    """Run inference on VWSD test set."""

    # Load model
    logger.info("Loading model...")

    model, processor = load_model(
        model_name=args.model,
        quantization=None,
        device="cuda",
        dtype="bfloat16"
    )

    # Initialize inference engine
    inference_engine = QwenVLInference(model, processor, device="cuda")

    # Load data
    logger.info("Loading test data...")

    data_loader = VWSDDataLoader(data_dir=args.data_dir)
    instances = data_loader.load_test_data(language=args.language)

    logger.info(f"Loaded {len(instances)} test instances")

    # Run inference
    logger.info(f"Running inference with method: {args.method}, batch size: {args.batch_size}")

    results = []

    # Batch processing: process multiple instances at once
    if args.batch_size > 1:
        num_batches = (len(instances) + args.batch_size - 1) // args.batch_size

        with tqdm(total=len(instances), desc="Processing instances") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(instances))
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

    # Save predictions
    logger.info("Saving predictions...")

    # Create output directory
    inference_mode = f"single_img_batch{args.batch_size}"
    output_dir = os.path.join(args.output_dir, f"{args.model}_{args.method}_{inference_mode}")
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

    # Model name for results
    inference_mode = f"single_img_batch{args.batch_size}"
    model_name = f"{args.model}_{args.method}_{inference_mode}"

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
        description="Visual Word Sense Disambiguation with Qwen3-VL"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-8b",
        choices=["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b", "qwen3-vl-32b"],
        help="Qwen3-VL model to use for inference"
    )

    # Inference arguments
    parser.add_argument(
        "--method",
        type=str,
        default="matching",
        choices=["matching", "matching_cot", "description"],
        help="Inference method: 'matching' (0-10 rating), 'matching_cot' (chain-of-thought reasoning + rating), or 'description' (generate description, score by word overlap)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of test instances to process in parallel (each instance has ~10 candidate images evaluated one-by-one)"
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

    args = parser.parse_args()

    # Print configuration
    logger.info("="*60)
    logger.info(f"Model: {args.model} | Method: {args.method} | Language: {args.language} | Batch size: {args.batch_size}")
    logger.info("="*60)

    # Run inference
    prediction_dir = run_inference(args)

    # Run evaluation
    run_evaluation(prediction_dir, args)

    logger.info("All done!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
