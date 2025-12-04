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

    # Stage 1: Load definitions if requested
    definitions = {}
    if args.use_definitions:
        from src.qwen3_inference import generate_or_load_definitions

        logger.info(f"[Stage 1/2] Loading/generating sense definitions with {args.definition_model}...")

        definitions = generate_or_load_definitions(
            test_instances=instances,
            language=args.language,
            model_name=args.definition_model,
            quantization=args.quantization,
            cache_dir=args.definition_cache_dir,
            force_regenerate=args.regenerate_definitions,
            batch_size=args.definition_batch_size
        )

        logger.info(f"Definitions ready: {len(definitions)} definitions")
    else:
        logger.info(f"[Stage 1/2] Skipping definition generation (use --use-definitions to enable)")

    # Stage 2: Load SigLIP2 model and run inference
    if args.siglip2_finetuned:
        logger.info(f"[Stage 2/2] Loading fine-tuned SigLIP2 model from {args.siglip2_finetuned}...")
    else:
        logger.info(f"[Stage 2/2] Loading SigLIP2 model...")

    # SigLIP2 always uses F32 or float16 (no quantization support)
    model, processor = load_siglip2_model(
        model_name=args.siglip2_model,
        quantization=None,  # SigLIP2 doesn't support quantization
        device="cuda",
        local_model_path=args.siglip2_finetuned,
    )

    # Initialize inference engine
    inference_engine = SigLIP2Inference(model, processor, device="cuda")

    prompt_type = "with definitions" if args.use_definitions else "default"
    logger.info(f"Running SigLIP2 inference ({prompt_type})...")

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

        batch_results = inference_engine.rank_images(
            instances_data=instance_data,
            definitions=definitions if args.use_definitions else None,
            use_definitions=args.use_definitions,
            prompt_template=args.prompt_template
        )
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


def run_inference_cascade(args, data_loader, instances):
    """Run two-stage cascade inference (SigLIP2 → VLM reranking)."""
    from src.cascade_reranker import CascadeReranker, CascadeResult

    # Load definitions if using description method
    definitions = {}
    if args.reranker_method == "description":
        from src.qwen3_inference import generate_or_load_definitions

        logger.info(f"Loading/generating sense definitions with {args.definition_model}...")
        definitions = generate_or_load_definitions(
            test_instances=instances,
            language=args.language,
            model_name=args.definition_model,
            quantization=args.quantization,
            cache_dir=args.definition_cache_dir,
            force_regenerate=args.regenerate_definitions,
            batch_size=args.definition_batch_size
        )
        logger.info(f"Definitions ready: {len(definitions)} definitions")

    # Initialize cascade reranker
    reranker = CascadeReranker(
        siglip2_model=args.siglip2_model,
        vlm_model=args.vlm_model,
        vlm_method=args.reranker_method,
        vlm_quantization=args.quantization,
        top_k=args.topk,
        device="cuda",
        definitions=definitions
    )

    # Process all instances
    cascade_results = reranker.process_instances(
        instances=instances,
        data_loader=data_loader,
        show_progress=True
    )

    # Convert CascadeResult to InferenceResult format for compatibility
    from src.siglip2_inference import InferenceResult
    results = []
    for cr in cascade_results:
        result = InferenceResult(
            instance_id=cr.instance_id,
            ranked_images=cr.ranked_images,
            scores=cr.scores,
            target_word=cr.target_word,
            full_phrase=cr.full_phrase,
            gold_image=cr.gold_image
        )
        results.append(result)

    # Cleanup
    reranker.cleanup()

    return results


def run_inference(args):
    """Run inference on VWSD test set (delegates to SigLIP2, VLM, or Cascade)."""

    # Load data first
    logger.info("Loading test data...")
    data_loader = VWSDDataLoader(data_dir=args.data_dir)
    instances = data_loader.load_test_data(language=args.language)
    logger.info(f"Loaded {len(instances)} test instances")

    # Route to appropriate inference method
    if args.cascade:
        results = run_inference_cascade(args, data_loader, instances)
    elif args.model_type == "siglip2":
        results = run_inference_siglip2(args, data_loader, instances)
    elif args.model_type == "vlm":
        results = run_inference_vlm(args, data_loader, instances)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Save predictions
    # Create output directory based on model type
    if args.cascade:
        # Cascade: siglip2_model + vlm_model + method + topk
        output_dir = os.path.join(
            args.output_dir,
            f"cascade_{args.siglip2_model}_{args.vlm_model}_{args.reranker_method}_top{args.topk}"
        )
    elif args.model_type == "siglip2":
        # SigLIP2: model_name + optional definition info + optional finetuned indicator
        if args.siglip2_finetuned:
            # Extract model name from path (e.g., "best_model" from ".../best_model/")
            model_dir_name = os.path.basename(args.siglip2_finetuned.rstrip('/'))
            siglip2_name = f"siglip2_finetuned_{model_dir_name}"
        else:
            siglip2_name = args.siglip2_model

        if args.use_definitions:
            output_dir = os.path.join(
                args.output_dir,
                f"{siglip2_name}_def-{args.definition_model}_{args.prompt_template}"
            )
        else:
            output_dir = os.path.join(
                args.output_dir,
                siglip2_name
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

    # Cascade reranking arguments
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Enable two-stage cascade: SigLIP2 ranks all 10 images, VLM reranks top-K. Combines speed of SigLIP2 with accuracy of VLM."
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top candidates to rerank with VLM in cascade mode (default: 3). Higher = more accurate but slower."
    )

    parser.add_argument(
        "--reranker-method",
        type=str,
        default="matching_cot",
        choices=["matching", "matching_cot", "description"],
        help="VLM method for Stage 2 reranking in cascade mode (default: matching_cot). 'description' requires definitions."
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

    parser.add_argument(
        "--siglip2-finetuned",
        type=str,
        default=None,
        help="Path to fine-tuned SigLIP2 model directory (overrides --siglip2-model)"
    )

    parser.add_argument(
        "--use-definitions",
        action="store_true",
        help="Use sense definitions in SigLIP2 prompts. Definitions are generated/loaded from cache using --definition-model."
    )

    parser.add_argument(
        "--prompt-template",
        type=str,
        default="with_definition",
        choices=["with_definition", "definition_only", "target_definition"],
        help="SigLIP2 prompt template when using definitions: 'with_definition' (default: word + phrase + def), 'definition_only' (just def), 'target_definition' (word + def, no phrase)"
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
        choices=["matching", "matching_cot", "description", "embedding", "caption"],
        help="VLM inference method: 'matching' (baseline: 0-10 rating, no definitions), 'matching_cot' (baseline + CoT, no definitions), 'description' (definition-based: uses sense definitions + rating, REQUIRES definitions), 'embedding' (direct cosine similarity, no prompts/definitions), 'caption' (image-to-text: VLM generates caption, Sentence-BERT computes similarity). Only used with --model-type vlm"
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
    if args.cascade:
        # Cascade mode: SigLIP2 → VLM reranking
        config_parts = [
            "Mode: CASCADE",
            f"Stage1: SigLIP2 ({args.siglip2_model})",
            f"Stage2: VLM ({args.vlm_model}, {args.reranker_method})",
            f"Top-K: {args.topk}",
            f"Language: {args.language}"
        ]
        if args.reranker_method == "description":
            config_parts.append(f"Definitions: {args.definition_model}")
    elif args.model_type == "siglip2":
        # Determine dtype based on model variant
        dtype_str = "F32" if "base" in args.siglip2_model or "large" in args.siglip2_model else "float16"
        if args.siglip2_finetuned:
            config_parts = [
                f"Model: SigLIP2 (fine-tuned from {args.siglip2_finetuned})",
                f"Language: {args.language}"
            ]
        else:
            config_parts = [
                f"Model: SigLIP2 ({args.siglip2_model})",
                f"Dtype: {dtype_str}",  # SigLIP2 uses F32 or float16, no quantization
                f"Language: {args.language}"
            ]
        if args.use_definitions:
            config_parts.append(f"Definitions: {args.definition_model}")
            config_parts.append(f"Template: {args.prompt_template}")
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
